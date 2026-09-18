/* CropSSL mobile PWA — offline k-NN / nearest-centroid classifier.
 *
 * Pairs an exported backbone ONNX (input 'input' → output 'features',
 * produced by utils/export.export_ssl_backbone) with a k-NN bundle JSON
 * (POST /models/{name}/knn-bundle). Everything runs on-device:
 * onnxruntime-web (WASM) computes embeddings; classify() does cosine
 * matching against class centroids — the same math as
 * scripts/onnx_knn.py's nearest_centroid().
 *
 * Storage: Cache Storage API ('cropssl-knn-v1') holds the pinned ORT
 * runtime, the .onnx, and the bundle JSON so a field device with no
 * connectivity can still run inference after a one-time download.
 */
(function (global) {
  "use strict";

  const CACHE_NAME = "cropssl-knn-v1";
  // Immutable pinned runtime (version path never changes upstream).
  const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
  const ORT_SCRIPT = ORT_BASE + "ort.min.js";

  function emptyState() {
    return { ready: false, reason: "no bundle", bundle: null, session: null, numRuns: 0 };
  }
  const state = emptyState();

  function resetState(reason) {
    // state is a const: mutate fields, never reassign
    Object.assign(state, emptyState());
    state.reason = reason || state.reason;
  }

  /* ---------- cache helpers ---------- */

  async function cachePut(url, resp) {
    const c = await caches.open(CACHE_NAME);
    await c.put(url, resp);
  }

  async function cacheGet(url) {
    const c = await caches.open(CACHE_NAME);
    return c.match(url);
  }

  async function cachedOrFetch(url) {
    const hit = await cacheGet(url);
    if (hit) return hit;
    const resp = await fetch(url);
    if (resp && resp.ok) await cachePut(url, resp.clone());
    return resp;
  }

  /* ---------- public status ---------- */

  function status() {
    return {
      ready: state.ready,
      reason: state.reason,
      model: state.bundle ? state.bundle.model_name : null,
      mode: state.bundle ? state.bundle.mode : null,
      classes: state.bundle ? state.bundle.num_classes : 0,
      support: state.bundle ? state.bundle.support.embeddings.length : 0,
      runs: state.numRuns,
    };
  }

  /* ---------- load / install ---------- */

  async function load(apiBase, modelName) {
    state.ready = false;
    state.reason = "loading";
    try {
      // 1) pinned runtime (cache-first so it works offline after install)
      if (!global.ort) {
        await cachedOrFetch(ORT_SCRIPT);
        await new Promise((resolve, reject) => {
          const s = document.createElement("script");
          s.src = ORT_SCRIPT;
          s.onload = resolve;
          s.onerror = () => reject(new Error("runtime load failed"));
          document.head.appendChild(s);
        });
      }
      if (!global.ort) throw new Error("onnxruntime-web unavailable");

      // 2) bundle JSON
      const bURL = apiBase + "/models/" + encodeURIComponent(modelName) + "/knn-bundle";
      const bResp = await cachedOrFetch(bURL);
      if (!bResp.ok) throw new Error("bundle " + bResp.status);
      const bundle = await bResp.json();

      // 3) the ONNX paired with the bundle (same export the server embedded with)
      const oURL = apiBase + "/models/" + encodeURIComponent(modelName) + "/export";
      const oResp = await cachedOrFetch(oURL);
      if (!oResp.ok) throw new Error("onnx " + oResp.status);
      const buf = await oResp.arrayBuffer();

      // 4) session (WASM backend — pure CPU, no server)
      state.session = await global.ort.InferenceSession.create(
        new Uint8Array(buf),
        { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
      );
      state.bundle = bundle;
      state.ready = true;
      state.reason = "ok";
      return status();
    } catch (e) {
      resetState((e && e.message) || "load failed");
      return status();
    }
  }

  /* ---------- image preprocessing (the bundle's contract) ---------- */

  function preprocess(img, size, mean, std) {
    // cover-crop to size×size, NHWC→NCHW float32, then ImageNet normalize.
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const scale = Math.max(size / img.naturalWidth, size / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
    const px = ctx.getImageData(0, 0, size, size).data; // RGBA
    const data = new Float32Array(3 * size * size);
    const plane = size * size;
    for (let i = 0; i < plane; i++) {
      data[i] = (px[i * 4] / 255 - mean[0]) / std[0];
      data[plane + i] = (px[i * 4 + 1] / 255 - mean[1]) / std[1];
      data[2 * plane + i] = (px[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
    return new global.ort.Tensor("float32", data, [1, 3, size, size]);
  }

  /* ---------- classification (cosine to centroids / k-NN vote) ---------- */

  function l2normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1e-8;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  function embed(img) {
    const b = state.bundle;
    const input = preprocess(img, b.image_size, b.normalize.mean, b.normalize.std);
    const inputName = state.session.inputNames[0] || "input";
    return state.session.run({ [inputName]: input }).then((out) => {
      const t = out[state.session.outputNames[0]] || out.features || Object.values(out)[0];
      return t.data; // Float32Array (already flat for the backbone flavor)
    });
  }

  function classifySync(feat) {
    const b = state.bundle;
    const cls = b.classes && b.classes.length === b.num_classes
      ? b.classes
      : Array.from({ length: b.num_classes }, (_, i) => "Class " + i);
    const f = l2normalize(Float32Array.from(feat));

    let scores; // per-class score in [0,1]
    if (b.k > 0) {
      // k-NN vote over support embeddings; score = votes for the winner / k
      const sims = b.support.embeddings.map((row, i) => ({
        label: b.support.labels[i],
        cos: cosine(f, Float32Array.from(row)),
      }));
      sims.sort((a, c) => c.cos - a.cos);
      const votes = {};
      sims.slice(0, b.k).forEach((s) => { votes[s.label] = (votes[s.label] || 0) + 1; });
      let best = -1, bestV = 0;
      for (const [label, v] of Object.entries(votes)) {
        if (v > bestV) { bestV = v; best = Number(label); }
      }
      scores = cls.map((_, i) => (i === best ? bestV / b.k : 0));
      return { index: best, label: cls[best], scores };
    }

    // nearest centroid (server bakes raw-mean centroids; L2 at classify time).
    // Confidence = softmax over cosine sims (T=0.07): raw cosine between
    // real embeddings lives in a narrow band, so (s+1)/2-style mapping
    // saturates at ~100% for every class — softmax keeps relative ordering
    // honest and only peaks when the winner is clearly closer.
    const cents = b.centroids.map((c) => l2normalize(Float32Array.from(c)));
    const sims = cents.map((c) => cosine(f, c));
    let best = 0;
    for (let i = 1; i < sims.length; i++) if (sims[i] > sims[best]) best = i;
    const T = 0.07;
    const exps = sims.map((s) => Math.exp((s - sims[best]) / T));
    const sum = exps.reduce((a, v) => a + v, 0) || 1;
    scores = exps.map((v) => v / sum); // proper distribution in [0,1]
    return { index: best, label: cls[best], scores };
  }

  function cosine(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

  async function classify(img) {
    if (!state.ready) return { ok: false, error: state.reason };
    const t0 = performance.now();
    try {
      const feat = await embed(img);
      const res = classifySync(feat);
      state.numRuns++;
      // Server-shaped response so app.js showResult() renders it unchanged:
      // prediction + confidence (0-100) + top_5 [{class, confidence}].
      // Sort on RAW scores (rounding first can tie and mis-order), then round.
      const cls = state.bundle.classes && state.bundle.classes.length === state.bundle.num_classes
        ? state.bundle.classes
        : Array.from({ length: state.bundle.num_classes }, (_, i) => "Class " + i);
      const order = res.scores
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s || a.i - b.i)
        .slice(0, 5);
      const top = order.map((o) => ({
        "class": cls[o.i],
        confidence: Math.round(o.s * 1000) / 10,
      }));
      return {
        ok: true,
        prediction: res.label,
        classIndex: res.index,
        confidence: top.length ? top[0].confidence : 0,
        top_5: top,
        inference_time_ms: performance.now() - t0,
        model_used: state.bundle.model_name + " (offline k-NN)",
        mode: state.bundle.mode,
      };
    } catch (e) {
      return { ok: false, error: (e && e.message) || "inference failed" };
    }
  }

  /* clear cached runtime + model + bundle */
  async function reset() {
    await caches.delete(CACHE_NAME);
    state.ready = false;
    state.bundle = null;
    state.session = null;
    state.reason = "no bundle";
  }

  global.CropSSLKNN = { load, classify, status, reset, emptyState };
})(window);
