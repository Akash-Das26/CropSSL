/* CropSSL mobile PWA — app logic */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ls = window.localStorage;
  const apiDefault =
    ls.getItem("cropssl_api") ||
    window.location.origin.replace(/:\d+$/, "") + ":8000";
  const API = () => $("apiBase").value.trim().replace(/\/+$/, "");

  const TABS = {
    scan: () => {
      $("tabScan").classList.add("active");
      $("tabStatus").classList.remove("active");
      $("landing").classList.remove("hidden");
      $("status").classList.add("hidden");
    },
    status: () => {
      $("tabStatus").classList.add("active");
      $("tabScan").classList.remove("active");
      $("status").classList.remove("hidden");
      $("landing").classList.add("hidden");
      loadStatus();
    },
  };

  function setConn(state) {
    const el = $("conn");
    el.className = "conn " + state;
    $("connTxt").textContent =
      state === "online" ? "ONLINE" : state === "offline" ? "OFFLINE" : "CONNECTING";
  }

  async function ping() {
    try {
      const r = await fetch(API() + "/health", { signal: AbortSignal.timeout(4000) });
      setConn(r.ok ? "online" : "offline");
      return r.ok;
    } catch (_) {
      setConn("offline");
      return false;
    }
  }

  function showResult(res) {
    $("analyzing").classList.add("hidden");
    const card = $("resultCard");
    card.classList.remove("hidden");

    const top = res.prediction || "Unknown";
    const conf = Math.round((res.confidence || 0) * 100) / 100;
    $("topLabel").textContent = top;
    $("confPct").textContent = conf.toFixed(0) + "%";
    $("subLabel").textContent = top.toLowerCase().includes("healthy")
      ? "Plant appears healthy 🟢"
      : "Disease detected — treat promptly";
    $("ring").style.setProperty("--p", Math.min(conf, 100));
    $("chipModel").textContent = res.model_used || "model";
    $("chipTime").textContent = (res.inference_time_ms || 0).toFixed(0) + " ms";

    const bars = $("bars");
    bars.innerHTML = "";
    const list = Array.isArray(res.top_5) ? res.top_5 : [];
    const maxConf = Math.max(conf, ...list.map((t) => t.confidence || 0), 1);
    list.slice(0, 5).forEach((t, i) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        '<div class="lbl">' + (i + 1) + ". " + esc(t.class || "?") + "</div>" +
        '<div class="pct">' + (t.confidence || 0).toFixed(1) + "%</div>";
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      const w = Math.max(3, ((t.confidence || 0) / maxConf) * 100);
      track.appendChild(fill);
      const holder = document.createElement("div");
      holder.style.gridColumn = "1 / -1";
      holder.appendChild(track);
      row.appendChild(holder);
      bars.appendChild(row);
      requestAnimationFrame(() => (fill.style.width = w + "%"));
    });
    if (!list.length) {
      bars.innerHTML = '<p class="hint">No top-5 detail returned.</p>';
    }
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function refreshModelList() {
    fetch(API() + "/models")
      .then((r) => r.json())
      .then((list) => {
        const sel = $("modelSelect");
        if (!sel || !Array.isArray(list)) return;
        const cur = sel.value;
        sel.innerHTML =
          '<option value="">(server active model)</option>' +
          list
            .map((m) => {
              const arch = (m.architecture || "SSL").toUpperCase();
              return '<option value="' + esc(m.name) + '">' +
                esc(m.name) + " · " + arch + "</option>";
            })
            .join("");
        if (cur) sel.value = cur;
      })
      .catch(() => {});
  }

  async function analyze(file) {
    const img = $("preview");
    img.src = URL.createObjectURL(file);
    $("imgTag").textContent = "PROCESSING";
    $("resultCard").classList.add("hidden");
    $("analyzing").classList.remove("hidden");

    const showOfflineError = (msg) => {
      $("imgTag").textContent = "ERROR";
      $("analyzing").classList.add("hidden");
      $("resultCard").classList.remove("hidden");
      $("topLabel").textContent = "Connection failed";
      $("confPct").textContent = "—";
      $("subLabel").textContent = msg;
      $("bars").innerHTML = "";
      setConn("offline");
    };

    try {
      const fd = new FormData();
      fd.append("file", file, "leaf.jpg");
      const model = ($("modelSelect") || {}).value || "";
      const q = model ? "?model_name=" + encodeURIComponent(model) : "";
      const r = await fetch(API() + "/predict" + q, { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error("Server " + r.status + " " + t.slice(0, 140));
      }
      const res = await r.json();
      $("imgTag").textContent = "ANALYZED";
      showResult(res);
    } catch (e) {
      // Offline fallback: classify on-device with the cached k-NN bundle.
      const knn = window.CropSSLKNN;
      if (knn && knn.status().ready) {
        try { await img.decode(); } catch (_) { /* decode best-effort */ }
        const res = await knn.classify(img);
        if (res.ok) {
          $("imgTag").textContent = "ANALYZED · OFFLINE";
          showResult(res);
          return;
        }
        showOfflineError("Offline k-NN failed: " + res.error);
        return;
      }
      showOfflineError(e.message);
    }
  }

  function loadStatus() {
    const hl = $("healthList");
    hl.innerHTML = "<li><span class='k'>Loading…</span></li>";
    fetch(API() + "/health")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((d) => {
        hl.innerHTML =
          li("Engine", d.status, "ok") +
          li("Device", d.device || "cpu") +
          li("Models loaded", String(d.models_loaded ?? 0), d.models_loaded ? "ok" : "warn") +
          li("Active model", d.active_model || "none") +
          li("Uptime", fmtUptime(d.uptime));
        setConn("online");
      })
      .catch(() => {
        hl.innerHTML = "<li><span class='k'>Engine offline</span></li>";
        setConn("offline");
      });

    // automation + models best-effort
    fetch(API() + "/system/automation-status")
      .then((r) => r.json())
      .then((d) => {
        const el = $("autoList");
        const rows = [];
        Object.keys(d || {}).forEach((k) => {
          const v = d[k];
          rows.push(li(
            k.replace(/_/g, " "),
            typeof v === "number" ? v.toLocaleString() : String(v),
            v === 0 || v === false || v === "ok" || v === "healthy" ? "ok" : ""
          ));
        });
        el.innerHTML = rows.join("") || "<li><span class='k'>—</span></li>";
      })
      .catch(() => {});

    fetch(API() + "/models")
      .then((r) => r.json())
      .then((list) => {
        $("modelList").innerHTML = (list || [])
          .map((m) =>
            li(
              m.name,
              (m.parameters / 1e6).toFixed(1) + "M · " + (m.architecture || "SSL"),
              "ok"
            )
          )
          .join("");
        const es = $("exportSelect");
        if (es && Array.isArray(list)) {
          es.innerHTML =
            '<option value="">Select a model…</option>' +
            list
              .map((m) => '<option value="' + esc(m.name) + '">' + esc(m.name) + "</option>")
              .join("");
        }
      })
      .catch(() => {});
  }

  function wireExport() {
    const btn = $("exportBtn");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const name = ($("exportSelect") || {}).value || "";
      const hint = $("exportHint");
      const dl = $("exportDl");
      if (!name) {
        hint.textContent = "Choose a model first.";
        return;
      }
      btn.disabled = true;
      btn.textContent = "Exporting…";
      hint.textContent = "";
      if (dl) dl.hidden = true;
      fetch(API() + "/models/" + encodeURIComponent(name) + "/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ opset: 14, input_size: 224 }),
      })
        .then(async (r) => {
          const j = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error((j.detail || r.status) + "");
          if (dl) {
            dl.href = API() + "/models/" + encodeURIComponent(name) + "/export";
            dl.hidden = false;
            dl.textContent = "⬇ " + (j.size_mb || "?") + " MB · " +
              (j.verified ? "verified ✓" : "not verified");
          }
          hint.textContent =
            "Exported " + (j.size_mb || "?") + " MB → tap Download to save the .onnx";
        })
        .catch((e) => {
          hint.textContent = "Export failed: " + e.message;
        })
        .finally(() => {
          btn.disabled = false;
          btn.textContent = "⚡ Export";
        });
    });
  }

  function knnStatusRow() {
    const knn = window.CropSSLKNN;
    if (!knn) return;
    const s = knn.status();
    const el = $("knnStatus");
    if (el) el.textContent = s.ready
      ? "✓ " + s.model + " · " + s.mode + " · " + s.classes + " classes · " + s.support + " support"
      : "No offline bundle loaded — " + s.reason;
  }

  function wireKnn() {
    const btn = $("knnBtn");
    if (!btn || !window.CropSSLKNN) return;
    btn.addEventListener("click", async () => {
      const name = ($("exportSelect") || {}).value || "";
      const hint = $("knnHint");
      const dl = $("knnDl");
      if (!name) {
        hint.textContent = "Choose a model first (same selector as ONNX export).";
        return;
      }
      btn.disabled = true;
      btn.textContent = "Building…";
      hint.textContent = "";
      if (dl) dl.hidden = true;
      try {
        const classesEl = $("knnClasses");
        const numClasses = Math.max(2, Math.min(20, parseInt((classesEl && classesEl.value) || "5", 10) || 5));
        const r = await fetch(API() + "/models/" + encodeURIComponent(name) + "/knn-bundle", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ num_classes: numClasses, shots: 5, k: 0 }),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error((j.detail || r.status) + "");
        dl.href = API() + "/models/" + encodeURIComponent(name) + "/knn-bundle";
        dl.hidden = false;
        dl.textContent = "⬇ " + (j.size_mb || "?") + " MB · " +
          j.num_classes + " classes · " + j.mode;
        hint.textContent =
          "Bundle ready — tap Download, then Load to go fully offline.";
      } catch (e) {
        hint.textContent = "Bundle build failed: " + e.message;
      } finally {
        btn.disabled = false;
        btn.textContent = "🧮 Build k-NN Bundle";
      }
    });

    const loadBtn = $("knnLoadBtn");
    if (loadBtn) {
      loadBtn.addEventListener("click", async () => {
        const name = ($("exportSelect") || {}).value || "";
        const hint = $("knnHint");
        if (!name) {
          hint.textContent = "Choose a model first (same selector as ONNX export).";
          return;
        }
        loadBtn.disabled = true;
        loadBtn.textContent = "Loading runtime + model…";
        try {
          await window.CropSSLKNN.load(API(), name);
          ls.setItem("cropssl_knn_model", name);
          hint.textContent = window.CropSSLKNN.status().ready
            ? "Offline inference ready — scans will work without the server."
            : "Load failed: " + window.CropSSLKNN.status().reason;
        } finally {
          loadBtn.disabled = false;
          loadBtn.textContent = "⬇ Load for Offline Use";
          knnStatusRow();
        }
      });
    }

    // Pair with the existing ONNX export: once an export downloads, offer the bundle too.
    const exportDl = $("exportDl");
    if (exportDl) {
      exportDl.addEventListener("click", () => {
        const name = ($("exportSelect") || {}).value || "";
        const hint = $("knnHint");
        if (hint && name) {
          hint.textContent = "ONNX saved. Build + download the k-NN bundle, then Load for offline use.";
        }
      });
    }
  }

  function li(k, v, cls) {
    return (
      "<li><span class='k'>" + esc(k) + "</span>" +
      "<span class='v " + (cls || "") + "'>" + esc(v) + "</span></li>"
    );
  }
  function fmtUptime(s) {
    s = Math.floor(s || 0);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  }

  /* wiring */
  document.addEventListener("DOMContentLoaded", () => {
    $("apiBase").value = apiDefault;
    $("apiBase").addEventListener("change", () => {
      ls.setItem("cropssl_api", $("apiBase").value.trim());
      ping();
    });

    $("openCam").addEventListener("click", () => {
      const fi = $("fileInput");
      fi.setAttribute("capture", "environment");
      fi.click();
    });
    $("openGallery").addEventListener("click", () => {
      $("fileInput").removeAttribute("capture");
      $("fileInput").click();
    });
    $("fileInput").addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) analyze(f);
    });
    $("reanalyze").addEventListener("click", () => {
      $("resultCard").classList.add("hidden");
      $("landing").scrollIntoView({ behavior: "smooth" });
    });

    $("tabScan").addEventListener("click", () => TABS.scan());
    $("tabStatus").addEventListener("click", () => TABS.status());

    // load model list once API base is set
    refreshModelList();
    wireExport();
    wireKnn();
    $("apiBase").addEventListener("change", refreshModelList);

    // offline k-NN: reflect status; auto-load a previously stored bundle
    knnStatusRow();
    const knnModel = ls.getItem("cropssl_knn_model");
    if (knnModel && window.CropSSLKNN && !window.CropSSLKNN.status().ready) {
      window.CropSSLKNN.load(API(), knnModel).then(knnStatusRow);
    }

    // service worker
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/app/sw.js").catch(() => {});
    }
    ping();
    setInterval(ping, 15000);
    // network came back? try to (re)load a stored bundle automatically
    window.addEventListener("online", () => {
      const knnModel = ls.getItem("cropssl_knn_model");
      if (knnModel && window.CropSSLKNN && !window.CropSSLKNN.status().ready) {
        window.CropSSLKNN.load(API(), knnModel).then(knnStatusRow);
      }
    });
  });
})();
