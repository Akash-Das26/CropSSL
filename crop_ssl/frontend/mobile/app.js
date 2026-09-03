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
      $("imgTag").textContent = "ERROR";
      $("analyzing").classList.add("hidden");
      $("resultCard").classList.remove("hidden");
      $("topLabel").textContent = "Connection failed";
      $("confPct").textContent = "—";
      $("subLabel").textContent = e.message;
      $("bars").innerHTML = "";
      setConn("offline");
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
      })
      .catch(() => {});
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
    $("apiBase").addEventListener("change", refreshModelList);

    // service worker
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/app/sw.js").catch(() => {});
    }
    ping();
    setInterval(ping, 15000);
  });
})();
