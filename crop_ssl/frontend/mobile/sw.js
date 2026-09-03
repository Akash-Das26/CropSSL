/* CropSSL mobile PWA — service worker */
const CACHE = "cropssl-v1";
const SHELL = ["/app/index.html", "/app/styles.css", "/app/app.js"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  // Never cache API calls — always go to network
  if (url.pathname.startsWith("/predict") || url.pathname.startsWith("/health") ||
      url.pathname.startsWith("/models") || url.pathname.startsWith("/system") ||
      url.pathname.startsWith("/classes") || url.pathname.startsWith("/auth") ||
      url.pathname.startsWith("/registry") || url.pathname.startsWith("/auto-retrain") ||
      url.pathname.startsWith("/drift") || url.pathname.startsWith("/audit") ||
      url.pathname.startsWith("/pipeline")) {
    return;
  }
  event.respondWith(
    caches.match(request).then((cached) => {
      const network = fetch(request)
        .then((resp) => {
          if (resp && resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE).then((c) => c.put(request, clone));
          }
          return resp;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
