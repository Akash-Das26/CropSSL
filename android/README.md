# CropSSL · Android App (WebView wrapper)

A minimal native Android wrapper that embeds the CropSSL mobile PWA
(`/app`) in a full-screen WebView. The PWA already provides the futuristic
camera-first UI, installability, and offline shell — this wrapper gives you a
real APK that can be sideloaded or shipped to the Play Store.

## How it works

```
┌──────────────────────────── Android APK ───────────────────────────┐
│  MainActivity (WebView)                                            │
│     └── loads  http://<server>:8000/app/                          │
│          │                                                         │
│          └── calls the CropSSL REST API (/predict, /health, …)     │
└─────────────────────────────────────────────────────────────────────┘
         ▲                                                            │
         └───── same network (Wi-Fi / LAN / VPN / ngrok tunnel) ──────┘
```

The phone does **no model computation** — the FastAPI backend on your
PC/server does inference and the app renders results.

## Requirements

- Android Studio (Hedgehog or newer) with SDK 34+
- JDK 17
- Your CropSSL backend reachable from the phone
  (e.g. `http://192.168.1.5:8000` on the same Wi-Fi, or a tunnel)

## Build & run

1. Open this `android/` folder in Android Studio.
2. Edit the server URL in
   `app/src/main/java/com/cropssl/app/MainActivity.java`:
   ```java
   private static final String SERVER_URL = "http://192.168.1.5:8000/app/";
   ```
   > Android blocks cleartext HTTP by default. For LAN testing the manifest
   > already enables `usesCleartextTraffic="true"` — **remove it** and serve
   > HTTPS before any production/Play Store release.
3. Connect a phone (enable Developer Options → USB debugging) and press Run,
   or build a release APK: **Build → Build App Bundle(s) / APK(s)**.

## First launch

- The app asks for **camera / storage** permission (needed for the
  "Take Photo" button — the PWA captures via the Android camera intent).
- Grant permissions, then use **Scan** to photograph a leaf and get the
  diagnosis, or **Engine** to watch backend health & automation status.

## Where the app files live

```
android/
├── app/
│   ├── build.gradle          # app module (single-activity, no deps)
│   └── src/main/
│       ├── AndroidManifest.xml
│       └── java/com/cropssl/app/MainActivity.java
├── build.gradle              # project-level
├── settings.gradle
└── gradle.properties
```

## Going further

- **Icon & name:** replace the launcher icon via Android Studio
  (right-click `res` → New → Image Asset). The `assets/logo.png` from the
  repo root is the recommended source image.
- **Production HTTPS:** serve the API behind HTTPS (reverse proxy / tunnel)
  and set `usesCleartextTraffic="false"`.
- **Play Store:** use the PWA directly instead — it is already installable
  from Chrome (Add to Home Screen) and needs no store upload at all.
