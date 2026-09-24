export type Platform = "mac" | "windows" | "linux" | "other";

/** The desktop platform named by a webview's user agent: WKWebView reports
 *  "Macintosh", WebView2 "Windows NT" and WebKitGTK "X11; Linux". `hint` is
 *  `navigator.platform`, which WebKitGTK keeps truthful even when a site
 *  quirk makes its user agent claim to be a Mac. */
export function platformFromUserAgent(userAgent: string, hint = ""): Platform {
  const text = `${hint} ${userAgent}`;
  if (/Windows|Win32|Win64/i.test(text)) return "windows";
  if (/Linux|X11/i.test(text)) return "linux";
  if (/Macintosh|Mac OS X|MacIntel/i.test(text)) return "mac";
  return "other";
}

/** `?preview=live&platform=windows|linux` in a plain browser shows the
 *  copy and controls of another platform. */
function previewOverride(): Platform | null {
  if (typeof window === "undefined" || "__TAURI_INTERNALS__" in window) return null;
  const query = new URLSearchParams(window.location.search);
  if (query.get("preview") !== "live") return null;
  const requested = query.get("platform");
  return requested === "mac" || requested === "windows" || requested === "linux" ? requested : null;
}

/** The platform EchoLingo runs on; "other" outside a webview (unit tests
 *  run in Node, whose navigator carries no desktop user agent). */
export function platform(): Platform {
  const override = previewOverride();
  if (override) return override;
  if (typeof navigator === "undefined") return "other";
  return platformFromUserAgent(navigator.userAgent ?? "", typeof window === "undefined" ? "" : navigator.platform ?? "");
}

/** The OS secure store EchoLingo saves API keys in, as users know it. */
export function secureStoreName(current: Platform): string {
  switch (current) {
    case "mac":
      return "the macOS Keychain";
    case "windows":
      return "Windows Credential Manager";
    case "linux":
      return "a Secret Service provider such as GNOME Keyring or KWallet";
    default:
      return "the system secure store";
  }
}
