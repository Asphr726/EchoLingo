import { describe, expect, it } from "vitest";
import { platform, platformFromUserAgent, secureStoreName } from "./platform";

describe("platform detection", () => {
  it("reads each desktop webview's user agent", () => {
    expect(platformFromUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)")).toBe("mac");
    expect(
      platformFromUserAgent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
      ),
    ).toBe("windows");
    expect(platformFromUserAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")).toBe("linux");
    expect(platformFromUserAgent("")).toBe("other");
  });

  it("trusts navigator.platform over a user agent that claims to be a Mac", () => {
    expect(platformFromUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15", "Linux x86_64")).toBe("linux");
    expect(platformFromUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "MacIntel")).toBe("mac");
    expect(platformFromUserAgent("", "Win32")).toBe("windows");
  });

  it("is safe outside a webview", () => {
    expect(() => platform()).not.toThrow();
    expect(platform()).toBe("other");
  });

  it("names the secure store of each platform", () => {
    expect(secureStoreName("mac")).toBe("the macOS Keychain");
    expect(secureStoreName("windows")).toBe("Windows Credential Manager");
    expect(secureStoreName("linux")).toContain("GNOME Keyring or KWallet");
    expect(secureStoreName("other")).toBe("the system secure store");
  });
});
