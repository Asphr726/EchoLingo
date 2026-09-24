fn main() {
    // Build scripts run on the host; key the native capture on the target so
    // cross-checks for Windows or Linux from a Mac skip it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .file("native/macos_capture.mm")
        .flag("-fobjc-arc")
        .flag("-std=c++17")
        .compile("echolingo_macos_capture");
    for framework in [
        "AppKit",
        "AudioToolbox",
        "AVFAudio",
        "CoreAudio",
        "CoreGraphics",
        "CoreMedia",
        "Foundation",
        "ScreenCaptureKit",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rerun-if-changed=native/macos_capture.mm");
}
