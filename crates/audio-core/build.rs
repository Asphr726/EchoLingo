fn main() {
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .cpp(true)
            .file("native/macos_capture.mm")
            .flag("-fobjc-arc")
            .flag("-std=c++17")
            .compile("echolingo_macos_capture");
        for framework in [
            "AppKit",
            "AudioToolbox",
            "CoreAudio",
            "CoreMedia",
            "Foundation",
            "ScreenCaptureKit",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rerun-if-changed=native/macos_capture.mm");
    }
}
