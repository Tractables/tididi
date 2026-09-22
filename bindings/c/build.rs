fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Consumers locate the installed library through their runtime search path.
        println!("cargo:rustc-link-arg-cdylib=-Wl,-install_name,@rpath/libtididi_c.dylib");
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    cbindgen::generate(root).expect("generate C header").write_to_file(output.join("tididi.h"));
}
