//! Dump a GGUF file's metadata and tensor inventory — the first look at any
//! candidate import: `cargo run -p mummu --example gguf-info -- <path.gguf>`.

use mummu::gguf::GgufFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf-info <path.gguf>");
    let f = GgufFile::open(std::path::Path::new(&path)).expect("gguf opens");
    println!("architecture: {:?}", f.architecture());

    for (k, v) in &f.metadata {
        // Token lists are megabytes of noise; show everything else.
        if k.contains("tokens") || k.contains("merges") || k.contains("scores") {
            continue;
        }
        let shown = format!("{v:?}");
        let shown = if shown.len() > 120 {
            &shown[..120]
        } else {
            &shown
        };
        println!("{k} = {shown}");
    }

    println!("tensors: {}", f.tensors.len());
    for t in f.tensors.iter().take(30) {
        println!("  {} {:?} {:?}", t.name, t.dims, t.dtype);
    }
    // The layer-count picture: show every distinct non-blk name too.
    for t in &f.tensors {
        if !t.name.starts_with("blk.") {
            println!("  top-level: {} {:?} {:?}", t.name, t.dims, t.dtype);
        }
    }
}
