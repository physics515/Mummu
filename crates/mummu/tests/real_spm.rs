//! SentencePiece-import BYTE gate: build the tokenizer from a checkpoint's
//! `spiece.model`/`tokenizer.model` proto ALONE and prove it produces
//! byte-identical ids to the same checkpoint's shipped `tokenizer.json`
//! (HF's own conversion of that proto). Ignored by default; run with:
//!
//! ```text
//! MUMMU_SPM_DIR=path/to/flan-t5-small-tok \
//!   cargo test -p mummu --test real_spm -- --ignored --nocapture
//! ```
//!
//! The dir needs the proto (`spiece.model` or `tokenizer.model`) and the
//! reference `tokenizer.json`. Encoding runs with `add_special_tokens =
//! false` on both sides: post-processing (T5's `</s>` append) is template
//! knowledge the proto does not carry, and added tokens BEYOND the proto
//! (T5's `<extra_id_*>`) live in sibling metadata — both are the caller's
//! layer, not the proto import's.

use std::path::PathBuf;

use tokenizers::Tokenizer;

fn spm_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MUMMU_SPM_DIR")?);
    dir.join("tokenizer.json").is_file().then_some(dir)
}

/// The proto file under either of its conventional names.
fn proto_path(dir: &std::path::Path) -> PathBuf {
    let spiece = dir.join("spiece.model");
    if spiece.is_file() {
        spiece
    } else {
        dir.join("tokenizer.model")
    }
}

#[test]
#[ignore = "needs a local SentencePiece checkpoint dir (MUMMU_SPM_DIR) with tokenizer.json"]
fn spm_proto_ids_byte_match_the_reference_tokenizer_json() {
    let dir = spm_dir().expect("set MUMMU_SPM_DIR to a dir with spiece.model + tokenizer.json");
    let proto = proto_path(&dir);
    assert!(
        proto.is_file(),
        "no spiece.model/tokenizer.model in {dir:?}"
    );

    let ours = mummu::tokenizer::tokenizer_from_spm(&proto).expect("proto import builds");
    let reference =
        Tokenizer::from_file(dir.join("tokenizer.json")).expect("reference tokenizer loads");

    // The same battery shape the GGUF tokenizer gate uses: plain text,
    // whitespace runs (the remove_extra_whitespaces leg), leading/trailing
    // space (the add_dummy_prefix leg), unicode/CJK/emoji (the Precompiled
    // charsmap leg), contractions, newlines, and empty.
    let battery = [
        "Translate to German: The house is wonderful.",
        "hello world",
        "  leading and trailing  ",
        "spaces    collapse     here",
        "Bonjour, ça va? Ærøskøbing — ﬁne. １２３ ｆｕｌｌｗｉｄｔｈ",
        "日本語のテキストと emoji 🙂🚀 mixed",
        "don't can't won't it's",
        "line\nbreaks\n\nand\ttabs",
        "™ and ½ and ﬂags",
        "",
    ];
    for prompt in battery {
        let a = ours.encode(prompt, false).expect("ours encodes");
        let b = reference.encode(prompt, false).expect("reference encodes");
        assert_eq!(
            a.get_ids(),
            b.get_ids(),
            "ids diverge on {prompt:?}: ours {:?} vs reference {:?} (tokens {:?} vs {:?})",
            a.get_ids(),
            b.get_ids(),
            a.get_tokens(),
            b.get_tokens(),
        );
        // Decode round-trips identically too (Metaspace decoder fidelity).
        let ad = ours.decode(a.get_ids(), true).expect("ours decodes");
        let bd = reference
            .decode(b.get_ids(), true)
            .expect("reference decodes");
        assert_eq!(ad, bd, "decodes diverge on {prompt:?}");
    }

    // The proto's own specials resolve to the same ids on both sides.
    for special in ["<unk>", "</s>", "<pad>"] {
        assert_eq!(
            ours.token_to_id(special),
            reference.token_to_id(special),
            "special {special:?} id diverges"
        );
    }
    eprintln!(
        "[real_spm] {} battery prompts byte-identical; vocab {} vs reference {}",
        battery.len(),
        ours.get_vocab_size(false),
        reference.get_vocab_size(false),
    );
}
