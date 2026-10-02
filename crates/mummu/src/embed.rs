//! Sentence embeddings: one API over every embedder in the zoo.
//!
//! Two shapes of embedder exist and both are here:
//!
//! - **Encoders** (`all-MiniLM`, BERT-class): bidirectional attention,
//!   masked-mean pooling — [`crate::models::minilm`].
//! - **Decoder embedders** (`harrier-oss-v1`, `Qwen3-Embedding`): a Qwen3
//!   trunk run causally, pooled at the **last token** (the `<|endoftext|>`
//!   their tokenizer appends), L2-normalized, with an instruction prefix on
//!   the *query* side only — [`crate::models::qwen3::Qwen3Trunk`].
//!
//! Nothing here is per-checkpoint code. How a checkpoint pools, whether it
//! normalizes, and which prompt it wants in front of a query are all read
//! from the sentence-transformers files that ship beside the weights
//! (`modules.json`, `1_Pooling/config.json`,
//! `config_sentence_transformers.json`, `sentence_bert_config.json`), so a
//! new checkpoint of a supported architecture embeds the way its authors
//! evaluated it without a line changing here. When those files are absent
//! the architecture's own convention stands in (mean + normalize for BERT,
//! last-token + normalize for a decoder).

use std::path::Path;

use burn::tensor::{Device, Tensor};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::import::ImportError;
use crate::models::minilm::{self, LoadedMiniLm};
use crate::models::qwen3::{self, Qwen3Trunk};

/// How token states become one sentence vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Pooling {
    /// The mean over every (non-padding) position.
    Mean,
    /// The first position (`[CLS]`).
    Cls,
    /// The last position — for a causal decoder the only one that has seen
    /// the whole input.
    LastToken,
}

/// Which side of a retrieval pair a text is. Asymmetric embedders (every
/// decoder embedder here) put an instruction in front of a query and
/// nothing in front of a document; symmetric ones treat both alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextKind {
    Query,
    Document,
}

/// Longest input an embedder takes, in tokens, before it truncates.
///
/// A bound on work and memory, not on the model: the decoder embedders here
/// train at 32k, and an input that long costs a 0.6B model ~2 GB of f32 KV
/// state on its way through. Retrieval embeds *chunks*, which
/// [`crate::rag::chunk_text`] keeps far below this; past it the tail is
/// dropped, as sentence-transformers does at its own `max_seq_length`.
pub const MAX_EMBED_TOKENS: usize = 4096;

/// The embedding-relevant half of a sentence-transformers checkpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SentenceConfig {
    /// `1_Pooling/config.json`'s mode, when the file is present.
    pub pooling: Option<Pooling>,
    /// `modules.json` lists a `Normalize` module. `None` when there is no
    /// `modules.json` to say either way.
    pub normalize: Option<bool>,
    /// Named prompts from `config_sentence_transformers.json`, in file order.
    pub prompts: Vec<(String, String)>,
    /// `sentence_bert_config.json`'s `max_seq_length`.
    pub max_seq_length: Option<usize>,
}

/// Largest sentence-transformers JSON file read (they are a few hundred bytes).
const MAX_ST_FILE: u64 = 1 << 20;

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, ImportError> {
    let Ok(meta) = std::fs::metadata(path) else {
        return Ok(None);
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_ST_FILE {
        return Err(ImportError::Parse {
            file: path.to_path_buf(),
            reason: format!("{} bytes is over the {MAX_ST_FILE}-byte cap", meta.len()),
        });
    }
    std::fs::read(path)
        .map(Some)
        .map_err(|e| ImportError::Parse {
            file: path.to_path_buf(),
            reason: e.to_string(),
        })
}

impl SentenceConfig {
    /// Read whatever sentence-transformers files sit in `dir`. Every file
    /// is optional; one that is present but malformed is a loud error.
    ///
    /// # Errors
    ///
    /// [`ImportError::Parse`] naming the file that does not parse.
    pub fn from_dir(dir: &Path) -> Result<Self, ImportError> {
        let modules_path = dir.join("modules.json");
        let modules = read_optional(&modules_path)?;
        // The pooling module's directory is named in modules.json; the
        // conventional `1_Pooling` stands in when it is not.
        let pooling_dir = modules
            .as_deref()
            .and_then(|b| module_path(b, "Pooling"))
            .unwrap_or_else(|| "1_Pooling".to_owned());
        let pooling_path = dir.join(&pooling_dir).join("config.json");
        let pooling = read_optional(&pooling_path)?;
        let st_path = dir.join("config_sentence_transformers.json");
        let st = read_optional(&st_path)?;
        let sbert_path = dir.join("sentence_bert_config.json");
        let sbert = read_optional(&sbert_path)?;

        let parse = |path: &Path, bytes: &[u8]| -> Result<Value, ImportError> {
            serde_json::from_slice(bytes).map_err(|e| ImportError::Parse {
                file: path.to_path_buf(),
                reason: e.to_string(),
            })
        };
        let mut out = Self::default();
        if let Some(b) = &modules {
            let v = parse(&modules_path, b)?;
            let list = v.as_array().ok_or_else(|| ImportError::Parse {
                file: modules_path.clone(),
                reason: "not a JSON array".into(),
            })?;
            out.normalize = Some(list.iter().any(|m| {
                m.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| t.ends_with(".Normalize"))
            }));
        }
        if let Some(b) = &pooling {
            let v = parse(&pooling_path, b)?;
            out.pooling = Some(pooling_mode(&v).map_err(|reason| ImportError::Parse {
                file: pooling_path.clone(),
                reason,
            })?);
        }
        if let Some(b) = &st {
            let v = parse(&st_path, b)?;
            if let Some(map) = v.get("prompts").and_then(Value::as_object) {
                for (name, text) in map {
                    let text = text.as_str().ok_or_else(|| ImportError::Parse {
                        file: st_path.clone(),
                        reason: format!("prompt {name:?} is not a string"),
                    })?;
                    out.prompts.push((name.clone(), text.to_owned()));
                }
            }
        }
        if let Some(b) = &sbert {
            let v = parse(&sbert_path, b)?;
            out.max_seq_length = v
                .get("max_seq_length")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0);
        }
        Ok(out)
    }

    /// The prompt a **query** gets: the checkpoint's `query` prompt, else
    /// its web-search one (`harrier-oss-v1` names it `web_search_query`),
    /// else none.
    #[must_use]
    pub fn query_prompt(&self) -> Option<&str> {
        self.prompt("query")
            .or_else(|| self.prompt("web_search_query"))
    }

    /// The prompt a **document** gets: `document` or `passage`, else none.
    /// (Qwen3-Embedding declares an empty `document` prompt — documents are
    /// embedded bare.)
    #[must_use]
    pub fn document_prompt(&self) -> Option<&str> {
        self.prompt("document").or_else(|| self.prompt("passage"))
    }

    fn prompt(&self, name: &str) -> Option<&str> {
        self.prompts
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_str())
    }
}

/// The directory of the first `modules.json` entry whose type ends in
/// `.{kind}`.
fn module_path(modules: &[u8], kind: &str) -> Option<String> {
    let v: Value = serde_json::from_slice(modules).ok()?;
    let suffix = format!(".{kind}");
    v.as_array()?.iter().find_map(|m| {
        let t = m.get("type")?.as_str()?;
        if !t.ends_with(&suffix) {
            return None;
        }
        let p = m.get("path")?.as_str()?;
        // A path from a checkpoint file is spliced into a filesystem path:
        // one plain component only.
        (!p.is_empty() && crate::manage::is_safe_component(p)).then(|| p.to_owned())
    })
}

/// The single pooling mode a `1_Pooling/config.json` turns on.
fn pooling_mode(v: &Value) -> Result<Pooling, String> {
    let on = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
    let modes = [
        ("pooling_mode_mean_tokens", Some(Pooling::Mean)),
        ("pooling_mode_cls_token", Some(Pooling::Cls)),
        ("pooling_mode_lasttoken", Some(Pooling::LastToken)),
        ("pooling_mode_max_tokens", None),
        ("pooling_mode_mean_sqrt_len_tokens", None),
        ("pooling_mode_weightedmean_tokens", None),
    ];
    let set: Vec<_> = modes.iter().filter(|(k, _)| on(k)).collect();
    match set.as_slice() {
        [(_, Some(p))] => Ok(*p),
        [(k, None)] => Err(format!("pooling mode {k} is not implemented")),
        [] => Err("no pooling mode is turned on".into()),
        _ => Err(format!(
            "{} pooling modes are on at once (concatenated pooling is not implemented)",
            set.len()
        )),
    }
}

/// A custom retrieval instruction in the checkpoint's own query-prompt
/// shape.
///
/// The decoder embedders train with `Instruct: {task}\nQuery:` and differ in
/// the byte after the colon (Qwen3-Embedding: nothing; harrier: a space).
/// Splicing the task into the checkpoint's own prompt keeps that byte right
/// for every checkpoint. `None` when the checkpoint has no query prompt of
/// that shape — a symmetric embedder, which takes no instruction.
#[must_use]
pub fn instructed_prompt(query_prompt: &str, instruction: &str) -> Option<String> {
    let rest = query_prompt.strip_prefix("Instruct: ")?;
    let at = rest.rfind("\nQuery:")?;
    let tail = &rest[at..];
    (tail == "\nQuery:" || tail == "\nQuery: ").then(|| format!("Instruct: {instruction}{tail}"))
}

/// One embedded text.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub vector: Vec<f32>,
    /// Tokens the model saw (prompt included, after truncation) — what an
    /// API reports as usage.
    pub tokens: usize,
    /// The input was longer than [`Embedder::max_tokens`] and lost its tail.
    pub truncated: bool,
}

/// Both boxed: the two loaded models differ in size by hundreds of bytes
/// of config, and the enum lives once per embedder.
enum Encoder {
    MiniLm(Box<LoadedMiniLm>),
    Qwen3(Box<Qwen3Trunk>),
}

/// A loaded embedder: model, tokenizer, and the checkpoint's own pooling,
/// normalization and prompt conventions.
pub struct Embedder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    sentence: SentenceConfig,
    pooling: Pooling,
    normalize: bool,
    max_tokens: usize,
}

impl Embedder {
    /// Load the embedder checkpoint in `dir`, dispatching on `config.json`'s
    /// `model_type`: `bert` → the `MiniLM` encoder, `qwen3` → a Qwen3 trunk.
    ///
    /// # Errors
    ///
    /// [`ImportError`] for a missing or malformed `config.json` /
    /// `tokenizer.json` / sentence-transformers file, a `model_type` that is
    /// not an embedder this crate implements, a pooling mode it does not
    /// implement, or a weight load that fails its check.
    pub fn load_from_dir(dir: &Path, device: &Device) -> Result<Self, ImportError> {
        let cfg_path = crate::import::required_file(dir, "config.json")?;
        let bytes = std::fs::read(&cfg_path).map_err(|e| ImportError::Parse {
            file: cfg_path.clone(),
            reason: e.to_string(),
        })?;
        let model_type = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v.get("model_type")?.as_str().map(str::to_owned))
            .unwrap_or_default();
        let sentence = SentenceConfig::from_dir(dir)?;
        let tok_path = crate::import::required_file(dir, "tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| ImportError::Parse {
            file: tok_path.clone(),
            reason: e.to_string(),
        })?;
        match model_type.as_str() {
            "bert" => {
                let lm = minilm::load_from_dir(dir, device)?;
                let positions = lm.config.max_position_embeddings;
                Self::assemble(
                    Encoder::MiniLm(Box::new(lm)),
                    tokenizer,
                    sentence,
                    positions,
                    &tok_path,
                )
            }
            "qwen3" => {
                let trunk = qwen3::load_trunk_from_dir(dir, device)?;
                let positions = trunk
                    .config
                    .max_position_embeddings
                    .unwrap_or(MAX_EMBED_TOKENS);
                Self::assemble(
                    Encoder::Qwen3(Box::new(trunk)),
                    tokenizer,
                    sentence,
                    positions,
                    &tok_path,
                )
            }
            other => Err(ImportError::Parse {
                file: cfg_path,
                reason: format!(
                    "model_type {other:?} is not an embedder this build implements (bert, qwen3)"
                ),
            }),
        }
    }

    fn assemble(
        encoder: Encoder,
        mut tokenizer: Tokenizer,
        sentence: SentenceConfig,
        positions: usize,
        tok_path: &Path,
    ) -> Result<Self, ImportError> {
        let decoder = matches!(encoder, Encoder::Qwen3(_));
        let pooling = sentence.pooling.unwrap_or(if decoder {
            Pooling::LastToken
        } else {
            Pooling::Mean
        });
        if !decoder && pooling != Pooling::Mean {
            return Err(ImportError::Parse {
                file: tok_path.to_path_buf(),
                reason: format!("{pooling:?} pooling is not implemented for BERT encoders"),
            });
        }
        let max_tokens = sentence
            .max_seq_length
            .unwrap_or(positions)
            .min(positions)
            .clamp(1, MAX_EMBED_TOKENS);
        // Truncation inside the tokenizer is post-processor aware: the EOS a
        // decoder embedder pools at (or BERT's [SEP]) survives the cut.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_tokens,
                ..Default::default()
            }))
            .map_err(|e| ImportError::Parse {
                file: tok_path.to_path_buf(),
                reason: format!("truncation: {e}"),
            })?;
        tokenizer.with_padding(None);
        Ok(Self {
            encoder,
            tokenizer,
            normalize: sentence.normalize.unwrap_or(true),
            sentence,
            pooling,
            max_tokens,
        })
    }

    /// Width of the vectors this embedder returns.
    #[must_use]
    pub const fn dims(&self) -> usize {
        match &self.encoder {
            Encoder::MiniLm(m) => m.config.hidden_size,
            Encoder::Qwen3(t) => t.config.hidden_size,
        }
    }

    /// Longest input, in tokens, before truncation.
    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// How this checkpoint pools.
    #[must_use]
    pub const fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// The sentence-transformers conventions it was loaded with.
    #[must_use]
    pub const fn sentence_config(&self) -> &SentenceConfig {
        &self.sentence
    }

    /// The prompt `kind` gets in front of it (empty when none).
    #[must_use]
    pub fn prompt(&self, kind: TextKind) -> &str {
        match kind {
            TextKind::Query => self.sentence.query_prompt(),
            TextKind::Document => self.sentence.document_prompt(),
        }
        .unwrap_or("")
    }

    /// `text` with the prompt `kind` gets, or with a custom task
    /// `instruction` in the checkpoint's own query-prompt shape. An
    /// instruction is ignored by an embedder that takes none.
    #[must_use]
    pub fn prompted(&self, text: &str, kind: TextKind, instruction: Option<&str>) -> String {
        let custom = instruction
            .filter(|_| kind == TextKind::Query)
            .and_then(|i| instructed_prompt(self.sentence.query_prompt()?, i));
        custom.map_or_else(
            || format!("{}{text}", self.prompt(kind)),
            |prefix| format!("{prefix}{text}"),
        )
    }

    /// Token ids for an already-prompted text, truncated to
    /// [`Self::max_tokens`], plus whether the cut happened.
    ///
    /// # Errors
    ///
    /// The tokenizer's own error, or an empty encoding.
    pub fn tokenize(&self, prompted: &str) -> Result<(Vec<u32>, bool), String> {
        let enc = self
            .tokenizer
            .encode(prompted, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids = enc.get_ids().to_vec();
        if ids.is_empty() {
            return Err("the input tokenized to nothing".into());
        }
        Ok((ids, !enc.get_overflowing().is_empty()))
    }

    /// Embed `text` as a `kind`, with the checkpoint's prompt for that kind
    /// (or `instruction` in its place, for a query).
    ///
    /// # Errors
    ///
    /// A tokenizer error, or a failed device readback.
    pub async fn embed(
        &self,
        text: &str,
        kind: TextKind,
        instruction: Option<&str>,
        device: &Device,
    ) -> Result<Embedding, String> {
        let prompted = self.prompted(text, kind, instruction);
        self.embed_prompted(&prompted, device).await
    }

    /// Embed a text exactly as given — no prompt added. What an
    /// OpenAI/ollama-style API serves: those callers send the prompt
    /// themselves, as they would to any other server.
    ///
    /// # Errors
    ///
    /// A tokenizer error, or a failed device readback.
    pub async fn embed_prompted(
        &self,
        prompted: &str,
        device: &Device,
    ) -> Result<Embedding, String> {
        let (ids, truncated) = self.tokenize(prompted)?;
        let vector = self.embed_ids(&ids, device).await?;
        Ok(Embedding {
            vector,
            tokens: ids.len(),
            truncated,
        })
    }

    /// Embed pre-tokenized ids (truncated to [`Self::max_tokens`] here too,
    /// keeping the head: ids from a caller carry no post-processor to
    /// respect).
    ///
    /// # Errors
    ///
    /// An empty `ids`, or a failed device readback.
    pub async fn embed_ids(&self, ids: &[u32], device: &Device) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("empty token input".into());
        }
        let ids = &ids[..ids.len().min(self.max_tokens)];
        match &self.encoder {
            Encoder::MiniLm(m) => {
                // MiniLM pools and normalizes inside its own forward (the
                // sentence-transformers recipe it was ported with).
                let mask = vec![1.0f32; ids.len()];
                m.embed_ids(ids, &mask, device)
            }
            Encoder::Qwen3(t) => {
                let pooled = t.pooled(ids, self.pooling, device);
                let out = if self.normalize {
                    l2_normalize(pooled)
                } else {
                    pooled
                };
                out.into_data_async()
                    .await
                    .map_err(|e| format!("embedding readback: {e:?}"))?
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .map_err(|e| format!("embedding readback: {e:?}"))
            }
        }
    }
}

/// Divide a `[1, d]` row by its L2 norm (floored, so an all-zero row stays
/// zero rather than becoming NaN — `torch.nn.functional.normalize`'s eps).
fn l2_normalize(x: Tensor<2>) -> Tensor<2> {
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    x.div(norm)
}

/// Cosine similarity of two vectors (their dot product when both are
/// unit-norm, which every normalizing embedder's are).
///
/// # Panics
///
/// When the lengths differ.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mummu-embed-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The real files `harrier-oss-v1-0.6b` ships, verbatim.
    #[test]
    fn harrier_sentence_files_read_as_last_token_normalized_with_a_spaced_query() {
        let dir = scratch("harrier");
        std::fs::write(
            dir.join("modules.json"),
            r#"[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
                {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
                {"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}]"#,
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
        std::fs::write(
            dir.join("1_Pooling/config.json"),
            r#"{"word_embedding_dimension":1024,"pooling_mode_cls_token":false,
                "pooling_mode_mean_tokens":false,"pooling_mode_max_tokens":false,
                "pooling_mode_mean_sqrt_len_tokens":false,"pooling_mode_weightedmean_tokens":false,
                "pooling_mode_lasttoken":true,"include_prompt":true}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("config_sentence_transformers.json"),
            r#"{"prompts":{
                "web_search_query":"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                "sts_query":"Instruct: Retrieve semantically similar text\nQuery: ",
                "bitext_query":"Instruct: Retrieve parallel sentences\nQuery: "},
                "default_prompt_name":null,"similarity_fn_name":"cosine"}"#,
        )
        .unwrap();
        let cfg = SentenceConfig::from_dir(&dir).unwrap();
        assert_eq!(cfg.pooling, Some(Pooling::LastToken));
        assert_eq!(cfg.normalize, Some(true));
        assert_eq!(
            cfg.query_prompt(),
            Some(
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
            )
        );
        assert_eq!(cfg.document_prompt(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dir_without_sentence_files_is_all_defaults() {
        let dir = scratch("bare");
        assert_eq!(
            SentenceConfig::from_dir(&dir).unwrap(),
            SentenceConfig::default()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unimplemented_or_ambiguous_pooling_is_refused_by_name() {
        let max = serde_json::json!({"pooling_mode_max_tokens": true});
        assert!(pooling_mode(&max).unwrap_err().contains("max_tokens"));
        let two =
            serde_json::json!({"pooling_mode_mean_tokens": true, "pooling_mode_cls_token": true});
        assert!(pooling_mode(&two).is_err());
        assert!(pooling_mode(&serde_json::json!({})).is_err());
        let mean = serde_json::json!({"pooling_mode_mean_tokens": true});
        assert_eq!(pooling_mode(&mean).unwrap(), Pooling::Mean);
    }

    /// The two decoder-embedder prompt shapes differ by one trailing space,
    /// and a custom instruction must keep each checkpoint's own byte.
    #[test]
    fn a_custom_instruction_keeps_the_checkpoints_own_query_suffix() {
        let qwen = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";
        let harrier = "Instruct: Retrieve semantically similar text\nQuery: ";
        assert_eq!(
            instructed_prompt(qwen, "Find the statute").as_deref(),
            Some("Instruct: Find the statute\nQuery:")
        );
        assert_eq!(
            instructed_prompt(harrier, "Find the statute").as_deref(),
            Some("Instruct: Find the statute\nQuery: ")
        );
        assert_eq!(instructed_prompt("query: ", "x"), None);
        assert_eq!(instructed_prompt("Instruct: a\nQuestion:", "x"), None);
    }

    #[test]
    fn a_traversing_module_path_is_ignored() {
        let modules = br#"[{"path":"../../etc","type":"sentence_transformers.models.Pooling"}]"#;
        assert_eq!(module_path(modules, "Pooling"), None);
        let ok = br#"[{"path":"1_Pooling","type":"sentence_transformers.models.Pooling"}]"#;
        assert_eq!(module_path(ok, "Pooling").as_deref(), Some("1_Pooling"));
    }

    #[test]
    fn cosine_is_scale_free_and_bounded() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        let c = [-1.0, -2.0, -3.0];
        assert!((cosine(&a, &c) + 1.0).abs() < 1e-6);
        assert!(cosine(&[0.0, 0.0], &[1.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_makes_unit_rows_and_leaves_zero_alone() {
        let device = crate::backend::cpu_device();
        let x = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(vec![3.0f32, 4.0], [1, 2]),
            (&device, crate::backend::float_dtype(&device)),
        );
        let v = l2_normalize(x).into_data().try_to_vec::<f32>().unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let z = Tensor::<2>::zeros([1, 2], &device);
        let v = l2_normalize(z).into_data().try_to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite() && *x == 0.0));
    }
}
