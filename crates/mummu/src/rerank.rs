//! Cross-encoder reranking with the Qwen3-Reranker family.
//!
//! A Qwen3-Reranker checkpoint *is* a Qwen3 causal LM — the arch
//! [`crate::models::qwen3`] already runs and parity-verifies — asked one
//! question per (query, document) pair: does this document meet the
//! requirement, "yes" or "no"? The relevance score is the probability the
//! model puts on "yes" against "no" at the first answer position,
//! `sigmoid(logit_yes - logit_no)` — the softmax over those two logits that
//! the model card computes, and what sentence-transformers' `LogitScore`
//! module reports before its identity activation.
//!
//! Only two of the 151 669 head rows are ever read, so the head projection
//! is restricted to them ([`LoadedQwen3::head_rows`]); everything else is the
//! ordinary prefill.
//!
//! The prompt is the checkpoint's own: the exact bytes its
//! `chat_template.jinja` renders, built here rather than through a Jinja
//! engine (a default build carries none), and checked against the template
//! file at load so a checkpoint that asks a different question is refused
//! instead of being scored with the wrong one.

use std::path::Path;

use burn::tensor::Device;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::import::ImportError;
use crate::models::qwen3::{self, LoadedQwen3};

/// The system turn every Qwen3-Reranker prompt opens with.
pub const SYSTEM: &str = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".";

/// The task instruction the checkpoints default to (their
/// `config_sentence_transformers.json` `query` prompt, and the template's
/// fallback).
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Everything before the instruction/query/document body.
fn prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
}

/// Everything after it: the user turn closes and the assistant opens with an
/// empty reasoning block, so the next token is the answer itself.
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The instruction/query/document body between [`prefix`] and [`SUFFIX`].
#[must_use]
pub fn body(instruction: &str, query: &str, document: &str) -> String {
    format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}")
}

/// The whole prompt for one pair, as the checkpoint's template renders it.
#[must_use]
pub fn prompt(instruction: &str, query: &str, document: &str) -> String {
    format!("{}{}{SUFFIX}", prefix(), body(instruction, query, document))
}

/// Longest pair, in tokens, before the document's tail is cut. The model
/// card's own example uses 8192; this is a bound on work and KV memory, and
/// reranking candidates are retrieval *chunks*, well under it.
pub const MAX_RERANK_TOKENS: usize = 4096;

/// One document's relevance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Index of the document in the caller's list.
    pub index: usize,
    /// `P(yes)`, in `[0, 1]`.
    pub relevance: f32,
    /// Prompt tokens this pair cost.
    pub tokens: usize,
}

/// A loaded Qwen3-Reranker.
pub struct Reranker {
    lm: LoadedQwen3,
    tokenizer: Tokenizer,
    yes: u32,
    no: u32,
    instruction: String,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    max_tokens: usize,
}

impl Reranker {
    /// Load the reranker checkpoint in `dir`.
    ///
    /// The yes/no token ids come from `1_LogitScore/config.json` when the
    /// checkpoint ships one and must agree with what `tokenizer.json` calls
    /// `yes` / `no`; the default instruction from
    /// `config_sentence_transformers.json`. A `chat_template.jinja` beside
    /// the weights must render the same system turn and body markers this
    /// module builds.
    ///
    /// # Errors
    ///
    /// [`ImportError`] when the weights or tokenizer fail to load, when the
    /// tokenizer has no `yes`/`no` token or disagrees with the declared ids,
    /// or when the template asks a different question.
    pub fn load_from_dir(dir: &Path, device: &Device) -> Result<Self, ImportError> {
        let tok_path = crate::import::required_file(dir, "tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| ImportError::Parse {
            file: tok_path.clone(),
            reason: e.to_string(),
        })?;
        let inconsistent = |file: &Path, reason: String| ImportError::Inconsistent {
            file: file.to_path_buf(),
            reason,
        };
        let id_of = |word: &str| {
            tokenizer.token_to_id(word).ok_or_else(|| {
                inconsistent(
                    &tok_path,
                    format!("tokenizer has no {word:?} token to score"),
                )
            })
        };
        let (yes, no) = (id_of("yes")?, id_of("no")?);

        let score_cfg = dir.join("1_LogitScore").join("config.json");
        if let Some(v) = read_json(&score_cfg)? {
            let declared = |k: &str| {
                v.get(k)
                    .and_then(Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok())
            };
            if declared("true_token_id").is_some_and(|d| d != yes)
                || declared("false_token_id").is_some_and(|d| d != no)
            {
                return Err(inconsistent(
                    &score_cfg,
                    format!("declared yes/no ids disagree with tokenizer.json ({yes}/{no})"),
                ));
            }
        }

        let template = dir.join("chat_template.jinja");
        if let Ok(text) = std::fs::read_to_string(&template) {
            let asks_ours = text.contains(SYSTEM)
                && text.contains("<Instruct>: ")
                && text.contains("<Query>: ")
                && text.contains("<Document>: ")
                && text.contains("<think>\n\n</think>\n\n");
            if !asks_ours {
                return Err(inconsistent(
                    &template,
                    "this chat template does not render the Qwen3-Reranker yes/no prompt".into(),
                ));
            }
        }

        let instruction = read_json(&dir.join("config_sentence_transformers.json"))?
            .and_then(|v| v.get("prompts")?.get("query")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| DEFAULT_INSTRUCTION.to_owned());

        let encode = |text: &str| -> Result<Vec<u32>, ImportError> {
            tokenizer
                .encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| ImportError::Parse {
                    file: tok_path.clone(),
                    reason: e.to_string(),
                })
        };
        let prefix_ids = encode(&prefix())?;
        let suffix_ids = encode(SUFFIX)?;

        let lm = qwen3::load_from_dir(dir, device)?;
        let positions = lm
            .config
            .max_position_embeddings
            .unwrap_or(MAX_RERANK_TOKENS);
        let max_tokens = positions.min(MAX_RERANK_TOKENS);
        if prefix_ids.len() + suffix_ids.len() + 16 > max_tokens {
            return Err(inconsistent(
                dir,
                "context too short to hold a prompt".into(),
            ));
        }
        Ok(Self {
            lm,
            tokenizer,
            yes,
            no,
            instruction,
            prefix_ids,
            suffix_ids,
            max_tokens,
        })
    }

    /// The instruction used when a caller gives none.
    #[must_use]
    pub fn default_instruction(&self) -> &str {
        &self.instruction
    }

    /// Longest pair, in tokens.
    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Token ids of the prompt for one pair, the body's tail cut to fit
    /// [`Self::max_tokens`] (the model card's `truncation='only_first'` on the
    /// formatted body), plus whether it was cut.
    ///
    /// # Errors
    ///
    /// The tokenizer's own error.
    pub fn prompt_ids(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> Result<(Vec<u32>, bool), String> {
        let instruction = instruction.unwrap_or(&self.instruction);
        let text = body(instruction, query, document);
        let mut body_ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        let room = self.max_tokens - self.prefix_ids.len() - self.suffix_ids.len();
        let truncated = body_ids.len() > room;
        body_ids.truncate(room);
        let mut ids =
            Vec::with_capacity(self.prefix_ids.len() + body_ids.len() + self.suffix_ids.len());
        ids.extend_from_slice(&self.prefix_ids);
        ids.extend_from_slice(&body_ids);
        ids.extend_from_slice(&self.suffix_ids);
        Ok((ids, truncated))
    }

    /// `P(yes)` for one (query, document) pair.
    ///
    /// # Errors
    ///
    /// A tokenizer error, or a failed device readback.
    pub async fn score(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
        device: &Device,
    ) -> Result<(f32, usize), String> {
        let (ids, _) = self.prompt_ids(query, document, instruction)?;
        let hidden = self.lm.last_hidden(&ids, device);
        let logits = self
            .lm
            .head_rows(hidden, &[self.no, self.yes], device)
            .into_data_async()
            .await
            .map_err(|e| format!("rerank readback: {e:?}"))?
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|e| format!("rerank readback: {e:?}"))?;
        let [no, yes] = logits[..] else {
            return Err(format!("expected 2 logits, got {}", logits.len()));
        };
        Ok((relevance(yes, no), ids.len()))
    }

    /// Score every document against `query`, most relevant first (ties keep
    /// the caller's order). `on_scored` sees each score as it lands and may
    /// stop the pass early with `Break` — a cancelled request should not
    /// keep the device busy for the rest of the list.
    ///
    /// # Errors
    ///
    /// The first document that fails to score.
    pub async fn rank<S: AsRef<str> + Sync>(
        &self,
        query: &str,
        documents: &[S],
        instruction: Option<&str>,
        device: &Device,
        mut on_scored: impl FnMut(&Score) -> std::ops::ControlFlow<()> + Send,
    ) -> Result<Vec<Score>, String> {
        let mut out = Vec::with_capacity(documents.len());
        for (index, doc) in documents.iter().enumerate() {
            let (relevance, tokens) = self.score(query, doc.as_ref(), instruction, device).await?;
            let s = Score {
                index,
                relevance,
                tokens,
            };
            let flow = on_scored(&s);
            out.push(s);
            if flow.is_break() {
                break;
            }
        }
        sort_by_relevance(&mut out);
        Ok(out)
    }
}

/// `softmax([no, yes])[1]`, computed as a sigmoid of the difference — the
/// same number without forming either exponential, so a large logit cannot
/// overflow it.
#[must_use]
pub fn relevance(yes: f32, no: f32) -> f32 {
    let d = yes - no;
    if d >= 0.0 {
        1.0 / (1.0 + (-d).exp())
    } else {
        let e = d.exp();
        e / (1.0 + e)
    }
}

/// Most relevant first; equal scores keep their input order.
pub fn sort_by_relevance(scores: &mut [Score]) {
    scores.sort_by(|a, b| {
        b.relevance
            .total_cmp(&a.relevance)
            .then(a.index.cmp(&b.index))
    });
}

fn read_json(path: &Path) -> Result<Option<Value>, ImportError> {
    let Ok(bytes) = std::fs::read(path) else {
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| ImportError::Parse {
            file: path.to_path_buf(),
            reason: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint template's render, written out by hand from
    /// `Qwen/Qwen3-Reranker-0.6B`'s `chat_template.jinja` (its trailing
    /// newline is stripped by Jinja's default `keep_trailing_newline=False`).
    #[test]
    fn the_prompt_is_the_templates_bytes() {
        let p = prompt("Find it", "What is X?", "X is Y.");
        assert_eq!(
            p,
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
             Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\
             <|im_end|>\n<|im_start|>user\n<Instruct>: Find it\n<Query>: What is X?\n<Document>: \
             X is Y.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn relevance_is_the_two_way_softmax_and_never_overflows() {
        assert!((relevance(0.0, 0.0) - 0.5).abs() < 1e-7);
        let (y, n) = (2.5f32, -1.0f32);
        let softmax = y.exp() / (y.exp() + n.exp());
        assert!((relevance(y, n) - softmax).abs() < 1e-6);
        assert!((relevance(1e4, -1e4) - 1.0).abs() < 1e-7);
        assert!(relevance(-1e4, 1e4).abs() < 1e-7);
        assert!(relevance(-1e4, 1e4).is_finite());
    }

    #[test]
    fn ranking_is_descending_and_stable() {
        let mut s = vec![
            Score {
                index: 0,
                relevance: 0.2,
                tokens: 1,
            },
            Score {
                index: 1,
                relevance: 0.9,
                tokens: 1,
            },
            Score {
                index: 2,
                relevance: 0.2,
                tokens: 1,
            },
            Score {
                index: 3,
                relevance: 0.5,
                tokens: 1,
            },
        ];
        sort_by_relevance(&mut s);
        let order: Vec<usize> = s.iter().map(|x| x.index).collect();
        assert_eq!(order, [1, 3, 0, 2]);
    }
}
