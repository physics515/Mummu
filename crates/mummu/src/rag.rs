//! Retrieval-augmented generation, the model-free half: split documents
//! into chunks, keep their vectors in an index, find the nearest ones to a
//! query, and turn what was found into a grounded prompt.
//!
//! The models are elsewhere — [`crate::embed::Embedder`] makes the vectors,
//! [`crate::rerank::Reranker`] re-scores the candidates, any chat model in
//! the zoo answers. This module is the glue between them, and it is pure:
//! no device, no I/O beyond [`Index::save`] / [`Index::load`], so every
//! piece of it is unit-testable without weights.
//!
//! The index is exact (brute-force dot products over unit vectors). That is
//! the right tool at local scale: 100 000 chunks of a 1024-wide embedder
//! are 400 MB and a full scan is a few tens of milliseconds, with no recall
//! loss to tune away and no build step. An approximate index earns its
//! complexity two orders of magnitude further out than a personal corpus
//! goes.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// How [`chunk_text`] cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOptions {
    /// Longest chunk, in characters (not bytes, so a CJK text is not cut
    /// three times finer than an English one).
    pub max_chars: usize,
    /// Characters each chunk repeats from the end of the one before, so a
    /// sentence that straddles a cut is whole in at least one chunk.
    pub overlap_chars: usize,
}

impl Default for ChunkOptions {
    /// ~1 200 characters (~300 tokens of English) with a 150-character
    /// overlap: short enough that one chunk is about one thing — which is
    /// what an embedding of it can represent — and long enough to carry its
    /// own context into an answer.
    fn default() -> Self {
        Self {
            max_chars: 1200,
            overlap_chars: 150,
        }
    }
}

/// One piece of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    /// Byte range of `text` in the source document.
    pub start: usize,
    pub end: usize,
}

/// Where a cut prefers to land, best first: a paragraph break, a line
/// break, the end of a sentence, a clause, a word.
const BREAKS: &[&str] = &["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "];

/// Split `text` into chunks of at most `opts.max_chars` characters.
///
/// Each is cut at the best natural break in the back half of its window,
/// and consecutive chunks overlap by about `opts.overlap_chars`, opening on
/// a sentence. Whitespace-only chunks are dropped; a text with no breaks at
/// all is cut hard at the limit.
///
/// # Panics
///
/// When `max_chars` is 0 or `overlap_chars >= max_chars` (the cut would
/// never advance).
#[must_use]
pub fn chunk_text(text: &str, opts: ChunkOptions) -> Vec<Chunk> {
    assert!(opts.max_chars > 0, "chunk_text: max_chars must be positive");
    assert!(
        opts.overlap_chars < opts.max_chars,
        "chunk_text: overlap must be shorter than a chunk"
    );
    // Byte offset of every char boundary, plus the end.
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let nchars = bounds.len() - 1;
    let mut out = Vec::new();
    let mut c0 = 0usize; // char index where this chunk starts
    while c0 < nchars {
        let c_limit = (c0 + opts.max_chars).min(nchars);
        let c1 = if c_limit == nchars {
            nchars
        } else {
            best_break(text, &bounds, c0, c_limit)
        };
        let (b0, b1) = (bounds[c0], bounds[c1]);
        let piece = &text[b0..b1];
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            let lead = piece.len() - piece.trim_start().len();
            out.push(Chunk {
                text: trimmed.to_owned(),
                start: b0 + lead,
                end: b0 + lead + trimmed.len(),
            });
        }
        if c1 == nchars {
            break;
        }
        c0 = overlap_start(text, &bounds, c0, c1, opts.overlap_chars);
    }
    out
}

/// The breaks that end a thought: a paragraph, a line, a sentence.
const SENTENCE_BREAKS: &[&str] = &["\n\n", "\n", ". ", "? ", "! "];

/// Where the chunk after one ending at `c1` starts: about `overlap` chars
/// back, at the first sentence start inside that window, so the repeated
/// context opens on a whole thought.
///
/// When the window holds no sentence start but the cut itself fell on a
/// sentence end, the next chunk starts AT the cut — no overlap. Starting
/// mid-sentence is worse than repeating nothing: a chunk that opens on
/// "…yard on day 48 and was signed for by clerk number 2. Entry 17: …"
/// hands a reader the tail of the previous record as if it were this one's —
/// measured on this change, a 0.6B chat model answered the question about
/// entry 17 with entry 16's clerk. Only text with no sentence structure at
/// the cut falls back to a clause or word start. Always moves forward.
fn overlap_start(text: &str, bounds: &[usize], c0: usize, c1: usize, overlap: usize) -> usize {
    let floor = c0 + 1;
    let back = c1.saturating_sub(overlap).max(floor);
    if overlap == 0 || back >= c1 {
        return c1.max(floor);
    }
    let window = &text[bounds[back]..bounds[c1]];
    let start_after = |seps: &[&str]| {
        seps.iter().find_map(|sep| {
            let pos = window.find(sep)?;
            let c = bounds
                .binary_search(&(bounds[back] + pos + sep.len()))
                .ok()?;
            (c > c0 && c < c1).then_some(c)
        })
    };
    if let Some(c) = start_after(SENTENCE_BREAKS) {
        return c;
    }
    if SENTENCE_BREAKS
        .iter()
        .any(|sep| text[..bounds[c1]].ends_with(sep))
    {
        return c1;
    }
    start_after(&["; ", ", ", " "]).unwrap_or(c1)
}

/// The char index (exclusive end) to cut at: just after the best-ranked
/// break found in the back half of `[c0, c_limit)`, else `c_limit`.
fn best_break(text: &str, bounds: &[usize], c0: usize, c_limit: usize) -> usize {
    let half = c0 + (c_limit - c0) / 2;
    let window = &text[bounds[half]..bounds[c_limit]];
    for sep in BREAKS {
        if let Some(pos) = window.rfind(sep) {
            let cut_byte = bounds[half] + pos + sep.len();
            // Back to a char index (cut_byte is a boundary: separators are ASCII).
            if let Ok(c) = bounds.binary_search(&cut_byte)
                && c > c0
            {
                return c;
            }
        }
    }
    c_limit
}

/// What the index keeps beside each vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// The caller's document id; every chunk of a document shares it.
    pub doc: String,
    /// Chunk number within the document.
    pub chunk: usize,
    pub text: String,
    /// Free-form caller metadata (a title, a URL, a path), carried through
    /// to every hit.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub meta: serde_json::Value,
}

/// One search result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// Position in the index.
    pub slot: usize,
    /// Cosine similarity to the query.
    pub score: f32,
}

/// Errors from the index.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("vector has {got} dimensions, the index holds {want}")]
    Dims { want: usize, got: usize },
    #[error("vector is not finite")]
    NotFinite,
    #[error("index file {path}: {reason}")]
    File { path: String, reason: String },
    #[error("index would exceed {0} entries")]
    Full(usize),
}

/// Most entries one index holds (~400 MB of f32 at 1024 dims).
pub const MAX_ENTRIES: usize = 100_000;

/// An exact cosine index over unit vectors, with the embedder that made
/// them recorded so vectors from two different models are never compared.
#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    /// Name of the embedder every vector came from.
    pub model: String,
    dims: usize,
    vectors: Vec<f32>,
    entries: Vec<Entry>,
}

impl Index {
    /// An empty index for `dims`-wide vectors from `model`.
    ///
    /// # Panics
    ///
    /// When `dims` is 0.
    #[must_use]
    pub fn new(model: impl Into<String>, dims: usize) -> Self {
        assert!(dims > 0, "Index::new: zero-width vectors");
        Self {
            model: model.into(),
            dims,
            vectors: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entry(&self, slot: usize) -> Option<&Entry> {
        self.entries.get(slot)
    }

    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Distinct document ids, in first-insertion order.
    #[must_use]
    pub fn documents(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        self.entries
            .iter()
            .filter(|e| seen.insert(e.doc.as_str()))
            .map(|e| e.doc.as_str())
            .collect()
    }

    /// Add one chunk. The vector is re-normalized on the way in, so the
    /// index holds unit vectors whatever the embedder returned.
    ///
    /// # Errors
    ///
    /// [`RagError::Dims`] on a width mismatch, [`RagError::NotFinite`] for a
    /// NaN/inf/all-zero vector, [`RagError::Full`] past [`MAX_ENTRIES`].
    pub fn add(&mut self, vector: &[f32], entry: Entry) -> Result<usize, RagError> {
        if vector.len() != self.dims {
            return Err(RagError::Dims {
                want: self.dims,
                got: vector.len(),
            });
        }
        if self.entries.len() >= MAX_ENTRIES {
            return Err(RagError::Full(MAX_ENTRIES));
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(RagError::NotFinite);
        }
        self.vectors.extend(vector.iter().map(|v| v / norm));
        self.entries.push(entry);
        Ok(self.entries.len() - 1)
    }

    /// Remove every chunk of document `doc`; returns how many went.
    pub fn remove_doc(&mut self, doc: &str) -> usize {
        let before = self.entries.len();
        let mut keep_vectors = Vec::with_capacity(self.vectors.len());
        let mut keep_entries = Vec::with_capacity(self.entries.len());
        for (e, v) in self
            .entries
            .drain(..)
            .zip(self.vectors.chunks_exact(self.dims))
        {
            if e.doc != doc {
                keep_vectors.extend_from_slice(v);
                keep_entries.push(e);
            }
        }
        self.vectors = keep_vectors;
        self.entries = keep_entries;
        before - self.entries.len()
    }

    /// The `k` entries nearest `query` by cosine, best first (ties by slot).
    ///
    /// # Errors
    ///
    /// [`RagError::Dims`] on a width mismatch, [`RagError::NotFinite`] for
    /// a degenerate query.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, RagError> {
        if query.len() != self.dims {
            return Err(RagError::Dims {
                want: self.dims,
                got: query.len(),
            });
        }
        let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(RagError::NotFinite);
        }
        let mut hits: Vec<Hit> = self
            .vectors
            .chunks_exact(self.dims)
            .enumerate()
            .map(|(slot, v)| {
                let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                Hit {
                    slot,
                    score: dot / norm,
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.slot.cmp(&b.slot)));
        hits.truncate(k);
        Ok(hits)
    }

    /// Write the index to `path` atomically (a temp file, then a rename), so
    /// a crash mid-save leaves the previous index, never half of one.
    ///
    /// Format: `MRAG1\n`, a little-endian `u64` JSON header length, the JSON
    /// header (model, dims, entries), then `len * dims` little-endian `f32`.
    ///
    /// # Errors
    ///
    /// [`RagError::File`] on any I/O failure.
    pub fn save(&self, path: &Path) -> Result<(), RagError> {
        let err = |reason: String| RagError::File {
            path: path.display().to_string(),
            reason,
        };
        let header = serde_json::to_vec(&Header {
            model: self.model.clone(),
            dims: self.dims,
            entries: self.entries.clone(),
        })
        .map_err(|e| err(e.to_string()))?;
        let tmp = path.with_extension("tmp");
        {
            let file = std::fs::File::create(&tmp).map_err(|e| err(e.to_string()))?;
            let mut w = std::io::BufWriter::new(file);
            w.write_all(MAGIC).map_err(|e| err(e.to_string()))?;
            w.write_all(&(header.len() as u64).to_le_bytes())
                .map_err(|e| err(e.to_string()))?;
            w.write_all(&header).map_err(|e| err(e.to_string()))?;
            for v in &self.vectors {
                w.write_all(&v.to_le_bytes())
                    .map_err(|e| err(e.to_string()))?;
            }
            let file = w.into_inner().map_err(|e| err(e.to_string()))?;
            file.sync_all().map_err(|e| err(e.to_string()))?;
        }
        std::fs::rename(&tmp, path).map_err(|e| err(e.to_string()))
    }

    /// Read an index written by [`Self::save`]. Bounded: a header over 512
    /// MiB, more than [`MAX_ENTRIES`] entries, or a payload whose length is
    /// not exactly `entries * dims` floats is an error, never an allocation.
    ///
    /// # Errors
    ///
    /// [`RagError::File`] for an unreadable, foreign, truncated or
    /// inconsistent file.
    pub fn load(path: &Path) -> Result<Self, RagError> {
        let err = |reason: String| RagError::File {
            path: path.display().to_string(),
            reason,
        };
        let mut f = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
        let total = f.metadata().map_err(|e| err(e.to_string()))?.len();
        let mut magic = [0u8; 6];
        f.read_exact(&mut magic).map_err(|e| err(e.to_string()))?;
        if &magic != MAGIC {
            return Err(err("not a mummu RAG index".into()));
        }
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| err(e.to_string()))?;
        let hlen = u64::from_le_bytes(len);
        if hlen > (512 << 20) || hlen > total {
            return Err(err(format!("header length {hlen} is implausible")));
        }
        let mut header = vec![0u8; usize::try_from(hlen).map_err(|e| err(e.to_string()))?];
        f.read_exact(&mut header).map_err(|e| err(e.to_string()))?;
        let h: Header = serde_json::from_slice(&header).map_err(|e| err(e.to_string()))?;
        if h.dims == 0 || h.entries.len() > MAX_ENTRIES {
            return Err(err("dims is 0 or the entry count is over the cap".into()));
        }
        let want = (h.entries.len() as u64)
            .checked_mul(h.dims as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| err("payload size overflows".into()))?;
        let have = total - 14 - hlen;
        if have != want {
            return Err(err(format!(
                "payload is {have} bytes, the header says {want}"
            )));
        }
        let mut raw = vec![0u8; usize::try_from(want).map_err(|e| err(e.to_string()))?];
        f.read_exact(&mut raw).map_err(|e| err(e.to_string()))?;
        let vectors: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if vectors.iter().any(|v| !v.is_finite()) {
            return Err(err("a stored vector is not finite".into()));
        }
        Ok(Self {
            model: h.model,
            dims: h.dims,
            vectors,
            entries: h.entries,
        })
    }
}

const MAGIC: &[u8; 6] = b"MRAG1\n";

#[derive(Serialize, Deserialize)]
struct Header {
    model: String,
    dims: usize,
    entries: Vec<Entry>,
}

/// One passage handed to the generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage<'a> {
    pub text: &'a str,
    /// What the answer cites it as — a title, a file name, a URL.
    pub source: &'a str,
}

/// The system message that grounds a chat model in `passages`.
///
/// Numbered sources the answer cites as `[n]`, and an explicit instruction
/// to say so when the sources do not hold the answer rather than to fill
/// the gap from memory — the failure RAG exists to prevent.
#[must_use]
pub fn grounding_system_prompt(passages: &[Passage<'_>]) -> String {
    let mut s = String::from(
        "Answer the user's question in full sentences, using only the numbered \
         sources below, and cite each source you use inline, like this: [1]. If \
         the sources do not contain the answer, say that they do not, rather \
         than answering from memory.\n\nSources:\n",
    );
    for (i, p) in passages.iter().enumerate() {
        let _ = write!(s, "\n[{}] {}\n{}\n", i + 1, p.source, p.text.trim());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(max: usize, overlap: usize) -> ChunkOptions {
        ChunkOptions {
            max_chars: max,
            overlap_chars: overlap,
        }
    }

    #[test]
    fn a_short_text_is_one_chunk_with_exact_offsets() {
        let text = "  Hello, world.  ";
        let c = chunk_text(text, ChunkOptions::default());
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].text, "Hello, world.");
        assert_eq!(&text[c[0].start..c[0].end], "Hello, world.");
    }

    #[test]
    fn chunks_respect_the_limit_and_prefer_sentence_breaks() {
        let text = "One two three. Four five six. Seven eight nine. Ten eleven twelve.";
        let c = chunk_text(text, opts(32, 0));
        assert!(c.len() > 1);
        for ch in &c {
            assert!(ch.text.chars().count() <= 32, "{:?}", ch.text);
            assert_eq!(&text[ch.start..ch.end], ch.text);
        }
        assert!(c[0].text.ends_with('.'), "{:?}", c[0].text);
    }

    #[test]
    fn every_char_is_covered_and_overlap_repeats_context() {
        let text = (0..200)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let c = chunk_text(&text, opts(100, 30));
        // Coverage: the chunks, in order, reach from the start to the end.
        assert_eq!(c[0].start, 0);
        assert_eq!(c.last().unwrap().end, text.len());
        for pair in c.windows(2) {
            assert!(pair[1].start <= pair[0].end, "gap between chunks");
            assert!(pair[1].start > pair[0].start, "no progress");
        }
        // Overlap: consecutive chunks share text.
        assert!(c.windows(2).any(|p| p[1].start < p[0].end));
    }

    #[test]
    fn multibyte_text_is_cut_on_char_boundaries() {
        let text = "日本語のテキスト。".repeat(40);
        let c = chunk_text(&text, opts(25, 5));
        assert!(c.len() > 1);
        for ch in &c {
            assert!(ch.text.chars().count() <= 25);
            assert_eq!(&text[ch.start..ch.end], ch.text);
        }
    }

    /// The overlap opens on a whole sentence, never on the tail of the one
    /// before (the record-boundary failure `overlap_start` documents).
    #[test]
    fn an_overlapping_chunk_starts_on_a_sentence() {
        let text = (0..30)
            .map(|i| {
                format!(
                    "Entry {i}: crate {i} left on day {} for clerk {}.",
                    i * 3,
                    i % 7
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let c = chunk_text(&text, opts(200, 60));
        assert!(c.len() > 3);
        for ch in &c[1..] {
            assert!(ch.text.starts_with("Entry "), "{:?}", &ch.text[..30]);
        }
    }

    #[test]
    fn a_text_without_breaks_is_cut_hard() {
        let text = "x".repeat(250);
        let c = chunk_text(&text, opts(100, 0));
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].text.len(), 100);
    }

    #[test]
    fn whitespace_only_input_has_no_chunks() {
        assert_eq!(
            chunk_text("   \n\n  ", ChunkOptions::default()),
            [] as [crate::rag::Chunk; 0]
        );
        assert_eq!(
            chunk_text("", ChunkOptions::default()),
            [] as [crate::rag::Chunk; 0]
        );
    }

    fn entry(doc: &str, chunk: usize) -> Entry {
        Entry {
            doc: doc.into(),
            chunk,
            text: format!("{doc}#{chunk}"),
            meta: serde_json::Value::Null,
        }
    }

    #[test]
    fn search_finds_the_nearest_and_normalizes_on_the_way_in() {
        let mut ix = Index::new("m", 3);
        ix.add(&[10.0, 0.0, 0.0], entry("a", 0)).unwrap();
        ix.add(&[0.0, 1.0, 0.0], entry("b", 0)).unwrap();
        ix.add(&[0.7, 0.7, 0.0], entry("c", 0)).unwrap();
        let hits = ix.search(&[1.0, 0.1, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(ix.entry(hits[0].slot).unwrap().doc, "a");
        assert_eq!(ix.entry(hits[1].slot).unwrap().doc, "c");
        assert!(hits[0].score <= 1.0 + 1e-6 && hits[0].score > hits[1].score);
    }

    #[test]
    fn bad_vectors_are_refused() {
        let mut ix = Index::new("m", 2);
        assert!(matches!(
            ix.add(&[1.0, 2.0, 3.0], entry("a", 0)),
            Err(RagError::Dims { want: 2, got: 3 })
        ));
        assert!(matches!(
            ix.add(&[0.0, 0.0], entry("a", 0)),
            Err(RagError::NotFinite)
        ));
        assert!(matches!(
            ix.add(&[f32::NAN, 1.0], entry("a", 0)),
            Err(RagError::NotFinite)
        ));
        assert!(ix.search(&[1.0], 1).is_err());
    }

    #[test]
    fn remove_doc_drops_only_that_documents_chunks() {
        let mut ix = Index::new("m", 2);
        ix.add(&[1.0, 0.0], entry("a", 0)).unwrap();
        ix.add(&[0.0, 1.0], entry("b", 0)).unwrap();
        ix.add(&[1.0, 1.0], entry("a", 1)).unwrap();
        assert_eq!(ix.remove_doc("a"), 2);
        assert_eq!(ix.len(), 1);
        assert_eq!(ix.documents(), vec!["b"]);
        let hits = ix.search(&[0.0, 1.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn save_then_load_round_trips_and_rejects_damage() {
        let dir = std::env::temp_dir().join(format!("mummu-rag-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ix.mrag");
        let mut ix = Index::new("harrier", 2);
        ix.add(&[1.0, 0.0], entry("a", 0)).unwrap();
        let mut e = entry("b", 3);
        e.meta = serde_json::json!({"title": "B"});
        ix.add(&[0.6, 0.8], e).unwrap();
        ix.save(&path).unwrap();
        assert_eq!(Index::load(&path).unwrap(), ix);

        // Truncate the payload by one byte: refused, not misread.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(Index::load(&path).is_err());
        std::fs::write(&path, b"not an index at all").unwrap();
        assert!(Index::load(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_grounding_prompt_numbers_and_names_every_source() {
        let p = grounding_system_prompt(&[
            Passage {
                text: "Paris is the capital of France.",
                source: "geo.txt",
            },
            Passage {
                text: "  Berlin is in Germany. ",
                source: "de.md",
            },
        ]);
        assert!(p.contains("[1] geo.txt\nParis is the capital of France.\n"));
        assert!(p.contains("[2] de.md\nBerlin is in Germany.\n"));
        assert!(p.contains("say that they do not"));
    }
}
