#!/usr/bin/env python3
"""HuggingFace reference for the retrieval tier: embeddings and rerank scores.

Runs the checkpoints through `transformers` exactly as their model cards do,
in f32 with eager attention, and writes the fixture
`crates/mummu/tests/fixtures/retrieval_reference.json` that
`crates/mummu/tests/real_retrieval.rs` replays against mummu's from-scratch
path. Three things are recorded per item, so a failure says WHERE it is:

1. the token ids (tokenization parity, checked exactly);
2. for an embedder, the unit vector (last-token pooled, L2-normalized: the
   sentence-transformers `Pooling(lasttoken)` + `Normalize` pipeline the
   checkpoint's `modules.json` declares, written out by hand because
   sentence-transformers is not in the reference image);
3. for a reranker, the yes/no logits at the answer position and P(yes),
   computed with the model card's own prefix/body/suffix construction — and
   cross-checked against one tokenization of the whole rendered template, so
   the fixture also proves the split tokenization is the template's.

One document is longer than 512 tokens, so mummu's chunked prefill (it pools
through the KV cache 512 tokens at a time) is held to the one-shot reference.

The reference image is llama.cpp's `full` image, which ships torch (CPU) and
transformers for its conversion scripts:

    docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp \\
      -v "$HOME/.cache/mummu-models:/models:ro" -v "$PWD:/work" -w /work \\
      --entrypoint python3 ghcr.io/ggml-org/llama.cpp:full \\
      tools/retrieval_reference.py /models > crates/mummu/tests/fixtures/retrieval_reference.json

`/models` must hold `harrier-oss-v1-0.6b/`, `qwen3-reranker-0.6b/` and
`qwen3-reranker-4b/` (the catalog names; `ModelSpec::fetch` lays them out).
"""

import json
import sys

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.manual_seed(0)

QUERIES = [
    "What is the capital of France?",
    "How do plants turn sunlight into energy?",
    "¿Cuál es el río más largo del mundo?",
]

LONG = " ".join(
    f"Entry {i}: the depot ledger records that crate {i} of copper wire left the "
    f"northern yard on day {i * 3} and was signed for by clerk number {i % 7}."
    for i in range(30)
)

DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Photosynthesis is the process by which green plants use sunlight to "
    "synthesize nutrients from carbon dioxide and water.",
    "The Nile is often regarded as the longest river in the world.",
    "Quarterly revenue grew twelve percent year over year.",
    LONG,
]

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

RERANK_PAIRS = [(q, d) for q in range(len(QUERIES)) for d in range(len(DOCUMENTS))]

SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def r7(xs):
    return [float(f"{x:.8g}") for x in xs]


def embedder(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(
        path, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    prompts = json.load(open(f"{path}/config_sentence_transformers.json"))["prompts"]
    query_prompt = prompts.get("query", prompts.get("web_search_query"))
    items = [("query", q, query_prompt + q) for q in QUERIES] + [
        ("document", d, d) for d in DOCUMENTS
    ]
    out = []
    for kind, text, prompted in items:
        enc = tok(prompted, return_tensors="pt")
        hidden = model(**enc).last_hidden_state[0, -1]
        vec = torch.nn.functional.normalize(hidden, p=2, dim=0)
        out.append(
            {
                "kind": kind,
                "text": text,
                "ids": enc["input_ids"][0].tolist(),
                "vector": r7(vec.tolist()),
            }
        )
    return {"query_prompt": query_prompt, "items": out}


def reranker(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    yes = tok.convert_tokens_to_ids("yes")
    no = tok.convert_tokens_to_ids("no")
    pre = tok.encode(PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    out = []
    for qi, di in RERANK_PAIRS:
        body = f"<Instruct>: {INSTRUCTION}\n<Query>: {QUERIES[qi]}\n<Document>: {DOCUMENTS[di]}"
        ids = pre + tok.encode(body, add_special_tokens=False) + suf
        whole = tok.encode(PREFIX + body + SUFFIX, add_special_tokens=False)
        assert ids == whole, "split tokenization differs from the whole template's"
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        pair = torch.stack([logits[no], logits[yes]])
        p_yes = torch.softmax(pair, dim=0)[1].item()
        out.append(
            {
                "query": qi,
                "document": di,
                "ids": ids,
                "yes_logit": float(f"{logits[yes].item():.8g}"),
                "no_logit": float(f"{logits[no].item():.8g}"),
                "relevance": float(f"{p_yes:.8g}"),
            }
        )
    return {"yes_id": yes, "no_id": no, "pairs": out}


def main():
    root = sys.argv[1]
    fixture = {
        "generator": "tools/retrieval_reference.py",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "queries": QUERIES,
        "documents": DOCUMENTS,
        "instruction": INSTRUCTION,
        "harrier-oss-v1-0.6b": embedder(f"{root}/harrier-oss-v1-0.6b"),
        "qwen3-reranker-0.6b": reranker(f"{root}/qwen3-reranker-0.6b"),
        "qwen3-reranker-4b": reranker(f"{root}/qwen3-reranker-4b"),
    }
    json.dump(fixture, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
