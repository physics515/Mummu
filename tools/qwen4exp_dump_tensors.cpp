// Full-precision per-op tensor dumper for llama.cpp: the reference half of
// the qwen4exp teacher-forced parity check
// (parity_qwen4exp::teacher_forced_ops_against_a_llama_cpp_dump).
//
// Why: llama-eval-callback / llama-debug print 3 values per axis at 4
// decimals, too coarse to tell a port bug from llama.cpp's activation
// rounding. This writes every observed tensor whole.
//
// Build and run inside the reference image (it ships gcc and libllama.so but
// no headers; fetch include/llama.h and ggml/include/{ggml,ggml-backend,
// ggml-alloc,ggml-cpu,ggml-opt,gguf}.h at the image's commit, 930e2fa59, into
// <work>/inc):
//
//   docker run --rm --name flashnext-dump -v <work>:/work --entrypoint /bin/bash \
//     ghcr.io/ggml-org/llama.cpp:full@sha256:5faf86f9... -c \
//     'cd /work && g++ -O2 -std=c++17 qwen4exp_dump_tensors.cpp -I inc -L/app \
//        -lllama -lggml -lggml-base -o dump'
//   docker run --rm --name flashnext-dump --user $(id -u):$(id -g) --memory 48g \
//     --memory-swap 48g -e LD_LIBRARY_PATH=/app -v <shards>:/models:ro -v <work>:/work \
//     --entrypoint /work/dump <image> /models/<first shard> /work/out '<name regex>' 16 \
//     primes <comma-separated prompt ids> moon <ids>
//
// usage: dump <model> <out dir> <regex> <n_threads> <leg name> <id,id,...> [<leg name> <ids>]...
// For each leg: fresh memory, one llama_decode of all ids (logits for the
// last). Pass 0 observes nothing, so the graph runs unsplit as the server
// runs it, and writes <out>/<leg>/logits-nocb.bin; pass 1 writes every
// observed tensor whose name matches <regex> as <out>/<leg>/<seq>.bin (raw
// little-endian values in ggml flat order, ne[0] fastest) indexed in
// <out>/<leg>/index.tsv (seq name occurrence type ne0 ne1 ne2 ne3), plus
// <out>/<leg>/logits.bin. Measured on qwen4exp (2026-09-16): both logits
// files equal the recorded fixture's top-10 logprobs within 1.7e-6 (its JSON
// rounding), so observing the graph does not change the realization.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct Dump {
    std::regex filter;
    std::string dir;
    FILE * index = nullptr;
    int seq = 0;
    std::map<std::string, int> occ;
    std::vector<uint8_t> buf;
    bool enabled = false;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (Dump *) ud;
    const bool want = d->enabled && std::regex_match(t->name, d->filter) &&
                      (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
    const size_t nbytes = ggml_nbytes(t);
    uint8_t * data = nullptr;
    if (host) {
        data = (uint8_t *) t->data;
    } else {
        d->buf.resize(nbytes);
        ggml_backend_tensor_get(t, d->buf.data(), 0, nbytes);
        data = d->buf.data();
    }
    const int occ = d->occ[t->name]++;
    const int seq = d->seq++;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%05d.bin", d->dir.c_str(), seq);
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    const size_t es = ggml_type_size(t->type);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        fwrite(data + off, es, 1, f);
    }
    fclose(f);
    fprintf(d->index, "%d\t%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\n", seq, t->name, occ, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    fflush(d->index);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 7 || (argc - 5) % 2 != 0) {
        fprintf(stderr, "usage: %s model outdir regex n_threads leg ids [leg ids]...\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out = argv[2];
    Dump d;
    d.filter = std::regex(argv[3]);
    const int n_threads = atoi(argv[4]);

    // the CPU variants are dynamic backends; the server loads them the same way (best score wins)
    ggml_backend_load_all_from_path("/app");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context failed\n");
        return 1;
    }
    fprintf(stderr, "flash_attn_type=%s\n", llama_flash_attn_type_name(cp.flash_attn_type));

    for (int a = 5; a + 1 < argc; a += 2) {
        const std::string leg = argv[a];
        std::vector<llama_token> ids;
        std::stringstream ss(argv[a + 1]);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            ids.push_back((llama_token) atoi(tok.c_str()));
        }
        d.dir = out + "/" + leg;
        std::string mk = "mkdir -p '" + d.dir + "'";
        if (system(mk.c_str()) != 0) {
            return 1;
        }
        // pass 0 observes nothing (one unsplit graph, as the server runs it); pass 1 dumps
        for (int pass = 0; pass < 2; ++pass) {
            d.enabled = pass == 1;
            d.index = fopen((d.dir + (pass == 1 ? "/index.tsv" : "/index-unused.tsv")).c_str(), "w");
            d.seq = 0;
            d.occ.clear();
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_get_one(ids.data(), (int32_t) ids.size());
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed for %s\n", leg.c_str());
                return 1;
            }
            fclose(d.index);
            const float * logits = llama_get_logits_ith(ctx, -1);
            FILE * f = fopen((d.dir + (pass == 1 ? "/logits.bin" : "/logits-nocb.bin")).c_str(), "wb");
            fwrite(logits, sizeof(float), n_vocab, f);
            fclose(f);
        }
        fprintf(stderr, "leg %s: %zu tokens, %d tensors dumped\n", leg.c_str(), ids.size(), d.seq);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
