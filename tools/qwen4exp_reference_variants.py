#!/usr/bin/env python3
"""Measure how far llama.cpp's OWN first forward moves on the qwen4exp parity
prompts under mathematically equivalent settings, and write the fixture the
replay test `parity_qwen4exp::llama_cpps_equivalent_settings_replay_through_the_gate_as_recorded`
reads.

Why: the recorded reference (tests/fixtures/qwen4exp_ud_q4kxl_parity.json)
is ONE realization of llama.cpp's CPU arithmetic, which quantizes activations
per dot product and accumulates flash attention in f16. Over 48 layers that
rounding is chaotic, so a kernel choice that changes only float summation
order (--no-repack) or the attention kernel (-fa off) moves the tail logprobs
by up to ~1 nat. This script records those realizations so the gate's
tolerance can be judged against the reference's own spread.

Each variant starts the SAME image with the fixture's server args plus the
variant flags (container name flashnext-refvar-<tag>, 48 GiB cap, CPU only),
asks each leg for n_predict=1 with n_probs=10 (the first forward only), and
removes the container. Needs ~48 GB free RAM per run, one run at a time, and
the shards on NVMe.

usage: python3 tools/qwen4exp_reference_variants.py <shard dir> \
           > crates/mummu/tests/fixtures/qwen4exp_reference_variants.json
"""

import json
import subprocess
import sys
import time
import urllib.request

REPO = __file__.rsplit('/tools/', 1)[0]
FIXTURE = f'{REPO}/crates/mummu/tests/fixtures/qwen4exp_ud_q4kxl_parity.json'
IMAGE = ('ghcr.io/ggml-org/llama.cpp:full@sha256:'
         '5faf86f95747fbb40014a8b28c505d8ec0da3d983d4d368a6650bc056f252688')
FIRST_SHARD = 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
PORT = 18660

# (tag, extra llama-server args). v0 must reproduce the fixture bit for bit;
# the rest only change kernels whose math is identical.
VARIANTS = [
    ('fixture-args', []),
    ('no-ctx-checkpoints', ['--ctx-checkpoints', '0']),
    ('no-repack', ['--no-repack']),
    ('fa-off', ['-fa', 'off']),
    ('no-repack-fa-off', ['--no-repack', '-fa', 'off']),
    ('kv-f32', ['-ctk', 'f32', '-ctv', 'f32']),
    ('threads-8', ['-t', '8']),
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run_variant(shard_dir, tag, extra, legs):
    name = f'flashnext-refvar-{tag}'
    cmd = ['docker', 'run', '-d', '--rm', '--name', name, '--memory', '48g', '--memory-swap', '48g',
           '-p', f'127.0.0.1:{PORT}:8080', '-v', f'{shard_dir}:/models:ro',
           '--entrypoint', '/app/llama-server', IMAGE,
           '-m', f'/models/{FIRST_SHARD}', '--host', '0.0.0.0', '--port', '8080',
           '-c', '2048', '-t', '16', '-ngl', '0', '-np', '1', '--no-webui'] + extra
    subprocess.run(cmd, check=True, capture_output=True)
    base = f'http://127.0.0.1:{PORT}'
    try:
        for _ in range(450):  # bounded: 450 x 2 s
            try:
                if json.loads(urllib.request.urlopen(f'{base}/health', timeout=10).read()).get('status') == 'ok':
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise SystemExit(f'{tag}: server never became healthy')
        out = {}
        for leg in legs:
            req = dict(leg['request'], n_predict=1)
            r = urllib.request.Request(f'{base}/completion', data=json.dumps(req).encode(),
                                       headers={'Content-Type': 'application/json'})
            v = json.loads(urllib.request.urlopen(r, timeout=1800).read())
            if v.get('tokens_evaluated') != len(leg['prompt_ids']):
                raise SystemExit(f'{tag}/{leg["name"]}: evaluated {v.get("tokens_evaluated")} tokens')
            out[leg['name']] = [{'id': t['id'], 'logprob': t['logprob']}
                                for t in v['completion_probabilities'][0]['top_logprobs']]
            log(tag, leg['name'], [(t['id'], round(t['logprob'], 3)) for t in out[leg['name']][:5]])
        return out
    finally:
        subprocess.run(['docker', 'stop', name], capture_output=True)


def main():
    shard_dir = sys.argv[1]
    fx = json.load(open(FIXTURE))
    variants = []
    for tag, extra in VARIANTS:
        variants.append({'tag': tag, 'extra_args': extra, 'first_forward_top': run_variant(shard_dir, tag, extra, fx['legs'])})
    json.dump({
        'format': 1,
        'reference': fx['reference'],
        'note': 'first-forward top-10 (n_predict=1) of the fixture legs under the fixture server args plus extra_args',
        'variants': variants,
    }, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
