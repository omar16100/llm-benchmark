#!/usr/bin/env python3
"""Benchmark GLM-5.1 with varying prompt lengths."""

import json
import time
import requests

API_URL = "http://localhost:1234/v1/chat/completions"

# base text block (~100 tokens per repetition)
BLOCK = (
    "The transformer architecture, introduced in the seminal paper 'Attention Is All You Need' "
    "by Vaswani et al. in 2017, revolutionized natural language processing by replacing recurrent "
    "neural networks with self-attention mechanisms. The key innovation was the multi-head attention "
    "mechanism, which allows the model to jointly attend to information from different representation "
    "subspaces at different positions. This parallel processing capability made transformers significantly "
    "faster to train than previous sequential architectures like LSTMs and GRUs. "
)

TARGET_TOKENS = [31, 500, 2000, 8000, 16000]
MAX_GEN = 256

def build_prompt(target_tok):
    if target_tok <= 50:
        return "Write a detailed explanation of how transformers work in machine learning, including attention mechanisms, positional encoding, and the encoder-decoder architecture."
    reps = (target_tok * 4) // len(BLOCK)  # rough char-to-token ratio ~4
    text = (BLOCK * max(reps, 1))[:target_tok * 4]
    return f"Summarize the following text in detail:\n\n{text}\n\nProvide a comprehensive summary."

def bench(target_tok):
    prompt = build_prompt(target_tok)
    payload = {
        "model": "GLM-5.1-UD-IQ3_XXS",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_GEN,
    }
    t0 = time.time()
    resp = requests.post(API_URL, json=payload, timeout=600)
    wall = time.time() - t0
    data = resp.json()
    t = data.get("timings", {})
    return {
        "target_tokens": target_tok,
        "prompt_n": t.get("prompt_n", 0),
        "prompt_per_second": round(t.get("prompt_per_second", 0), 2),
        "predicted_n": t.get("predicted_n", 0),
        "predicted_per_second": round(t.get("predicted_per_second", 0), 2),
        "wall_seconds": round(wall, 1),
    }

if __name__ == "__main__":
    print(f"{'Target':>8} {'Actual':>8} {'Prefill':>12} {'Decode':>12} {'Wall':>8}")
    print(f"{'tokens':>8} {'tokens':>8} {'tok/s':>12} {'tok/s':>12} {'sec':>8}")
    print("-" * 56)

    results = []
    for target in TARGET_TOKENS:
        r = bench(target)
        results.append(r)
        print(f"{r['target_tokens']:>8} {r['prompt_n']:>8} {r['prompt_per_second']:>12.2f} {r['predicted_per_second']:>12.2f} {r['wall_seconds']:>8.1f}")

    print("\n" + json.dumps(results, indent=2))
