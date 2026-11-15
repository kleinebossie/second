# Phase 8 Browser Export Verification: Second-1.0-2B-GGUF-Q4_K_M

- **Date**: 2026-11-08 16:30:37 UTC
- **Quantization Format**: `Q4_K_M`
- **Runtime Target**: `wllama` (WebAssembly)
- **Test Set Sample**: 100 prompts from `eval/testset-1.0.jsonl`
- **Release Gate Status**: **PASSED**

## 1. Post-Quantization Quality Gate Summary

| Metric | Unquantized Baseline (`RL-B2`) | Target Gate | Quantized (`Q4_K_M`) | Status |
|---|---|---|---|---|
| **Hard-Fail Rate (9 Verifiers)** | 2.00% | &le; 6.70% | **4.00%** | PASS |
| **Hard-Fail Delta** | 0.00% | &le; +2.00% | **+2.00%** | PASS |
| **Format Pass Rate** | 100.00% | &ge; 99.00% | **100.00%** | PASS |
| **In Target Length Band** | 100.00% | &ge; 80.00% | **99.00%** | PASS |
| **First-Person Diary Voice** | 0.00% | &le; 2.00% | **0.00%** | PASS |
| **Average Length** | 64.0 words | 40–80 words | **61.7 words** | TRACKED |

## 2. Hard Check Breakdown on 100 Prompts

| Hard Check | Baseline Failure (%) | Quantized Failure (%) | Delta (%) |
|---|---|---|---|
| `legal_moves` | 0.00% | 1.00% | +1.00% |
| `grounded_moves` | 0.00% | 0.00% | +0.00% |
| `piece_on_square` | 2.00% | 3.00% | +1.00% |
| `verdict_tone` | 0.00% | 0.00% | +0.00% |
| `format` | 0.00% | 0.00% | +0.00% |
| `beginner_jargon` | 0.00% | 0.00% | +0.00% |
| `coach_voice` | 0.00% | 0.00% | +0.00% |
| `opening_claims` | 0.00% | 0.00% | +0.00% |
| `relational_claims` | 0.00% | 0.00% | +0.00% |

## 3. Decision & Next Steps

- **Recommendation**: PASS: Quantization quality meets criteria (<= 6.70%). Proceed to Hugging Face upload.
- **Action**: Proceed to upload Second-1.0-2B-GGUF to Hugging Face Hub.
