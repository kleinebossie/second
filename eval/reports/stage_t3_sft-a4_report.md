# Stage T3 SFT v2 Evaluation Report: SFT-A4

**Model**: `Second-1.0-2B-SFT-v2`  
**Evaluation Set**: `eval/devset-1.0.jsonl` (150 items)  
**Gate Status**: **FAILED**

## Stage T3 Pass Gates

| Gate Dimension | Metric Target | Actual Result | Gate Outcome |
|---|---|---|---|
| **Hard Fail Rate** | &le; 6.0% | 16.0% | FAIL |
| **Length In-Band (Overall)** | &ge; 75.0% | 100.0% | PASS |
| **Length In-Band (Short)** | &ge; 70.0% | 100.0% | PASS |
| **Length In-Band (Long)** | &ge; 70.0% | 100.0% | PASS |
| **Key Point Score** | &ge; 0.65 | 0.9867 | PASS |
| **Mean Scalar Reward** | Informational | 0.6645 | TRACKED |

## Gate Decision

The model did not satisfy all Stage T3 pass gates. Review gate failure details before proceeding to Stage T4.
