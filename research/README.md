# Research Reports and Critical Analysis

This directory holds the post-training research reports and our critical analysis for the Second 2B model iteration.

## Files

| File | Description |
|---|---|
| [00-research-brief.md](00-research-brief.md) | The project brief and Phase 6 failure summary that we gave to the research tool. |
| [01-post-training-methods-for-small-models.md](01-post-training-methods-for-small-models.md) | Research report on post-training methods, adapter capacity, DAPO, S-GRPO, GR3, MO-GRPO, and teacher distillation. |
| [02-length-collapse-and-reward-hacking.md](02-length-collapse-and-reward-hacking.md) | Research report on length collapse, reward hacking, Constrained Markov Decision Processes (CMDP), and length-controlled evaluation. |
| [03-training-data-strategies.md](03-training-data-strategies.md) | Research report on synthetic teacher data, engine grounding, data filtering (IFD, S-IFD, DEITA), and reasoning templates. |
| [04-evaluation-framework.md](04-evaluation-framework.md) | Research report on evaluation metrics, LLM judge biases, human calibration, and release gates. |
| [analysis.md](analysis.md) | Critical analysis of all four reports against our Phase 5 and Phase 6 data, with adopt/test/defer/reject decisions. |

## Implementation Plan

See [TRAINING_PLAN.md](../TRAINING_PLAN.md) in the repository root for the agent execution plan that implements the accepted findings.
