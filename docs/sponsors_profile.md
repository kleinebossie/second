# GitHub Sponsors Setup Kit for Second

This document contains everything needed to set up your GitHub Sponsors profile for **Second** (`kleinebossie/second`).

---

## 1. Quick Profile Details

| Field | Content |
|---|---|
| **Sponsorship Account** | Individual (`kleinebossie`) or Organization |
| **Short Description (Bio)** *(Max 160 chars)* | `Creator of Second: Your local chess second. Open-source, private, on-device chess explanation models and browser extension.` |
| **Repository with Sponsors button** | Enabled via `.github/FUNDING.yml` |

---

## 2. Profile Overview Markdown

*Copy and paste the markdown below into the Introduction field on GitHub Sponsors:*

```markdown
### Hi, I'm Joe! 👋 I'm building Second.

**Second** is your local chess second — an open-source chess explanation system that runs directly on your device.

Standard chess engines calculate numbers (`+1.4`, `-3.2`) and long variations. They do not explain **ideas**.
Commercial platforms lock chess coaching behind monthly subscriptions and cloud servers.

I'm building an open alternative: **Second**.

- **Open Weights and Open Source:** Fine-tuned and reinforcement-learned (GRPO) Qwen 3.5 models (`Second-1.0-2B`, plus `0.8B`, `4B`, and `9B`).
- **Private and On-Device:** Runs language models locally via WebAssembly (`wllama`) and Stockfish WebAssembly in a Chrome side panel. Turn off optional Lichess Cloud Eval in Settings for offline position evaluation.
- **Fair Play First:** Active only on finished games, analysis boards, and studies — never during live games or puzzles.
- **Zero Subscriptions:** Free under Apache 2.0 and GPLv3.

---

### Why Sponsor?

Second has no venture funding, no ads, and no paywalls.
We publish everything freely for the chess and open-source AI community.

Training and evaluating language models still requires compute:

1. **GPU Compute:** LoRA SFT and token-level GRPO training runs require rented GPU compute (RunPod, Lambda, Colab Pro). Training and scaling `0.8B`, `4B`, and `9B` models needs dedicated GPU hours.
2. **Automated Evaluation:** Automated judge runs on our 300-position benchmark use API tokens.
3. **Packaging and Distribution:** Covers Chrome Web Store fees, Edge Add-ons, and Hugging Face hosting.

Every dollar goes directly into compute, training runs, and keeping Second free for all players.

Thank you for supporting independent, open-source AI.
```

---

## 3. Recommended Sponsorship Tiers

### Monthly Tiers

#### Tier 1: $3 / month — Pawn Supporter
- **Description:**
  > Support ongoing maintenance of your local chess second.
  >
  > - Sincere thanks for backing open-source chess AI
  > - Your name listed in `SPONSORS.md`
  > - GitHub Sponsor badge on your profile
- **Welcome Message:**
  > Thank you for backing Second. Your support helps cover project infrastructure and evaluation costs.

#### Tier 2: $10 / month — Knight Explorer
- **Description:**
  > Fund active evaluation runs and extension test iterations.
  >
  > - All Pawn perks
  > - Early access to pre-release GGUF weights and experimental extension test builds
  > - Input on feature requests and roadmap priorities
- **Welcome Message:**
  > Welcome as a Knight Explorer. Thank you for helping bring local chess explanations to everyone.

#### Tier 3: $25 / month — Bishop Strategist
- **Description:**
  > Fund dedicated GPU training hours for SFT and GRPO RL runs.
  >
  > - All Knight perks
  > - Funds ~10–15 hours of rented cloud GPU time
  > - Shoutout in release notes and Hugging Face model cards
- **Welcome Message:**
  > Thank you for backing Second as a Bishop Strategist. Your support directly powers our GPU training runs.

#### Tier 4: $50 / month — Rook Patron
- **Description:**
  > Make large model training (`4B` and `9B`) and multi-browser support possible.
  >
  > - All Bishop perks
  > - Direct feedback channel for your suggested test positions
  > - Dedicate a test benchmark item or release checkpoint
- **Welcome Message:**
  > Thank you for becoming a Rook Patron. Your support accelerates larger models and multi-browser work.

#### Tier 5: $250 / month — Queen / Corporate Sponsor
- **Description:**
  > For clubs, organizations, and partners who champion open-source chess tools.
  >
  > - All Rook perks
  > - Prominent logo and link in `README.md`, `SPONSORS.md`, and extension documentation
- **Welcome Message:**
  > Thank you for your institutional support of Second. Please reach out so we can place your logo and link.

---

### One-Time Tiers

- **$5 — Blunder Recovery:** A one-time contribution to support development.
- **$25 — GPU Training Hour:** Funds a few hours of cloud GPU compute for RL experiments.
- **$100 — Checkpoint Sponsor:** Funds a full fine-tuning and evaluation run.
- **$500 — Release Patron:** Acknowledgment in the release notes of our next major milestone.

---

## 4. Sponsorship Goals

- **Goal 1: $50 / month — Evaluation and Infrastructure Baseline**
  > Covers monthly automated judge calls on our 300-item benchmark and domain/store fees.
- **Goal 2: $200 / month — Dedicated Training Compute for 0.8B and 4B**
  > Funds regular cloud GPU compute to train and evaluate `Second-1.0-0.8B` and `Second-1.0-4B`.
- **Goal 3: $500 / month — Continuous Release Loop and Multi-Platform**
  > Funds automated retraining pipelines and Firefox / Safari extension research.

---

## 5. Sponsor Update Template

```markdown
**Title:** Second Project Update: [Milestone Name]

Hello everyone,

Here is what your sponsorship funded this month:
- **Model Progress:** [e.g. Certified Second-1.0-2B RC with 100% counterfactual accuracy and 99.7% in-band length rate]
- **Extension Progress:** [e.g. WebAssembly integration running local wllama Q4_K_M weights]
- **Compute Spend:** [e.g. 45 GPU hours on RunPod for GRPO v2 token-level DAPO training]

Next steps:
- [Next milestone, e.g. Second-1.0-0.8B lightweight model training]

Thank you for making this possible.
```
