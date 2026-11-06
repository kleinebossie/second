# Second Project: Frontier Training Context for NotebookLM

## 1. Project Goal
Second is an open-source chess coach model fine-tuned on Qwen 3.5 2B.
The goal is to explain chess positions and moves in simple words for beginner and club players.
It must never hallucinate illegal moves or pieces.
It must not sound like a computer chess engine.

## 2. Model Architecture and Pipeline
- **Base Model**: Qwen/Qwen3.5-2B (Instruct).
- **Phase 5 (SFT)**: LoRA fine-tuning (r=32, alpha=32) on human Lichess study comments.
  - Used non-thinking prompt template (empty `<think>\n\n</think>`).
- **Phase 6 (GRPO)**: Group Relative Policy Optimization using Hugging Face TRL on an NVIDIA L4 GPU.
  - Group size: 8 generations per prompt.
  - Max answer length: 300 tokens.
  - Temperature: 0.9.
  - Learning rate: 5e-6, KL beta: 0.02.
  - Steps: 450 steps with early stopping.

## 3. Current Phase 6 Results and Failure Modes
- **Format Pass Rate**: 100.0% (Clean plain text, no headings).
- **Rule Hard-Fail Rate**: 7.7% (Improved from 11.3% in SFT and 100.0% in Base Qwen).
- **Rule Mean Reward**: 0.5428 (Improved from 0.4772 in SFT and -1.0 in Base Qwen).
- **Gemini Judge Rubric**:
  - Clarity: 3.54 / 5.0
  - Level Fit: 3.53 / 5.0
  - Correctness: 2.16 / 5.0
  - Usefulness: 1.87 / 5.0
  - Overall: 2.78 / 5.0
- **Pairwise vs SFT Baseline**: 53 wins, 50 losses, 197 ties (83.3% win+tie rate).
- **Pairwise vs Base Qwen**: 13 wins, 79 losses, 208 ties. Blind judges preferred long verbose text despite base model chess hallucinations.
- **Pairwise vs Gemini Flash**: 0 wins, 295 losses, 5 ties.

## 4. Key Bottlenecks to Solve
1. **Length Collapse**:
   - The policy learned to write extremely short answers (average 30.7 words).
   - 80% of outputs fall below the target length band.
   - The model writes short answers to avoid negative penalties from rule checkers.
2. **Low Usefulness and Pedagogical Depth**:
   - The model avoids naming tactical concepts (forks, pins, skewers, deflections).
   - Explanations lack instructional substance (e.g., "I think this is a good move, but I would have preferred d6.").
3. **No Chain-of-Thought (CoT) Reasoning**:
   - The 2B model had zero thinking budget (`<think>\n\n</think>` was empty).
   - A 2B parameter network struggles to verify legal moves, board geometry, and pedagogical narrative in a single forward pass without scratchpad tokens.
4. **SFT Training Data Limitations**:
   - Raw human Lichess comments are noisy, inconsistent, and often omit move explanations.
   - Garbage in, garbage out: RL cannot optimize quality that does not exist in the initial policy distribution.
