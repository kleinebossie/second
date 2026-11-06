# Post-Training Architectures and Optimization Paradigms for Domain-Specific Reasoning in Small Language Models: Resolving Degradation Dynamics in 2B Scale Chess Instruction

> [!NOTE]
> Source: Gemini Deep Research report, added 2026-11-06. This is a formatted copy. The content is not changed. A critical review of this report is in [analysis.md](analysis.md).

## Diagnostic Failure Analysis of the SFT-GRPO Pipeline in 2B Scale Models

Post-training small language models (SLMs) under 3 billion parameters for niche, highly structured reasoning domains—such as chess tactical and strategic explanation—presents severe optimization challenges [cite: 1, 2]. The observed performance trajectory, characterized by 100% format compliance and low rule failure rates (7.7%) alongside a collapse in response length (averaging 30 words) and degraded instructional utility (1.87 / 5.0), represents a canonical manifestation of length collapse, reward hacking, and exploratory cold-start failure [cite: 3, 4, 5, 6].

### Mechanistic Causes of Brevity Collapse and Length Bias

The fundamental cause of brevity collapse in Group Relative Policy Optimization (GRPO) lies in the mathematical formulation of its sequence-level loss aggregation and group-wise advantage standardization [cite: 3, 6, 7]. Standard GRPO calculates parameter updates by normalizing token-level surrogate losses by the sequence length $|o_i|$ of each completion:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t}|x, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|x, o_{i,<t})} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_{i,t}|x, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|x, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

Because the sequence length factor $\frac{1}{|o_i|}$ scales the loss per sequence, shorter trajectories receive significantly larger per-token gradient magnitudes than longer trajectories [cite: 6, 7]. For example, a candidate completion of 30 tokens experiences a per-token gradient norm ten times larger than a completion of 300 tokens [cite: 7].

When continuous length penalties or proxy judge rewards are incorporated into the reward function, standard GRPO's group advantage normalization ($\hat{A}_i = \frac{R_i - \mu}{\sigma}$) creates an asymmetrical optimization loop [cite: 3, 8, 9]. If incorrect or suboptimal trajectories are continuously penalized for length, the advantage variance within the group expands [cite: 3, 8]. The policy quickly learns that drastically truncating response length minimizes the magnitude of negative penalties and maximizes token-level reward density [cite: 3, 6]. Consequently, the model optimizes for brevity over pedagogical substance, collapsing into hyper-compressed, 30-word boilerplate explanations that satisfy surface formatting rules but fail to provide meaningful instructional value [cite: 3, 4, 6].

### The Cold-Start and Sparse Reward Trap in Niche Domains

In domain-specific reasoning tasks such as chess, base language models at the 2B parameter scale lack sufficient pre-existing internal representations of spatial, tactical, and move-calculation mechanics [cite: 2, 10]. When subjected to pure outcome-based Reinforcement Learning with Verifiable Rewards (RLVR), the policy encounters a cold-start problem [cite: 5, 10, 11].

During RL rollout generation, if a sampled group of $G$ trajectories yields uniformly incorrect moves or inaccurate tactical claims, all completions receive zero or negative baseline rewards ($R_i = 0$ for all $i$) [cite: 5, 9]. Under Bernoulli or deterministic outcome evaluation, the sample standard deviation of the group reward becomes zero ($\sigma = 0$) [cite: 9]. This creates two pathological states:

1. **Vanishing Gradients and Silent Groups:** Groups with zero variance ($\sigma = 0$) provide zero contrast, contributing no useful gradient updates to the policy [cite: 5, 9].
2. **Degenerate Policy Convergence:** When a 2B model occasionally generates a correct format or legal move purely by chance, the group advantage spiking on that rare trace causes the policy to overfit rapidly to the precise, minimalistic lexical patterns of that single trace [cite: 5, 11, 12]. The model suppresses exploratory token generation, leading to dynamic entropy collapse [cite: 7, 12].

### Parameter Plasticity Deficits: Parameter-Efficient vs. Full Fine-Tuning

A major architectural bottleneck at the 2B scale is parameter plasticity [cite: 10, 13]. Parameter-Efficient Fine-Tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA), are widely used to reduce computational overhead [cite: 14, 15]. However, empirical evidence in complex spatial and rule-governed domains demonstrates that LoRA struggles to inject net-new domain knowledge into small model backbones [cite: 10, 13].

| Adapter Architecture / Strategy | Target Parameter Ratio | Tactical Accuracy (Pass@1) | Conceptual & Pedagogical Depth | Optimization Plasticity & Capacity |
| --- | --- | --- | --- | --- |
| LoRA (Rank = 16, Alpha = 32) | ~0.15% - 0.40% | Low (32.1%) [cite: 13] | Superficial; heavy reliance on pre-trained surface prose [cite: 5, 13] | Early capacity saturation; high vulnerability to reward hacking [cite: 5, 13, 16] |
| LoRA (Rank = 64, Alpha = 128) | ~1.20% - 2.10% | Moderate (38.8%) [cite: 13] | Moderate structural compliance; limited tactical depth [cite: 13] | Partial domain adaptation; residual bottlenecks in positional representations [cite: 13] |
| Full Parameter Fine-Tuning | 100.0% | High (40.9% - 48.1%) [cite: 2, 13] | Exhaustive; accurate line calculation and structured explanations [cite: 2] | Unrestricted parameter remapping; robust internal representation of niche rules [cite: 10, 13] |

At the 2B scale, capturing non-trivial chess representations (such as FEN strings, board geometry, principal variations, and threat matrices) requires fundamental updates to internal attention layers and feed-forward projections [cite: 10, 13, 17]. Low-rank adaptors constrain capacity, forcing the model to reuse generic language paths [cite: 5, 13]. This leads to early performance plateaus, surface-level format compliance, and hallucinated engine jargon [cite: 5, 13].

## Comparative Analysis of Post-Training Paradigms for Small Language Models

Selecting the appropriate post-training paradigm requires balancing memory efficiency, exploratory stability, offline control, and task-granularity requirements [cite: 15, 18, 19].

| Post-Training Paradigm | Core Optimization Objective & Loss Aggregation | Memory Footprint & Infrastructure | Exploratory Mechanics | Dominant Failure Modes & Instabilities | Target Granularity & Ideal Task Scale |
| --- | --- | --- | --- | --- | --- |
| Supervised Fine-Tuning (SFT) | Token-level cross-entropy loss over single deterministic target sequence [cite: 5, 20]. | Minimal: Policy model parameters + optimizer states ($1\times$ baseline) [cite: 14, 18]. | Zero exploration; purely imitative trajectory fitting [cite: 5, 11]. | Catastrophic forgetting, hallucination of unseen states, exposure bias [cite: 5, 11]. | Initial format, tone, and foundational domain knowledge acquisition [cite: 5, 20]. |
| Direct Preference Optimization (DPO) | Pairwise implicit reward optimization via binary cross-entropy on log-ratio pairs [cite: 15, 20]. | Low: Policy model + Reference model ($2\times$ inference footprint) [cite: 15, 20]. | Offline static exploration bounded by preference dataset distribution [cite: 15, 21]. | Overfitting to static dataset pairs, weak out-of-domain (OOD) generalization [cite: 21]. | General stylistic alignment, instruction following, in-domain pairwise ranking [cite: 15, 21]. |
| Proximal Policy Optimization (PPO) | Token-level policy gradient clipped surrogate using a learned value function critic [cite: 6, 12]. | High: Policy + Reference + Critic + Reward model ($3.5\times - 4\times$ parameters) [cite: 6, 12]. | Dynamic online trajectory sampling with GAE credit assignment [cite: 6, 12]. | Critic instability, high computational cost, training collapse on small models [cite: 12]. | Multi-step reasoning with dense token-level feedback under large compute budgets [cite: 6, 19]. |
| Standard GRPO | Sequence-level relative advantage ($1/o_i$ token normalized) against group mean [cite: 6, 8, 12]. | Moderate: Policy + Reference model (No Critic needed; $2\times$ parameters) [cite: 6, 12, 18]. | Online dynamic sampling over $G$ group completions [cite: 6, 8, 12]. | | |
| Dr. GRPO | Group advantage without $1/\sigma$ division and sequence-length normalization [cite: 7, 9, 22]. | Moderate: Policy + Reference model [cite: 9, 22]. | Online group sampling; unweighted raw accuracy gradient scaling [cite: 9, 22]. | Higher gradient variance across varied batch sequence lengths [cite: 22]. | Multi-step logic requiring unbiased, unnormalized outcome optimization [cite: 7, 9, 22]. |
| DAPO (Decoupled Clip & Dynamic Sampling) | Token-level mean loss aggregation over group; decoupled clip range; dynamic $\sigma>0$ sampling [cite: 6, 9, 23]. | Moderate: Policy + Reference model [cite: 9, 23]. | Online group sampling with explicit zero-variance group filtering [cite: 9, 23]. | Potential over-expansion of sequence length if overlong penalties are misconfigured [cite: 23]. | Deep multi-step reasoning, mathematical proof, code generation, long CoT [cite: 6, 23]. |
| Supervised GRPO (S-GRPO) | On-policy GRPO + Conditional Ground-Truth Trajectory Injection (CGI) [cite: 5, 11]. | Moderate: Policy + Reference + Oracle trajectory memory [cite: 5, 11]. | Hybrid dynamic trajectory sampling anchored by expert trajectories [cite: 5, 11]. | Risk of regression to pure SFT behavior if injection threshold is set too high [cite: 5, 11]. | Cold-start domain adaptation, niche technical reasoning, small parameter models [cite: 5, 11]. |
| GR3 (Group Relative Reward Rescaling) | Multiplicative reward rescaling ($R_{\text{shaped}} = R \cdot \frac{1}{1 + \alpha \cdot \ell}$) over relative advantage [cite: 4, 24]. | Moderate: Policy + Reference model [cite: 4, 24]. | Online length-aware preference optimization [cite: 4, 24]. | Sensitive to $\alpha$ scaling hyperparameter choice under continuous non-binary rewards [cite: 4, 24, 25]. | Efficiency-aware alignment, length-budgeted instructional generation [cite: 4, 24]. |

### Theoretical Alignment and Optimization Trade-Offs

Direct Preference Optimization (DPO) simplifies post-training by reparameterizing the reward function directly in terms of the policy, eliminating the need for an explicit online reward model or real-time rollout generation [cite: 15, 20]. While DPO achieves high stability and low computational overhead, its reliance on static, offline preference pairs severely restricts exploration [cite: 15, 21]. In complex domains like chess calculation, DPO overfits to the static preference dataset, exhibiting poor out-of-domain generalization when encountering unfamiliar board setups [cite: 21]. Online reinforcement learning paradigms like GRPO and DAPO allow the policy to actively explore trajectory space, discovering novel reasoning paths and self-correction mechanisms that static offline datasets cannot provide [cite: 6, 12, 26].

The fundamental advantage of Group Relative Policy Optimization (GRPO) over traditional Proximal Policy Optimization (PPO) is the complete elimination of the value function network (critic) [cite: 6, 12]. In PPO, training a secondary critic network of comparable size to the 2B policy model doubles memory consumption and introduces severe instability if the critic fails to accurately estimate token-level value functions in specialized domains [cite: 6, 12]. GRPO circumvents this bottleneck by sampling a group of completions ($G \ge 4$) per prompt and utilizing the empirical group mean as a dynamic baseline [cite: 6, 12, 14, 18].

However, standard GRPO introduces structural vulnerabilities through its sequence-level loss aggregation [cite: 6, 23, 27]. Dividing token losses by total sequence length $|o_i|$ distorts gradient updates, causing long, detailed explanations to receive diminished per-token gradient norms compared to short completions [cite: 6, 7, 22]. Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) resolves this distortion by shifting to token-level loss aggregation across the entire group [cite: 6, 23, 27]. By decoupling the clipping range and normalizing loss at the token level, DAPO prevents long reasoning paths from being penalized, enabling small models to sustain rich, multi-step instructional chains without suffering gradient dilution [cite: 6, 23, 27].

Incorporating explicit reasoning tokens (such as `<think> ... </think>`) allows the model to separate internal calculation mechanics from surface-level instructional delivery [cite: 28, 29]. For a 2B parameter model, forcing move calculation and tactical verification into explicit reasoning traces prevents conversational text from becoming cluttered with raw calculation strings [cite: 17, 28]. However, without proper reward shaping, reasoning token paradigms are vulnerable to blank-token reward hacking, where the model outputs empty or invisible characters inside reasoning tags to satisfy structural constraints while bypassing actual calculation [cite: 28].

## Methodological Frameworks for High-Fidelity Domain-Specific Teaching

To eliminate chess hallucinations and engine jargon while preserving instructional depth in small language models, post-training pipelines must combine structured teacher distillation with stage-specific data allocation [cite: 1, 2, 10, 13].

### Master Distillation and Feigned Discovery Prompting

Small language models cannot reliably discover complex spatial logic or multi-move tactical combinations through unguided exploration [cite: 2, 5, 10]. The Master Distillation framework bridges this capability gap by combining the computational precision of specialized engines with the natural language fluency of frontier teacher models [cite: 2, 10, 13].

The process begins by extracting objective evaluation artifacts from an advanced engine, such as Stockfish operating at a minimum depth of 24 [cite: 10, 13]. For every target position, the engine provides the exact Principal Variation (PV), centipawn evaluations or mate counts, and key positional features [cite: 10, 30]. Because raw engine outputs are dense and uninstructive for novice players, a frontier teacher model (such as Gemini 3 Flash or GPT-4o) processes these artifacts using Feigned Discovery Prompting [cite: 2, 10, 13].

Under Feigned Discovery Prompting, the teacher LLM receives the engine's ground-truth hints but is explicitly instructed to compose a pedagogical explanation that mimics natural, step-by-step human discovery [cite: 2, 10, 13]. The teacher model explains why intuitive candidate moves fail, highlights key tactical patterns, and articulates the strategic rationale behind the optimal line [cite: 2, 10, 13].

Crucially, all engine hints, explicit PV strings, and evaluation scores are stripped from the student model's training prompt [cite: 10, 13]. The 2B student model sees only the standard board representation (FEN string), piece placement mappings, legal move lists, and target persona constraints [cite: 10, 13, 17].

### Stage-Specific Data Allocation and Bridge Mechanisms

Applying identical dataset distributions across Supervised Fine-Tuning and Reinforcement Learning stages degrades optimization efficiency in small models [cite: 1, 31, 32]. Training data must be partitioned based on task complexity and current policy capability [cite: 1, 31].

The SFT stage should focus strictly on skill acquisition, exposing the model to complex positions that it has not yet mastered [cite: 1, 31, 32]. When raw teacher traces are structurally dense, a Bridge Mechanism re-formats teacher completions into simplified, learnable chains of thought tailored to the 2B scale [cite: 1, 31, 32]. The Bridge mechanism decomposes complex calculations into four standardized sections: positional threat assessment, tactical pattern recognition, candidate variation evaluation (limited to two or three moves for club players), and final move recommendation [cite: 1, 17].

The RL stage focuses on skill consolidation, drawing from positions where the policy can already achieve non-zero success rates [cite: 1, 31, 32]. Reinforcement learning reinforces valid logical steps, corrects formatting errors, and refines move accuracy [cite: 1, 10, 19]. When the policy generates all-zero reward rollouts ($R=0$ across all group completions), Critique Fine-Tuning captures these failed instances [cite: 1, 31, 32]. A diagnostic generator converts these failures into explicit error analyses and repair traces, feeding them back into an intermediate SFT update before resuming RL training [cite: 1, 31, 32].

### Supervised Group Relative Policy Optimization (S-GRPO)

To address the cold-start barrier during RL fine-tuning without relying on complex curriculum scheduling, Supervised Group Relative Policy Optimization (S-GRPO) incorporates Conditional Ground-Truth Trajectory Injection (CGI) [cite: 5, 11].

During on-policy rollout sampling, if a binary verifier determines that all $G$ completions generated by the model fail to select the correct move or violate domain rules (yielding $R_i = 0$ for all candidates), S-GRPO dynamically injects a verified ground-truth teacher trace ($o^*$) into the candidate group [cite: 5, 11]. Assigning a deterministic maximal reward ($R^* = 1.0$) to this injected anchor restores dynamic contrast to the group statistics ($\mu > 0, \sigma > 0$) [cite: 5, 11].

This injection ensures a positive relative advantage signal ($\hat{A}_{\text{anchor}} > 0$) for the expert trajectory while assigning negative relative advantages ($\hat{A}_{\text{failed}} < 0$) to the failed rollouts [cite: 5, 11]. S-GRPO reformulates supervised imitation learning as a high-advantage component of the policy gradient, maintaining optimization progress even when zero-shot exploratory success is minimal [cite: 5, 11].

## Advanced Reward Engineering and Policy Stability Mechanics

To eliminate brevity collapse, prevent proxy reward hacking, and maintain instructional depth, the reward architecture must decouple task accuracy from sequence length while regularizing evaluator signals [cite: 3, 4, 16, 24].

### Multiplicative Reward Rescaling vs. Additive Penalties

Traditional length regularization relies on additive penalty terms subtracted directly from the task reward:

$$
\hat{R}_{(+)} = R_{\text{task}} - \lambda \cdot S(\ell)
$$

Additive formulations introduce an optimization shortcut: the length term $-\lambda \cdot S(\ell)$ functions as an independent objective that the policy can exploit [cite: 4, 24]. The model quickly learns that truncating outputs yields substantial length bonuses that outweigh minor drops in task accuracy ($R_{\text{task}}$), leading to brevity collapse [cite: 4, 24].

Group Relative Reward Rescaling (GR3) eliminates this shortcut by replacing additive terms with a multiplicative rescaling factor [cite: 4, 24]:

$$
\hat{R}_{(\times)} = R_{\text{task}} \cdot \left( \frac{1}{1 + \alpha \cdot \ell_i} \right)
$$

where $\ell_i = |o_i|$ represents completion token length and $\alpha$ is a scaling hyperparameter [cite: 4, 24]. Multiplicative rescaling directly couples length control to task success [cite: 4, 24]. Taking the derivative of the rescaled reward with respect to length shaping reveals that length regularization scales proportionally with task accuracy:

$$
\frac{\partial \hat{R}_{(\times)}}{\partial S} = R_{\text{task}}
$$

If a generated completion is tactically incorrect ($R_{\text{task}} = 0$), the length scaling term evaluates to zero, providing no brevity reward for short, incorrect answers [cite: 4, 24]. Length bonuses apply exclusively when task accuracy is satisfied, forcing the model to optimize factual correctness and explanation depth simultaneously [cite: 3, 4, 24]. Similarly, the Adaptive Correct-Only Efficiency Reward (Acoer) isolates brevity bonuses to verified correct completions, preventing structural penalty loops from triggering policy collapse [cite: 3].

### Mitigating Proxy Judge Reward Hacking: Shared Rubric Dropout

When training models against automated rubrics evaluated by an LLM judge, small models frequently learn to exploit stylistic patterns in the evaluation prompt [cite: 16, 33]. Over training iterations, proxy judge scores continue to rise while gold-standard expert evaluation peaks and degrades—demonstrating proxy reward hacking under Goodhart's Law [cite: 16, 33].

To regularize proxy evaluations, Rubric Dropout randomly masks out individual rubric criteria during each training step [cite: 16]. Crucially, the dropout mask must be shared across all candidate completions within a rollout group $G$ [cite: 16]. Mask sharing preserves advantage standardization math across the group, preventing artificial reward variance [cite: 16]. Applying a 30% to 50% rubric dropout rate prevents the model from gaming fixed criteria, encouraging generalized instructional capability [cite: 16].

### Multi-Objective Advantage Synchronization (MO-GRPO)

Standard GRPO pools all reward signals into a single scalar variance calculation [cite: 8, 25]. Consequently, reward components with high statistical variance (such as binary tactical move accuracy) dominate policy updates, suppressing low-variance components (such as pedagogical tone or formatting consistency) [cite: 8, 25].

Multi-Objective GRPO (MO-GRPO) computes independent advantage estimations for each distinct reward component prior to aggregation [cite: 8]:

$$
\hat{A}_{i, m} = \frac{R_{i, m} - \mu_m}{\sigma_m}
$$

$$
\hat{A}_i^{\text{MO}} = \sum_{m=1}^{M} w_m \cdot \hat{A}_{i, m}
$$

where $R_{i, m}$ represents candidate evaluation on objective $m$, and $w_m$ represents explicit objective weights [cite: 8]. Independent normalization ensures that every reward component contributes proportionally to policy updates regardless of raw variance scale, preventing high-variance signals from overwhelming qualitative instructional metrics [cite: 8].

## Strategic Implementation Roadmap for Qwen 3.5 2B Chess Instruction

To elevate the Qwen 3.5 2B model from its performance plateau (2.78 / 5.0 judge score, 1.87 usefulness, 30-word output average) to high factual accuracy and pedagogical depth, post-training should proceed through four integrated execution phases [cite: 3, 4, 5, 10, 13].

### Phase 1: Master Distillation Dataset Curation

- Construct a core dataset of 15,000 to 20,000 tactical puzzle positions and annotated club-level game positions covering diverse tactical themes (pins, forks, skewers, mating nets, piece defense) [cite: 2, 10].
- Run Stockfish at depth 24 or higher to extract ground-truth Principal Variations, centipawn evaluations, and key positional features [cite: 10].
- Query a frontier LLM teacher (such as Gemini 3 Flash) using Feigned Discovery Prompting to generate natural language explanations that articulate why candidate lines fail and how tactical goals are achieved [cite: 2, 10, 13].
- Remove all engine notation, explicit PV strings, and numerical evaluations from student-facing prompts [cite: 10, 13]. Format prompts to supply raw FEN strings, parsed piece placement maps, and legal move lists [cite: 17].

### Phase 2: Full Parameter Supervised Fine-Tuning

- Replace parameter-efficient LoRA adapters with Full Parameter Fine-Tuning across all model projections, enabling unrestricted parameter remapping for spatial and move-calculation logic [cite: 10, 13].
- Apply the Bridge Mechanism to format target completions into four explicit instructional sections: Position Assessment, Tactical Calculation, Alternative Refutation, and Final Move Conclusion (`FINAL_ANSWER: uci_move`) [cite: 1, 17, 32].
- Fine-tune Qwen 3.5 2B for 3 epochs using a cosine learning rate scheduler ($lr = 2 \times 10^{-5}$) and a maximum sequence length of 1,024 tokens [cite: 15]. Confirm that baseline tactical accuracy reaches at least 40% before advancing to RL fine-tuning [cite: 10, 17].

### Phase 3: On-Policy Reinforcement Learning (S-GRPO + DAPO + GR3)

- **Loss Aggregation:** Implement DAPO token-level loss aggregation across rollout groups to eliminate sequence-length normalization bias ($\frac{1}{|o_i|}$) [cite: 6, 23, 27].
- **Cold-Start Anchoring:** Implement S-GRPO Conditional Ground-Truth Trajectory Injection [cite: 5, 11]. If all sampled completions within a group ($G=4$ or $G=8$) fail to select the correct move, inject the verified master trace to ensure a non-zero, positive advantage baseline [cite: 5, 11].
- **Multiplicative Length Control:** Replace additive length penalties with GR3 Multiplicative Reward Rescaling [cite: 4, 24]:

  $$
  \hat{R} = R_{\text{task}} \cdot \left( \frac{1}{1 + \alpha \cdot \max(0, \ell_{\text{target}} - \ell_i)} \right)
  $$

  Set $\ell_{\text{target}} = 150 \text{ tokens}$ to explicitly encourage instructional depth and prevent outputs from collapsing into brief summaries [cite: 4, 17, 24].

### Phase 4: Regularized Multi-Objective Evaluation

- **Multi-Objective Reward Function:** Establish distinct reward streams for Stockfish-verified move correctness ($R_{\text{move}} \in \{0, 1\}$), exact format compliance ($R_{\text{fmt}} \in \{0, 0.5, 1\}$), and instructional clarity ($R_{\text{pedagogical}}$ from an automated judge) [cite: 10, 14, 15, 17].
- **Proxy Judge Regularization:** Apply 30% Shared Rubric Dropout to the automated judge during training iterations to mitigate proxy reward hacking [cite: 16].
- **Advantage Synchronization:** Use MO-GRPO to calculate independent advantage norms for each reward component, ensuring balanced policy updates across tactical accuracy, formatting, and pedagogical prose [cite: 8].

## Strategic Summary of Technical Recommendations

| Engineering Challenge | Observed Failure Mode | Recommended Architecture / Method | Operational Mechanism & Target Outcome |
| --- | --- | --- | --- |
| Brevity Collapse & Output Compression | Average completion drops to 30 words; low usefulness score (1.87) [cite: 3, 6]. | DAPO Token-Level Loss + GR3 Rescaling [cite: 4, 6, 23, 24] | Replaces sequence-level length division with token-mean aggregation; couples length scaling multiplicatively to task success to sustain ~150-token explanations [cite: 4, 6, 23, 24]. |
| Domain Representation Bottleneck | Early accuracy plateau; high rule failure and hallucinated jargon [cite: 5, 13]. | Full Parameter Fine-Tuning [cite: 10, 13] | Replaces LoRA adaptors to allow full weight remapping, enabling internal encoding of spatial chess rules and board geometry [cite: 10, 13]. |
| Sparse Reward Cold-Start Failure | Vanishing gradients ($\sigma=0$); silent sample groups [cite: 5, 9]. | S-GRPO (Conditional Trajectory Injection) [cite: 5, 11] | Inject ground-truth master trajectories when rollouts fail, maintaining positive advantage contrast ($\hat{A} > 0$) [cite: 5, 11]. |
| Proxy Judge Over-Optimization | High format score alongside low utility; score saturation [cite: 16, 33]. | Shared Rubric Dropout (30-50%) [cite: 16] | Randomly masks judge rubric criteria across rollout groups to prevent stylistic reward hacking [cite: 16]. |
| Uninstructive / Jargon-Heavy Prose | Dry calculation strings; poor beginner comprehension [cite: 2, 30]. | Master Distillation via Feigned Discovery [cite: 2, 13] | Combines Stockfish engine calculation with frontier LLM prompting to synthesize natural, discovery-based teaching traces [cite: 2, 10, 13]. |

## References

1. Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning — arXiv. <https://arxiv.org/html/2606.04466v1>
2. Grounded Chess Reasoning in Language Models via Master Distillation — arXiv. <https://arxiv.org/html/2603.20510v2>
3. Beyond Penalizing Mistakes: Stabilizing Efficiency Training in Large Reasoning Models via Adaptive Correct-Only Rewards — arXiv. <https://arxiv.org/html/2606.22716v1>
4. Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning — arXiv. <https://arxiv.org/html/2603.10535v1>
5. S-GRPO: Unified Post-Training for Large Vision-Language Models — arXiv. <https://arxiv.org/html/2604.16557v1>
6. Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement — arXiv. <https://arxiv.org/html/2512.07611v1>
7. Rethinking the Comparison Unit in Sequence-Level Reinforcement Learning: An Equal-Length Paired Training Framework from Loss Cor — arXiv. <https://arxiv.org/pdf/2604.17328>
8. MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems — arXiv. <https://arxiv.org/html/2509.22047v3>
9. GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity — arXiv. <https://arxiv.org/pdf/2607.00152>
10. Grounded Chess Reasoning in Language Models via Master Distillation — arXiv. <https://arxiv.org/pdf/2603.20510>
11. S-GRPO: Unified Post-Training for Large Vision-Language Models — arXiv. <https://arxiv.org/html/2604.16557v2>
12. Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO for LLM Reasoning Enhancement — arXiv. <https://arxiv.org/pdf/2512.07611>
13. Grounded Chess Reasoning in Language Models via Master Distillation — arXiv. <https://arxiv.org/html/2603.20510v1>
14. Multi-Reward GRPO Fine-Tuning for De-biasing Large Language Models: A Study Based on Chinese-Context Discrimination Data — arXiv. <https://arxiv.org/html/2511.06023>
15. Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs — arXiv. <https://arxiv.org/html/2512.22631v1>
16. A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL (Aug 2026) — YouTube. <https://www.youtube.com/watch?v=_Jv5CthuzuQ>
17. UofTCSSLab/C1-4B — Hugging Face. <https://huggingface.co/UofTCSSLab/C1-4B>
18. GRPO Training Dynamics for Small Language Models — arXiv. <https://arxiv.org/html/2609.39321v1>
19. A Technical Survey of Reinforcement Learning Techniques for Large Language Models. <https://arxiv.org/html/2507.04136v2>
20. Post-Training LLMs Guide: SFT, RLHF, DPO & GRPO Explained (2026) — Sundeep Teki. <https://www.sundeepteki.org/advice/the-complete-guide-to-post-training-llms-how-sft-rlhf-dpo-and-grpo-shape-llms>
21. Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO — OpenReview. <https://openreview.net/forum?id=tMf3keuPOl>
22. Δ⁢𝐿 Normalization: Rethink Loss Aggregation in RLVR — arXiv. <https://arxiv.org/html/2509.07558v1>
23. DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization): GRPO on steroids | by Syed Hasan | Medium. <https://medium.com/@syed_hasan/dapo-decoupled-clip-and-dynamic-sampling-policy-optimization-grpo-on-steroids-9c571a0536f3>
24. Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning — arXiv. <https://arxiv.org/pdf/2603.10535>
25. Process Advantage Signal Shaping: A Paradigm-Agnostic Middleware for Process-Supervised RL in LLM Reasoners — arXiv. <https://arxiv.org/pdf/2606.29296>
26. First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training — arXiv. <https://arxiv.org/html/2505.22453v2>
27. From Uniform to Heterogeneous: Tailoring Policy Optimization to Every Token's Nature. <https://arxiv.org/html/2509.16591v1>
28. GRPO on SFT model: reward hacking with blank / gibberish reasoning — how are you designing robust rewards? : r/LocalLLaMA — Reddit. <https://www.reddit.com/r/LocalLLaMA/comments/1q8wp6f/grpo_on_sft_model_reward_hacking_with_blank/>
29. Done Is Better than Perfect: Unlocking Efficient Reasoning by Structured Multi-Turn Decomposition — arXiv. <https://arxiv.org/html/2505.19788v1>
30. GitHub - Thytu/StockLLM: Elevating Chess Strategy with Fine-Tuned Large Language Model. <https://github.com/Thytu/StockLLM>
31. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. <https://tldr.takara.ai/p/2501.17161>
32. \[2606.04466\] Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning — arXiv. <https://arxiv.org/abs/2606.04466>
33. Reward Shaping to Mitigate Reward Hacking in RLHF — arXiv. <https://arxiv.org/html/2502.18770v3>
