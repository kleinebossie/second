# Systemic Mitigation of Length Collapse and Reward Hacking in Verifier-Guided Policy Optimization

> [!NOTE]
> Source: Gemini Deep Research report, added 2026-11-06. This is a formatted copy. The content is not changed. A critical review of this report is in [analysis.md](analysis.md).

Reinforcement learning (RL) post-training for large language models frequently encounters optimization pathologies when deterministic rule verifiers are combined with soft scalar preference rewards [cite: 1, 2, 3]. A primary failure mode is length collapse, wherein the policy minimizes negative penalties by producing degenerate, ultra-short completions [cite: 1, 2]. When evaluated against unaligned base models that generate long, articulate essays containing factual errors, automated and human evaluators frequently rank the models equally due to pervasive verbosity and style biases inherent in evaluation frameworks [cite: 4, 5].

Resolving these challenges requires analyzing the mathematical mechanics of length collapse, evaluating algorithmic strategies to maintain policy entropy and trajectory length, formulating multi-objective optimization via Constrained Markov Decision Processes (CMDPs), and establishing a comparative taxonomy of modern reward modeling paradigms.

## Mechanistic Analysis of Length Collapse and Evaluation Asymmetries

### Autoregressive Trajectory Risk Accumulation

In autoregressive language generation, a policy $\pi_\theta$ generates a sequence of tokens $o = (y_1, y_2, \dots, y_T)$ for a given prompt $x$ according to the joint probability distribution:

$$\pi_\theta(o \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid x, y_{<t})$$

When an environment incorporates a strict rule verifier—such as a chess engine evaluating move legality or a database validator verifying coordinate names—any token step $t$ that transitions the environment into an illegal state triggers a severe terminal penalty $R_{\text{rule}} = -1.0$ [cite: 2, 6].

Because autoregressive decoding compounds token probabilities over time, the likelihood of a stochastic policy generating an error-free trajectory decays as sequence length $T$ increases [cite: 7]. Under a policy with non-zero entropy, longer completions face an exponentially higher cumulative risk of introducing a rule failure [cite: 7].

Autoregressive models favor brevity under negative penalization because multiplying token probabilities penalizes longer paths [cite: 7]. Consequently, the optimization algorithm quickly identifies a shortcut: truncating generation at the earliest possible step minimizes the trajectory space over which a rule violation can occur, driving output length toward zero [cite: 1, 7].

### Ungradated Advantage Estimation in Group Optimization

In Group Relative Policy Optimization (GRPO), policy updates are guided by normalizing scalar rewards across a group of $G$ rollouts $\{o_1, o_2, \dots, o_G\}$ sampled from the current policy $\pi_{\theta_{\text{old}}}$ for a prompt $x$ [cite: 1, 8, 9]:

$$\hat{A}_i = \frac{R(o_i) - \text{mean}(\{R_j\})}{\text{std}(\{R_j\}) + \epsilon}$$

The objective optimized by GRPO is defined as:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)} \hat{A}_i, \, \text{clip}\left(\frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) \right] + \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

When negative rollouts carry ungradated penalties—treating minor formatting errors identically to fatal domain rule failures—group relative advantage normalization assigns strong negative advantage values to longer rollouts that trigger a rule violation late in their generation sequence [cite: 1, 2].

If soft positive rewards for target length or key points are additive and small (e.g., $+0.1$) relative to the hard negative penalty ($-1.0$), the policy gradient vectors derived from the rule verifier dominate the update step [cite: 2].

The policy gradient forcefully depresses log-probabilities across long trajectories, causing a sharp reduction in generated token length and a collapse of policy entropy [cite: 1, 7, 10]. The model collapses into evasive, generic outputs like "The move is valid" or minimal sentences that guarantee zero penalty exposure [cite: 1, 10].

### The Evaluation Paradox and Bradley-Terry Length Bias

The paradox where an untuned base model (generating long essays with factual errors) ties a length-collapsed RL model (which is factually strict but brief) stems from structural evaluative bias in human and automated judges [cite: 4, 5]. Evaluator preferences are typically modeled using a Bradley-Terry logistic regression formulation [cite: 5, 11, 12]:

$$P(y_1 \succ y_2 \mid x) = \sigma \left( \beta_0 + \beta_{\text{quality}} \Delta Q(y_1, y_2) + \beta_{\text{length}} \Delta \text{Len}(y_1, y_2) \right)$$

where $\Delta Q(y_1, y_2)$ represents the true differential in factual accuracy and semantic correctness, and $\Delta \text{Len}(y_1, y_2) = \text{Length}(y_1) - \text{Length}(y_2)$ denotes the token length difference [cite: 5, 12].

In automated judge models (such as LLM-as-a-Judge) and human evaluators, the length coefficient $\beta_{\text{length}}$ is consistently positive [cite: 4, 5]. An untuned base model producing an elaborate, multi-paragraph explanation gains a significant score boost from the length component $\beta_{\text{length}} \Delta \text{Len}$ [cite: 4, 5].

This positive length bias offsets the negative score penalty incurred in $\Delta Q$ due to factual errors [cite: 4, 5]. Conversely, the RL-tuned model produces factually accurate but truncated output, incurring a heavy penalty via $\beta_{\text{length}} \Delta \text{Len}$ [cite: 4, 5].

The evaluator interprets the base model's thorough prose as informative and the RL model's brief response as unhelpful, resulting in equal preference win-rates despite the base model's underlying domain errors [cite: 4, 5].

## Algorithmic Mechanisms to Prevent Policy Collapse

### Length Normalization and Trajectory-Level Regularization

Decoupling autoregressive likelihood accumulation from sequence length requires incorporating explicit length normalization directly into reward metrics and policy estimates [cite: 7, 13, 14]. Simple Preference Optimization (SimPO) addresses length bias by replacing unnormalized sequence log-probabilities with length-normalized margin targets [cite: 13, 14]:

$$r_{\text{norm}}(x, o) = \frac{\beta}{|o|} \sum_{t=1}^{|o|} \log \pi_\theta(y_t \mid x, y_{<t})$$

By taking the average token log-probability rather than the cumulative sequence sum, the mathematical bias favoring shorter completions is eliminated [cite: 7, 13, 14]. In policy optimization frameworks employing rule verifiers, rewards can be regularized via geometric means or trajectory-length scaling terms to prevent short paths from achieving artificially high expected advantages [cite: 7].

### Distribution-Matching Policy Optimization and Trajectory Balance

Standard on-policy RL algorithms minimize the reverse Kullback-Leibler (KL) divergence between the policy and the target reward distribution [cite: 9]:

$$\min_\theta D_{\text{KL}}(\pi_\theta \parallel p^*) \quad \text{where} \quad p^*(o \mid x) \propto \exp\left(\frac{R(x, o)}{\alpha}\right)$$

Reverse KL divergence is inherently mode-seeking; when applied to multimodal or heavily constrained reward landscapes, it collapses the policy onto a single narrow trajectory mode that minimizes risk [cite: 9].

Distribution-Matching Policy Optimization (DMPO) addresses this by incorporating a forward KL divergence term:

$$\min_\theta D_{\text{KL}}(p^* \parallel \pi_\theta)$$

Because forward KL divergence is mode-covering, it forces the policy to maintain probability mass across all valid trajectory modes, preserving reasoning paths and output diversity [cite: 9].

FlowBalance leverages trajectory balance optimization to fit a normalized energy distribution over full completions [cite: 15]:

$$p_{\text{FlowBalance}}(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp\left( \frac{E_{\text{FlowBalance}}(y \mid x)}{\tau} \right)$$

By conditioning self-guidance scores on verifier outcomes and utilizing privileged hindsight context, FlowBalance maintains complete response distributions and avoids the length collapse observed in direct on-policy self-distillation [cite: 15, 16].

### Dynamic Conciseness Rewards and Difficulty-Adaptive Length Scaling

Rather than applying static length bonuses—which are easily gamed by repetitive generation—reward modeling should employ dynamic conciseness and difficulty-adaptive length scaling [cite: 1, 17].

A Conciseness Reward Model (CRM) evaluates the logical density of a reasoning path relative to prompt difficulty [cite: 1]. The reward function couples outcome correctness $R_{\text{outcome}} \in \{0, 1\}$ directly with a non-linear conciseness score $S_{\text{concise}}$ [cite: 1]:

$$R_{\text{composite}}(x, o) = R_{\text{outcome}}(x, o) \cdot \left( 1 + \gamma(x) \cdot S_{\text{concise}}(x, o) \right)$$

where $\gamma(x)$ is an adaptive coefficient tuned to the estimated difficulty of query $x$ [cite: 1]. For complex prompts requiring multi-step reasoning, $\gamma(x)$ relaxes length penalties, permitting longer generations [cite: 1].

For simple queries, $\gamma(x)$ increases pressure against unnecessary verbosity [cite: 1]. Crucially, $S_{\text{concise}}$ is gated by $R_{\text{outcome}}$; an incorrect or invalid response receives zero positive scaling from its conciseness score, preventing the model from exploiting brevity shortcuts on failed rollouts [cite: 1].

### Latent Variance Tracking and Confidence-Conditioned Interventions

Length collapse is preceded by a sharp decline in the standard deviation of response length within rollout groups ($\sigma_L$) [cite: 18, 19]. Tracking $\sigma_L$ across training steps serves as an early indicator of exploration failure [cite: 18, 19].

Confidence-conditioned framework designs (such as TTRL-CoCoV) adjust exploration incentives across distinct verifier confidence regimes [cite: 18, 19]:

- **Low-Confidence Regime:** The verifier filters out noisy pseudo-labels, preventing incorrect negative penalties from corrupting policy gradients [cite: 18, 19].
- **High-Confidence Regime:** Majority voting is reliable, but policy diversity tends to decay; the system injects length-variance exploration rewards exclusively into these rollouts to sustain trajectory exploration without introducing pseudo-label noise [cite: 18, 19].

## Multi-Objective Optimization via Constrained Markov Decision Processes

Combining hard factual rules with soft qualitative preferences using linear scalarization ($R = w_1 R_{\text{rule}} + w_2 R_{\text{quality}}$) induces severe gradient interference [cite: 20, 21]. If $w_1 \gg w_2$, length collapses [cite: 1, 2]. If $w_2 \gg w_1$, the model hacks the soft reward by generating verbose, pleasant prose that violates domain constraints [cite: 3, 4, 6].

This optimization challenge is properly formulated as a Constrained Markov Decision Process (CMDP) [cite: 20, 21, 22].

### CMDP Formalism for Post-Training

Let the policy $\pi_\theta$ aim to maximize soft subjective prose quality, tone, and key-point coverage, subject to strict upper bounds on expected factual rule violation costs [cite: 20, 21, 22]:

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, o \sim \pi_\theta} \left[ R_{\text{quality}}(x, o) \right] \quad \text{subject to} \quad \mathbb{E}_{x \sim \mathcal{D}, o \sim \pi_\theta} \left[ C_k(x, o) \right] \le l_k, \quad \forall k \in \{1, \dots, K\}$$

where $R_{\text{quality}}(x, o) \in \mathbb{R}$ is a scalar score for tone, structure, and depth, while $C_k(x, o) \in \{0, 1\}$ is an indicator cost function returning $1$ if factual rule $k$ is violated, and $0$ otherwise [cite: 20, 21]. The parameter $l_k \ge 0$ represents the maximum allowable constraint violation threshold (set near zero for strict rule adherence) [cite: 20, 21].

### Primal-Dual Lagrangian Relaxation and Adaptive Multiplier Kinetics

The constrained problem is converted into an unconstrained saddle-point optimization problem using Lagrangian relaxation [cite: 20, 21, 23, 24]:

$$\max_{\theta} \min_{\boldsymbol{\lambda} \ge 0} \mathcal{L}(\theta, \boldsymbol{\lambda}) = \mathbb{E}_{\pi_\theta} \left[ R_{\text{quality}}(x, o) \right] - \sum_{k=1}^K \lambda_k \left( \mathbb{E}_{\pi_\theta} \left[ C_k(x, o) \right] - l_k \right)$$

where $\boldsymbol{\lambda} = (\lambda_1, \dots, \lambda_K)^T$ is a vector of non-negative Lagrange multipliers (dual variables) [cite: 20, 21, 23]. Optimization proceeds via two-scale primal-dual iterations [cite: 20, 21, 23, 24, 25, 26]:

1. **Primal Policy Update (Policy Gradient Ascent)**

   $$\theta^{(t+1)} = \theta^{(t)} + \eta_\theta \nabla_\theta \mathbb{E}_{x, o \sim \pi_\theta} \left[ R_{\text{quality}}(x, o) - \sum_{k=1}^K \lambda_k^{(t)} C_k(x, o) \right]$$

2. **Dual Variable Update (Projected Gradient Descent)**

   $$\lambda_k^{(t+1)} = \max\left( 0, \, \lambda_k^{(t)} + \eta_\lambda \left( \bar{C}_k^{(t)} - l_k \right) \right)$$

where $\bar{C}_k^{(t)}$ is the empirical violation rate of constraint $k$ in iteration batch $t$, and $\eta_\lambda$ is the dual learning rate [cite: 21, 23, 24, 27].

This dynamic multiplier mechanism prevents length collapse by dynamically adjusting penalty weights:

- When the policy satisfies domain constraints ($\bar{C}_k^{(t)} \le l_k$), the multiplier $\lambda_k$ decays toward zero [cite: 21, 23, 27]. The surrogate reward is dominated by $R_{\text{quality}}$, encouraging the model to expand sequence length, improve tone, and elaborate on key points [cite: 20, 21].
- If the policy begins making factual errors or illegal moves ($\bar{C}_k^{(t)} > l_k$), $\lambda_k$ increases automatically [cite: 21, 23, 27]. This scales up the penalty term, pulling the policy back into the feasible constraint space [cite: 20, 21, 27].

### Augmented Lagrangian and Regularized Primal-Dual Policy Search

Basic primal-dual updates can suffer from instability and multiplier oscillations caused by noisy empirical cost estimations in group rollouts [cite: 21, 26]. To stabilize dual convergence, an Augmented Lagrangian introduces a quadratic penalty on constraint violations [cite: 22, 23, 28]:

$$\mathcal{L}_{\text{aug}}(\theta, \boldsymbol{\lambda}) = \mathbb{E}_{\pi_\theta} \left[ R_{\text{quality}}(x, o) \right] - \sum_{k=1}^K \lambda_k \left( \mathbb{E}_{\pi_\theta} [C_k] - l_k \right) - \frac{\mu}{2} \sum_{k=1}^K \left[ \max\left(0, \mathbb{E}_{\pi_\theta} [C_k] - l_k\right) \right]^2$$

where $\mu > 0$ acts as a penalty parameter that penalizes constraint violations even when $\lambda_k$ is small [cite: 23, 28].

Regularized Policy Gradient Primal-Dual (RPG-PD) adds an explicit token entropy regularization term $\tau \mathcal{H}(\pi_\theta)$ to the primal objective [cite: 23, 24]:

$$\mathcal{L}_{\text{RPG-PD}}(\theta, \boldsymbol{\lambda}) = \mathcal{L}(\theta, \boldsymbol{\lambda}) + \tau \mathcal{H}(\pi_\theta)$$

Entropy regularization prevents the policy distribution from collapsing into deterministic local minima, ensuring sufficient exploration to discover completions that are both factually compliant and stylistically rich [cite: 23, 24].

## Comparative Taxonomy of Reward Modeling Paradigms

| Reward Paradigm | Primary Supervision Signal | Computational Complexity | Reward Hacking & Collapse Risk | Generalization & Tone Sensitivity | Optimal Usage Context |
| --- | --- | --- | --- | --- | --- |
| Deterministic Rule Verifiers (RLVR) | Programmatic execution, regex, schema checkers, game state validators [cite: 8, 29]. | Low (CPU-bound execution checks; zero model inference cost) [cite: 8]. | High risk of length and mode collapse due to binary, ungradated outcome penalties [cite: 1, 2, 7]. | Zero generalization outside codified rules; insensitive to prose style or tone [cite: 6]. | Verifiable domains: code execution, SQL generation, formal logic, legal move verification [cite: 8, 29]. |
| Learned Scalar Reward Models (RM) | Neural preference model trained on pairwise human comparisons [cite: 2, 6]. | Moderate (Forward pass of scalar RM per completion) [cite: 2, 6]. | High vulnerability to style hacking, verbosity exploitation, and token pattern gaming [cite: 4, 6]. | High capability to evaluate open-ended nuance, prose quality, and stylistic alignment [cite: 6, 30]. | General assistant alignment, conversational tone, open-ended creative writing [cite: 6]. |
| Process Reward Models (PRM) | Step-by-step correctness scores assigned to intermediate reasoning steps [cite: 15, 17]. | High (Multiple forward passes or dense scoring across sequence steps) [cite: 15, 17]. | Moderate; mitigates outcome shortcuts but vulnerable if step verifiers drift [cite: 15, 17]. | Moderate; effective at logical structure, limited on subjective prose elegance [cite: 15, 29]. | Complex multi-step CoT reasoning, mathematical proofs, long-form problem solving [cite: 15, 17, 29]. |
| LLM-as-a-Judge Evaluators | Prompted frontier LLM evaluating completions via zero-shot/few-shot rubrics [cite: 4, 5, 31]. | Very High (Requires full autoregressive generation from judge model) [cite: 12, 31]. | High vulnerability to judge length bias, position bias, and self-preference bias [cite: 4, 5, 31]. | Superior semantic generalization across unstructured, highly complex tasks [cite: 30, 31]. | Offline evaluation, synthetic preference dataset creation, meta-judging [cite: 4, 5, 30, 31]. |

### Paradigm Trade-Offs and Architectural Interactions

Deterministic verifiers provide ground-truth precision but offer sparse, ungradated feedback [cite: 2, 8]. Learned scalar reward models provide dense feedback but suffer from reward over-optimization (Goodhart's Law), where the policy learns to game proxy features such as verbosity or formatting without improving underlying quality [cite: 4, 6]. Combining them within a CMDP framework ensures that the deterministic verifier enforces strict domain boundaries while the learned reward model guides stylistic exploration within the feasible space [cite: 6, 20, 21].

Process Reward Models (PRMs) provide step-level supervision, identifying precisely which step in a reasoning chain contains an error [cite: 15, 17]. This structural credit assignment prevents credit assignment failure, where a policy is penalized for an entire sequence due to a late-stage error [cite: 15, 17].

However, PRMs require substantial computational overhead during training [cite: 15, 17]. LLM-as-a-Judge evaluators offer broad semantic flexibility but are unsuitable for direct inner-loop policy gradient updates due to inference latency, cost, and structural verbosity bias [cite: 4, 12, 31]. LLM judges are most effective in offline evaluation pipelines that utilize length-controlled regression models [cite: 5, 11, 12].

## Integrated Post-Training System Architecture

| Pipeline Stage | Core Mechanism | Technical Realization | Operational Safeguard |
| --- | --- | --- | --- |
| 1. Cold-Start SFT Warm-Up | Supervised Fine-Tuning on domain demonstrations [cite: 3, 8]. | Fine-tune base model on curated CoT reasoning and structured prose datasets [cite: 3, 8]. | Establishes baseline sequence length distributions and formatting templates prior to RL [cite: 3, 8]. |
| 2. Dual-Engine CMDP Policy Optimization | Primal-Dual GRPO with dynamic Lagrange multipliers [cite: 8, 20, 21]. | Optimize $R_{\text{quality}}$ while dynamically adjusting $\lambda_k$ based on violation rates $C_k$ [cite: 20, 21, 27]. | Prevents length collapse by decaying penalties when rule violation rates satisfy constraints [cite: 20, 21, 23]. |
| 3. Policy Regularization & Variance Monitoring | Length-normalized log-probs & length-variance tracking [cite: 13, 18]. | Apply SimPO/DMPO objectives; track rollout group length standard deviation ($\sigma_L$) [cite: 9, 13, 18]. | Triggers exploration rewards if $\sigma_L$ drops, preventing policy entropy collapse [cite: 18, 19]. |
| 4. Length-Debiased Evaluation Protocol | AlpacaEval 2.0 Length-Controlled (LC) Win-Rate [cite: 5, 11, 12]. | Fit Bradley-Terry logistic regression across balanced length buckets ($B$) [cite: 11, 12, 30]. | Eliminates length bias in judge evaluation, ensuring accurate win-rate comparisons [cite: 5, 12]. |

### System Execution Strategy

To resolve length collapse and evaluation asymmetries in verifier-guided RL, post-training pipelines should adopt a phased deployment strategy. First, cold-start supervised fine-tuning establishes initial formatting and length distributions, ensuring the model enters RL with sufficient initial competence to avoid immediate penalty loops [cite: 3, 8].

During policy optimization, Group Relative Policy Optimization is executed within a CMDP primal-dual framework [cite: 8, 20, 21]. Deterministic verifiers track rule violations as hard cost constraints, while a learned scalar or process reward model evaluates prose quality, tone, and key-point coverage [cite: 17, 20, 21].

Dynamic Lagrange multipliers adjust penalty weights automatically based on empirical constraint violation rates, preventing the optimization gradient from over-penalizing length when the model is compliant [cite: 20, 21, 27].

Throughout training, the system monitors intra-group length variance ($\sigma_L$) and applies forward-KL distribution matching (DMPO) or length-normalized likelihood margins (SimPO) to maintain policy entropy [cite: 9, 13, 18].

Finally, policy evaluation relies on length-controlled preference metrics like AlpacaEval 2.0 LC win-rate, isolating factual accuracy and quality from verbosity artifacts [cite: 5, 11, 12].

## Strategic Conclusion

Length collapse and evaluation ties in verifier-guided reinforcement learning result from mathematical mismatches between autoregressive penalty accumulation, linear reward scalarization, and judge verbosity biases [cite: 1, 2, 4, 7]. Treating factual rule adherence as a simple negative component in scalar reward functions forces the policy to truncate output length to minimize exposure to rule violations [cite: 1, 2].

By reframing post-training as a Constrained Markov Decision Process (CMDP), employing primal-dual Lagrangian optimization, integrating forward-KL distribution matching, and applying length-controlled evaluation protocols, post-training pipelines can successfully train language models that maintain strict domain compliance while producing thorough, high-quality responses [cite: 9, 12, 20, 21].

## References

1. Efficient Reasoning via Reward Model — arXiv. <https://arxiv.org/pdf/2511.09158>
2. (PDF) Are complicated loss functions necessary for teaching LLMs to reason?. <https://www.researchgate.net/publication/402860572_Are_complicated_loss_functions_necessary_for_teaching_LLMs_to_reason>
3. Scaling Reinforcement Learning for Content Moderation with Large Language Models — ACL Anthology. <https://aclanthology.org/2026.emnlp-industry.17.pdf>
4. Your LLM Judge Has a Length Bias, a Position Bias, and a Format Bias — and Nobody Is Auditing Yours — TianPan.co. <https://tianpan.co/blog/2026/04/27/llm-judge-bias-audit-length-position-format>
5. AlpacaEval - Learn AI — Miraheze. <https://ai.miraheze.org/wiki/AlpacaEval>
6. Bridging Offline and Online RL for LLMs | PDF | Machine Learning — Scribd. <https://www.scribd.com/document/906763877/2506-21495v1>
7. HOW FAR CAN UNSUPERVISED RLVR SCALE LLM TRAINING? — ICLR Proceedings. <https://proceedings.iclr.cc/paper_files/paper/2026/file/18929f3253e4d4e28086a89ab88d1276-Paper-Conference.pdf>
8. Reinforcement Learning for Reasoning Models (GRPO and RLVR) | RuneLab Corporate Course. <https://runelab.ai/en/course/reinforcement-learning-for-reasoning-models-grpo-and-rlvr>
9. Beyond Mode Collapse: Distribution Matching for Diverse Reasoning — arXiv. <https://arxiv.org/html/2605.19461v1>
10. CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution — arXiv. <https://arxiv.org/html/2603.17775v2>
11. S-SPPO: Semantic-Calibrated Self-Play Preference ... — OpenReview. <https://openreview.net/pdf?id=uVZUAKrA1t>
12. GitHub - tatsu-lab/alpaca_eval: An automatic evaluator for instruction-following language models. Human-validated, high-quality, cheap, and fast.. <https://github.com/tatsu-lab/alpaca_eval>
13. A Guide to Reinforcement Learning Post-Training for LLMs: PPO, DPO, GRPO, and Beyond. <https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms>
14. [Notes] PPO, GRPO, and GSPO | Veritable Tech Blog — Ceshine Lee. <https://blog.ceshine.net/post/ppo-grpo-gspo/>
15. FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience. <https://arxiv.org/html/2609.03241v1>
16. chrisliu298/awesome-on-policy-distillation — GitHub. <https://github.com/chrisliu298/awesome-on-policy-distillation>
17. Healthcare AI GYM for Medical Agents — arXiv. <https://arxiv.org/html/2605.02943v1>
18. Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification — arXiv. <https://arxiv.org/html/2606.03608v1>
19. (PDF) Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification — ResearchGate. <https://www.researchgate.net/publication/405852934_Exploiting_Verification-Generation_Gap_Test-Time_Reinforcement_Learning_with_Confidence-Conditioned_Verification>
20. A Constrained Multi-Objective Reinforcement Learning Framework. <https://proceedings.mlr.press/v164/huang22a/huang22a.pdf>
21. Constrained Markov Decision Processes — Matthew Landers. <https://mattlanders.net/constrained-mdps.html>
22. Cancellation-Free Regret Bounds for Lagrangian Approaches in Constrained Markov Decision Processes — Research Collection. <https://www.research-collection.ethz.ch/server/api/core/bitstreams/7d793edb-1810-43ac-aa70-604075de5019/content>
23. Deterministic Policy Gradient Primal-Dual Methods for Continuous-Space Constrained MDPs — Dongsheng Ding. <https://dongshed.github.io/papers/25rozadadeterministic.pdf>
24. Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs. <https://proceedings.neurips.cc/paper_files/paper/2023/file/d0949cbcec31c09431610553a284f94a-Paper-Conference.pdf>
25. Policy-Based Primal-Dual Methods for Convex Constrained Markov Decision Processes. <https://ojs.aaai.org/index.php/AAAI/article/view/26299/26071>
26. OFF-POLICY PRIMAL-DUAL SAFE REINFORCEMENT LEARNING — ICLR Proceedings. <https://proceedings.iclr.cc/paper_files/paper/2024/file/2f8b56543953d60f262fb2c4b85c50b3-Paper-Conference.pdf>
27. Adaptive Lagrangian Attention for Constrained Multimodal Multi-objective Optimization — ResearchGate. <https://www.researchgate.net/publication/413560547_Adaptive_Lagrangian_Attention_for_Constrained_Multimodal_Multi-objective_Optimization/fulltext/6a8c4c4241e813557796422f/Adaptive-Lagrangian-Attention-for-Constrained-Multimodal-Multi-objective-Optimization.pdf?origin=scientificContributions>
28. Solving Multiobjective Optimization Problems with Inequality Constraint Using An Augmented Lagrangian Function. <https://iapress.org/index.php/soic/article/view/1875>
29. Accepted papers — FoRLM @ NeurIPS'25. <https://reasoning-workshop.github.io/accepted/>
30. AlpacaEval 2.0: LLM Instruction Benchmark — Emergent Mind. <https://www.emergentmind.com/topics/alpacaeval_2-0>
31. AlpacaEval Definition, Win-Rate & FutureAGI Guide. <https://futureagi.com/glossary/alpacaeval/>
