# Comprehensive Framework for Evaluating Specialized Chess Explanation Models: Pedagogical Value, Automated Judgment, and Release Readiness

> [!NOTE]
> Source: Gemini Deep Research report, added 2026-11-06. This is a formatted copy. The content is not changed. A critical review of this report is in [analysis.md](analysis.md).

Evaluating specialized natural language models in complex, rule-bound domains requires a multi-faceted approach. In domain-specific tasks such as chess instruction, a language model must unify discrete symbolic calculation, tactical accuracy, and pedagogical prose [cite: 1, 2, 3]. A 2-billion parameter (2B) specialized chess explanation model operates at a critical architectural intersection: balancing limited parameter capacity against the demanding requirements of exact positional evaluation, multi-step tactical calculation, and user-adapted instructional explanation [cite: 1, 4, 5].

Current evaluation pipelines rely on automated rule checkers, 4-dimension rubric judges, and blind swap-consistent pairwise comparisons [cite: 6, 7]. However, establishing parity with human chess coaches and frontier large language models (LLMs) requires a deeper framework [cite: 1, 8, 9]. This report analyzes the evaluation methodologies, meta-evaluations of automated judges against expert human standards, and quantitative release readiness gateways necessary for small specialized models [cite: 5, 9, 10, 11].

## Evaluation Frameworks and Metrics for Explanation Quality, Pedagogical Efficacy, and User Preference

### Taxonomy of Explanation Quality: Move Accuracy, Line Substantiation, and Conceptual Coherence

Natural language explanations of complex decision sequences cannot be assessed on linguistic fluency alone [cite: 1, 2]. In domain-adapted architectures, such as models that couple a silent expert chess encoder with a language model decoder via cross-attention, explanation quality depends directly on underlying move selection strength [cite: 1, 2]. A fluent explanation that justifies an objective blunder is counterproductive [cite: 2]. Consequently, explanation quality must be evaluated across three interdependent dimensions: move accuracy, line substantiation, and conceptual coherence [cite: 2].

Move accuracy establishes the foundational requirement that the recommended action minimizes evaluation loss compared to engine ground truth, such as Stockfish 17 or Komodo Dragon 1 [cite: 6, 8]. It is evaluated quantitatively through centipawn loss (CPL), top-1 move agreement rates, and tactical puzzle accuracy [cite: 1, 2, 8].

Line substantiation measures the model's capacity to justify its chosen move through concrete variation analysis [cite: 2]. A substantiated explanation must outline the predicted principal variation (PV), analyze critical counter-responses, and demonstrate why alternative candidate moves are inferior [cite: 2].

Conceptual coherence measures the degree to which natural language prose accurately maps to underlying board state dynamics without hallucinating tactical motifs or misidentifying piece interactions [cite: 1, 2, 3].

Small specialized models frequently suffer from verbalization debt, a phenomenon where implicit positional representation or tree-search capabilities outpace the model's natural language generation fidelity [cite: 1, 12]. While search-distillation algorithms—such as natural-language analogs of Bellman updates—can dramatically increase playing strength and tactical accuracy, translating latent look-ahead trees into clear conceptual prose remains a structural bottleneck [cite: 1, 2, 12].

### Pedagogical Scaffolding and Curriculum-Based Abstraction Layers

Evaluating an explanation model against professional chess coaches requires metrics that capture pedagogical efficacy [cite: 13]. Direct answer disclosure is rarely optimal for human skill acquisition; effective instruction relies on diagnostic scaffolding that provides targeted hints, highlights structural motifs, and prompts multi-step calculation [cite: 13]. A robust evaluation framework structures its test suite across a spectrum of ascending abstraction levels [cite: 3].

As demonstrated in domain-specific benchmarks like ChessQA, evaluation items should be categorized into five structured curriculum tiers [cite: 3]:

1. **Structural Mechanics:** Verifying single-step move legality, board awareness, and static piece geometry [cite: 3].
2. **Static Motifs:** Identifying non-calculative tactical and positional patterns, including pins, skewers, batteries, outposts, and weak color complexes [cite: 3].
3. **Short Calculation and Concrete Tactics:** Executing accurate multi-ply line verification and calculating forcing sequences [cite: 3].
4. **Position Judgment:** Generating quantitative and qualitative evaluations that mirror engine assessment scores [cite: 3].
5. **Semantic Synthesis:** Producing high-level natural language prose that contextualizes strategic plans, pawn structures, and long-term imbalances [cite: 3].

Pedagogical evaluation protocols should measure whether the model adapts its explanation depth to user skill tiers [cite: 3, 13]. For lower-rated users, the model must emphasize basic motif identification and immediate tactical safety [cite: 3]. For advanced users, it must prioritize subtle positional imbalances, pawn structure transformations, and deep principal variations [cite: 2, 3].

### Quantifying User Preference: Beyond Bradley-Terry to Intransitive Preference Architectures

In automated preference modeling, pairwise win rates are traditionally aggregated using the scalar Bradley-Terry (BT) model [cite: 14, 15, 16]:

$$
\mathbb{P}(y_i \succ y_j \mid x) = \sigma(r(y_i \mid x) - r(y_j \mid x)) = \frac{1}{1 + \exp\left(-(r(y_i \mid x) - r(y_j \mid x))\right)}
$$

where $r(y \mid x)$ represents a learned scalar reward assigned to response $y$ under prompt $x$ [cite: 15, 16].

However, natural language explanations in complex technical domains violate the core transitivity assumptions of the Bradley-Terry model [cite: 17, 18]. Human preferences over domain explanations are multi-dimensional, governed by trade-offs between conciseness, depth of calculation, instructional tone, and formatting [cite: 16]. User preferences frequently exhibit cyclic or intransitive properties, where Explanation A is preferred over B for conciseness, B over C for tactical depth, yet C is preferred over A for pedagogical clarity [cite: 17, 18].

To address these limitations, evaluation pipelines should adopt general preference or game-theoretic models, such as Direct Nash Optimization (DNO) or Self-Play Preference Optimization (SPPO) architectures [cite: 17, 18]. These models treat pairwise evaluation as a constant-sum two-player game aimed at identifying a Nash equilibrium policy $\pi^*$, where a response's quality is defined by its expected win rate against the policy distribution itself rather than a static scalar reward [cite: 17, 18]:

$$
r(x, y) = \mathbb{E}_{y' \sim \pi}\left[\mathcal{P}(y \succ y' \mid x)\right]
$$

Furthermore, standard pairwise win rates are systematically confounded by response token length [cite: 7, 14]. To isolate genuine explanation quality from sheer verbosity, preference evaluation must report Length-Controlled Win Rates (LC-WR) [cite: 14, 17, 18]. Length-controlled evaluation fits a logistic regression model over preference outcomes while residualizing the token length differential between candidate responses, preventing verbose yet tactically shallow responses from inflating preference scores [cite: 7, 14, 19].

| Evaluation Dimension | Primary Metric / Probe | Benchmark Baseline | Target Threshold | Primary Failure Mode / Artifact |
| --- | --- | --- | --- | --- |
| Tactical Accuracy | Top-1 Move Match; Centipawn Loss (CPL); Puzzle Accuracy | Engine Ground Truth (Stockfish 17 / Komodo Dragon 1) [cite: 6, 8] | CPL $< 15$; Top-1 Match $> 85\%$ [cite: 1, 8] | Recommending tactical blunders with high linguistic confidence [cite: 2]. |
| Line Substantiation | Principal Variation (PV) Depth; Refutation Coverage | ChessBase 17 Human Annotation; Engine Search Trees [cite: 2, 3] | Mean PV Depth $\ge 3$ plies; Refutation Rate $> 80\%$ [cite: 2] | Omitting critical defensive counter-responses in line calculations [cite: 2]. |
| Conceptual Coherence | Semantic Feature Alignment; Motif Naming Accuracy | ChessQA Semantic Category Benchmark [cite: 3] | Motif Naming Accuracy $> 90\%$ [cite: 3] | Hallucinating non-existent pins, skewers, or file control [cite: 3, 7]. |
| Pedagogical Efficacy | Diagnostic Scaffolding Index; Direct-Answer Ratio | MathTutorBench Scaffolding Rubric [cite: 13] | Scaffolding Score $> 4.0 / 5.0$ [cite: 13] | Immediately revealing solutions rather than guiding reasoning [cite: 13]. |
| User Preference | Length-Controlled Win Rate (LC-WR); Nash Equilibrium Win % | AlpacaEval 2.0 LC-WR Protocol; SPPO / DNO [cite: 14, 17, 18] | LC-WR $> 50.0\%$ vs Frontier LLM Baselines [cite: 14, 17] | Verbosity bias inflating win rates for unhelpful long text [cite: 7, 14]. |

## Meta-Evaluation of Automated LLM Judges versus Expert Human Ratings

### Statistical Metrics for Inter-Rater Reliability and Expert Alignment

Deploying automated LLM judges to evaluate specialized 2B outputs requires rigorous meta-evaluation ("evaluating the evaluator") [cite: 7, 20]. Relying solely on Pearson or Spearman correlation metrics ($r$) between judge scores and human ratings is insufficient; high correlation can exist even when an automated judge suffers from severe calibration offset or systematic rating compression [cite: 9, 21].

Automated judge validity must be established using chance-corrected inter-rater agreement statistics, primarily Cohen’s Kappa ($\kappa$) for unweighted pairwise verdicts and Weighted Kappa ($\kappa_w$) or Krippendorff’s Alpha ($\alpha$) for multi-criteria rubric scales [cite: 9, 10, 22].

To achieve Tier 1 judging status in technical domains, an LLM judge must meet a two-part criterion [cite: 9]:

1. **Absolute Agreement Threshold:** $\kappa \ge 0.78$, placing performance in the substantial to almost perfect alignment range [cite: 9].
2. **Human-Likeness Test:** The judge's agreement score with human experts must fall within the natural variance of human-to-human agreement [cite: 9]. Defined via the $z$-score statistic:

   $$
   z = \frac{\kappa_{\text{LLM}} - \mu_{\text{human}}}{\sigma_{\text{human}}}
   $$

A judge passes the human-likeness test when $|z| < 1.0$ [cite: 9]. Models with $z > 1.0$ exhibit hyper-consistency (artificially zero variance), while models with $z < -1.0$ exhibit excessive noise or fundamental divergence from expert standards [cite: 9, 10].

Empirical audits reveal that while frontier models achieve 80–85% raw agreement with human reviewers on broad conversational tasks [cite: 7, 10], agreement drops significantly in domain-specific technical environments requiring multi-step verification [cite: 23, 24].

### Mathematical Modeling and Mitigation of Systematic Judge Biases

Automated LLM judges introduce systematic statistical biases that distort evaluation signals if left uncorrected [cite: 7, 25]. In domain-specific technical evaluation, four primary biases must be isolated and neutralized: position bias, verbosity bias, self-preference bias, and anchoring bias [cite: 7, 19, 25, 26].

#### Position Bias

In pairwise evaluation, LLM judges frequently favor whichever candidate is presented first or second, demonstrating up to 40% to 55% position inconsistency across presentation orderings [cite: 7, 19]. To mathematically diagnose position bias, presentation reversal rates must be decomposed into content-noise versus slot-following dynamics [cite: 19, 27]. If a judge chooses candidate outputs based on independent content evaluation with probability $p$, the expected reversal rate under symmetric noise is bounded [cite: 19, 27]:

$$
R_{\text{noise}} = 2p(1-p) \le 0.50
$$

Conversely, if the judge exhibits structural first-position preference with slot-following probability $q$, the reversal rate is [cite: 19, 27]:

$$
R_{\text{slot}} = q^2 + (1-q)^2
$$

When $q > 0.50$, $R_{\text{slot}}$ strictly exceeds 0.50, approaching 1.0 under deterministic slot-following [cite: 19, 27]. Empirical evaluations show models like Gemma exhibiting reversal rates of 78.8% ($q \approx 0.88$), whereas more balanced judges like Qwen display lower reversal rates around 42.1% [cite: 19, 27].

Mitigation requires full blind-swap evaluation ($A \text{ vs } B$ and $B \text{ vs } A$). A candidate is awarded a win if and only if it is selected by the judge in both presentation orders; ordering ties are discarded [cite: 7].

#### Verbosity Bias

LLM judges consistently assign higher scores to longer responses, even when extra length introduces non-functional padding or minor factual errors [cite: 7, 25]. Mitigation requires incorporating explicit length-neutralizing instructions into the system prompt ("Length is not a proxy for quality. A concise 50-word accurate explanation must score higher than a 300-word verbose explanation containing redundant lines") [cite: 7]. Post-hoc, length-controlled Bradley-Terry residualization must be applied to compute LC-WR [cite: 14].

#### Self-Preference and Model Family Bias

Judges assign statistically significant higher scores to outputs generated by their own model family or architectural lineage [cite: 7, 19, 25]. To isolate model family bias from genuine candidate quality, candidate-judge scoring matrices must undergo double-centering residualization [cite: 19]. The observed win rate $P(j, c)$ of candidate family $c$ scored by judge family $j$ is modeled as [cite: 19]:

$$
P(j, c) = \mu + a_j + b_c + \gamma_{jc}
$$

where $\mu = 0.50$ in a balanced zero-sum evaluation, $a_j$ represents the judge main effect, $b_c$ captures true candidate quality, and $\gamma_{jc}$ represents the isolated judge-by-candidate interaction parameter [cite: 19]. Unadjusted evaluation metrics conflate candidate ability $b_c$ with family affinity $\gamma_{jc}$ [cite: 19].

Mitigation requires enforcing cross-family judging protocols [cite: 7, 11]. A 2B model distilled from a specific teacher architecture must never be evaluated by a judge from that same model family [cite: 7, 11].

#### Anchoring and Sycophancy Biases

LLM judges exhibit anchoring bias by over-weighting the initial criteria listed in a rubric prompt, while sycophancy bias leads judges to favor responses that echo user misconceptions [cite: 7, 26]. Mitigation involves permuting rubric criteria ordering across evaluation runs and injecting explicit adversarial misstatements into user prompts to test whether the judge rewards factual correction [cite: 7, 26].

| Bias Mode | Mathematical / Behavioral Signature | Primary Diagnostic Test | Operational Mitigation Strategy |
| --- | --- | --- | --- |
| Position Bias | $R_{\text{observed}} > 0.50$; $q = \sqrt{\frac{R_{\text{slot}} + 1}{2}}$ [cite: 19, 27] | Swap Consistency Test ($A/B$ vs $B/A$) [cite: 7, 26] | Enforce strict dual-swap evaluation; discard ordering ties [cite: 7]. |
| Verbosity Bias | $\frac{\partial \text{Score}}{\partial \text{Tokens}} > 0$ given equal correctness [cite: 7] | Token-Length Correlation on Isomorphic Inputs [cite: 26] | Prompt-level length penalization; Length-Controlled Win Rate [cite: 7, 14]. |
| Self-Preference | Significant diagonal lift $\gamma_{jj} > 0$ in double-centered matrix [cite: 19] | Interaction ANOVA ($P(j, c) = \mu + a_j + b_c + \gamma_{jc}$) [cite: 19] | Cross-family judge assignment; exclude parent teacher LMs [cite: 7, 11]. |
| Anchoring Bias | Differential scoring variance based on rubric ordering | Permutation ANOVA across Rubric Orderings [cite: 7] | Randomize rubric dimension sequence per API call [cite: 7]. |

### Human-in-the-Loop Adjudication and Ground-Truth Calibration Workflows

Automated judges cannot function as ungrounded evaluators [cite: 11]. A robust evaluation pipeline relies on a curated Ground Truth Calibration Dataset comprising 50 to 100 expert-annotated domain traces [cite: 11]. Human experts, including Grandmasters, International Masters, and professional chess coaches, evaluate these reference traces against explicit behavioral rubrics [cite: 10, 11].

The meta-evaluation pipeline operates through four systematic steps [cite: 11]:

1. **Rubric Behavioral Anchoring:** Translating abstract concepts into concrete domain criteria with explicit score anchors (e.g., Score 1: Recommends move causing blunder $> 1.5$ CPL; Score 5: Recommends engine top-1 move with full 3-ply refutation of major alternatives) [cite: 7, 10, 11].
2. **Calibration Benchmarking:** Running the automated judge against the ground-truth dataset to compute baseline Cohen’s $\kappa$ and dimension-specific disagreement vectors [cite: 11]. Target threshold for production deployment is set at $0.75 \le \kappa \le 0.90$ [cite: 11]. Scores above 0.90 often signal overly simplified rubrics that fail to capture nuanced domain errors [cite: 11].
3. **Disagreement Review Workflow:** Routing cases where the LLM judge and human experts diverge to an adjudication panel [cite: 11]. Disagreements highlight ambiguous rubric wording, prompt vulnerability, or edge-case domain complexity, driving prompt iterative refinement [cite: 11].
4. **Continuous Monthly Audits:** Sampling a fixed percentage (e.g., 100 outputs monthly) of live production traffic for expert re-audit to detect evaluation drift, prompt injection vulnerabilities, and performance degradation across novel domain distributions [cite: 7, 11].

## Release Readiness Criteria and Verification Gateways for Small Specialized Language Models

### Architectural Integration and Capacity Constraints

For a 2B parameter specialized model, direct autoregressive next-token prediction on pure text corpora is insufficient to achieve master-level domain reasoning [cite: 1, 2]. Parameter-constrained models lack the capacity to store vast multi-step search trees directly in their feed-forward layers [cite: 1, 5]. State-of-the-art specialized performance is achieved by coupling a domain-expert encoder (such as a high-throughput, silent expert chess engine like AlphaZero or Stockfish-derived representations) with an instruction-tuned language model decoder using cross-attention mechanisms [cite: 1, 2].

To maximize instructional efficacy, model training should incorporate iterative search distillation [cite: 1, 2]. During training, the model evaluates board positions after candidate moves, executes localized look-ahead search, consolidates these search trajectories into textual explanations, and distills the consolidated insights back into the parameter weights [cite: 1, 2]. As demonstrated in architectures like QUEEN, this distillation process enables compact models to achieve substantial Elo rating gains and approach frontier LLM performance in explanation coherence despite possessing orders of magnitude fewer parameters [cite: 1, 2].

### Schema Compliance, Edge Constraints, and Safety-Capacity Trade-offs

Deploying small language models ($\le 7\text{B}$, and specifically 2B specialized models) introduces operational constraints distinct from those of cloud-hosted frontier models [cite: 5, 28, 29].

#### Schema Compliance and Structured Formatting

Small models are prone to strict-evaluation failures caused by formatting deviations [cite: 30]. When downstream application interfaces require structured JSON, Universal Chess Interface (UCI) notation, or strict FEN strings, output generation can break if the model includes extraneous conversational commentary, reasoning traces, or malformed whitespace [cite: 30]. Release pipelines must evaluate schema compliance deterministically using code-based validators rather than LLM judges [cite: 11, 30].

#### Post-Training Quantization (PTQ) and Compression Impact

To run efficiently on edge hardware, 2B models are frequently subjected to 4-bit or 8-bit quantization (e.g., AWQ, GPTQ) and structured pruning [cite: 5, 29]. However, post-training compression disproportionately degrades multi-step reasoning capabilities and safety guardrails compared to simple text perplexity [cite: 5, 29]. Empirical studies indicate that safety-critical alignments occupy a tiny fraction ($\sim 3\%$) of parameter space, making small quantized models highly vulnerable to compression-induced safety degradation and jailbreaks [cite: 5, 31].

#### Capacity-Safety Trade-off (The Alignment Tax)

Allocating parameter capacity in a 2B model is a zero-sum trade-off between domain knowledge, instruction following, and safety refusals [cite: 5]. Over-aligning a small model leads to excessive over-refusal rates, where benign domain prompts containing aggressive terminology (e.g., "attack the king", "execute a brutal sacrifice", "destroy the pawn center") trigger false-positive safety refusals [cite: 5]. Conversely, shallow alignment creates vulnerabilities to prompt injection attacks embedded inside board state strings or commentary histories [cite: 5, 11].

### Verification Gateways for Public Model Release

Before authorizing the public release of a specialized 2B chess explanation model, the candidate weight checkpoint must pass a sequential, non-negotiable set of quantitative release gateways across six operational domains. Failure at any single gateway halts deployment.

| Deployment Gateway | Primary Metric | Quantitative Release Threshold | Operational Verification Method |
| --- | --- | --- | --- |
| 1. Rule Legality & Schema | Invalid Action Rate; UCI Compliance Rate | $0.00\%$ illegal moves; $> 99.5\%$ strict JSON/UCI schema compliance [cite: 6, 8, 30] | Deterministic Python-Chess Parsing Engine [cite: 3, 11]. |
| 2. Tactical Accuracy | Top-1 Move Match; Blunder Rate ($>1.5$ CPL); Puzzle Score | Top-1 Match $> 85\%$ vs Engine; Blunder Rate $< 2.0\%$ on standard tactical suites [cite: 1, 8] | Automated Engine Evaluation Benchmarks [cite: 6, 8]. |
| 3. Explanation Quality | Length-Controlled Win Rate vs Frontier LLMs; PV Depth | LC-WR $> 50.0\%$ vs GPT-4o / Claude Sonnet; Mean PV Depth $\ge 3$ plies [cite: 2, 14, 17] | Cross-Family Dual-Swap Automated LLM Judge [cite: 7, 19]. |
| 4. Expert Alignment | Cohen’s Kappa ($\kappa$) against Expert Ground Truth | $\kappa \ge 0.78$; Human-Likeness score $\lvert z \rvert < 1.0$ on 100-sample calibration set [cite: 9] | Double-Blind Grandmaster Review Panel [cite: 9, 11]. |
| 5. Safety & Over-Refusal | Attack Success Rate (ASR); Benign Over-Refusal Rate | ASR $< 1.0\%$ on domain jailbreaks; Over-Refusal $< 0.5\%$ on aggressive chess prose [cite: 5, 31] | Adversarial Red-Teaming & Benchmark Suites [cite: 5, 31]. |
| 6. Edge Efficiency | P99 Latency per Turn; Quantization Stability | P99 Latency $< 250\text{ ms}$ on target hardware; $< 2.0\%$ score loss post-quantization [cite: 28, 29] | Hardware Profiling on Target Edge Platform [cite: 28, 29]. |

## Strategic Recommendations and Synthesis

Validating and releasing a specialized 2B chess explanation model capable of competing with commercial coaches and frontier LLMs requires an integrated evaluation pipeline [cite: 1, 8, 9].

Engineering teams must first transition from scalar Bradley-Terry preference formulations to game-theoretic optimization models, such as Direct Nash Optimization or Self-Play Preference Optimization [cite: 17, 18]. User preference must be quantified using Length-Controlled Win Rates to ensure length artifacts do not mask shallow reasoning [cite: 7, 14].

Automated judging pipelines must be rigorously debiased [cite: 7]. Implementing full blind dual-swap orderings neutralizes position bias [cite: 7, 19], while selecting cross-family judges prevents self-preference inflation [cite: 7, 11]. Automated judges should only be deployed after achieving verified expert agreement ($\kappa \ge 0.78$) and passing human-likeness testing ($|z| < 1.0$) against a calibrated ground-truth trace dataset [cite: 9, 11].

Finally, the model must be evaluated along a structured 5-tier abstraction curriculum, measuring both concrete calculation accuracy and diagnostic pedagogical scaffolding [cite: 3, 13]. Prior to public deployment, candidate weight checkpoints must satisfy strict release gateways, ensuring zero illegal moves, $<2.0\%$ tactical blunder rates, robust adversarial safety, and minimal performance loss post-quantization on target edge hardware [cite: 5, 8, 28, 29, 30]. Adopting this multi-layered framework ensures that small specialized models deliver master-level tactical precision alongside transparent, high-quality pedagogical explanations [cite: 1, 2, 3].

## References

1. [2610.03695] Language Models that Play Chess and Explain Their Moves — arXiv. <https://arxiv.org/abs/2610.03695>
2. Language Models that Play Chess and Explain Their Moves — arXiv. <https://arxiv.org/html/2610.03695v1>
3. ChessQA: Evaluating Large Language Models for Chess Understanding — ResearchGate. <https://www.researchgate.net/publication/397006403_ChessQA_Evaluating_Large_Language_Models_for_Chess_Understanding>
4. Small Language Models (SLM): A Comprehensive Overview — Hugging Face. <https://huggingface.co/blog/jjokah/small-language-model>
5. Safety and Alignment of Small LanguageModels: A Systematic Survey of Methods,Evaluation, and Open Challenges — ResearchGate. <https://www.researchgate.net/publication/404771125_Safety_and_Alignment_of_Small_LanguageModels_A_Systematic_Survey_of_MethodsEvaluation_and_Open_Challenges>
6. BENCHMARKING REASONING AND INSTRUCTION-FOLLOWING IN LLMS THROUGH CHESS — OpenReview. <https://openreview.net/pdf?id=65R1Dbfwzk>
7. LLM-as-Judge: Automated AI Quality Evaluation for Product Teams. <https://www.institutepm.com/knowledge-hub/llm-as-judge-ai-evaluation>
8. LLM CHESS: Benchmarking Reasoning and Instruction-Following in LLMs through Chess — OpenReview. <https://openreview.net/pdf?id=i0UHrCKEXZ>
9. A Comprehensive Analysis of LLM Judge Capability Through Human Agreement — arXiv. <https://arxiv.org/html/2510.09738v1>
10. LLM Judge Agreement: What ImpactBench's Expert Audit Reveals — Kili Technology. <https://kili-technology.com/blog/data-story-impactbench-llm-judge-expert-audit>
11. How to use LLM-as-judge for agent evaluation with Label Studio. <https://labelstud.io/learning-center/how-to-use-llm-as-judge-for-agent-evaluation/>
12. Language Models that Play Chess and Explain Their Moves (Oct 2026) — YouTube. <https://www.youtube.com/watch?v=yZMQcjAMsnY>
13. MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors — ResearchGate. <https://www.researchgate.net/publication/389392159_MathTutorBench_A_Benchmark_for_Measuring_Open-ended_Pedagogical_Capabilities_of_LLM_Tutors>
14. Segmenting Text and Learning Their Rewards for Improved RLHF in. <https://www.alphaxiv.org/abs/2501.02790>
15. Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment — arXiv. <https://arxiv.org/html/2410.02197v3>
16. Debiasing Online Preference Learning via Preference Feature Preservation — ACL Anthology. <https://aclanthology.org/2025.findings-acl.1034.pdf>
17. Self-Play Preference Optimization for Language Model Alignment — arXiv. <https://arxiv.org/html/2405.00675v5>
18. sources/arxiv-2404.03715.md · rl-llm-wiki/knowledge-base at refs/pr/117 — Hugging Face. <https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/blob/refs%2Fpr%2F117/sources/arxiv-2404.03715.md>
19. Measuring Family-Conditioned Preference in LLM-as-Judge Panels — arXiv. <https://arxiv.org/pdf/2609.17857>
20. EvalSense: A Framework for Domain-Specific LLM (Meta-)Evaluation — arXiv. <https://arxiv.org/html/2602.18823v1>
21. Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data — arXiv. <https://arxiv.org/html/2410.13341v4>
22. Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why — OpenReview. <https://openreview.net/forum?id=DHI9h7Zcni>
23. Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks — alphaXiv. <https://www.alphaxiv.org/abs/2410.20266>
24. posts/llm-as-a-judge-vs-expert-graders.md ... — Hugging Face. <https://huggingface.co/datasets/sarathkavuru/anthroly-research/blob/main/posts/llm-as-a-judge-vs-expert-graders.md>
25. LLMs Judging LLMs: A Simplex Perspective — arXiv. <https://arxiv.org/html/2505.21972v3>
26. Position Bias in LLM Judges: Measurement and Mitigation — Interactive. <https://mbrenndoerfer.com/writing/position-bias-in-llm-judges>
27. Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels. <https://arxiv.org/html/2609.17857v1>
28. Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification — arXiv. <https://arxiv.org/html/2601.07790v1>
29. Small Language Models: Architecture, Evolution, and the Future of Artificial Intelligence. <https://www.preprints.org/manuscript/202601.0973>
30. Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification — MDPI. <https://www.mdpi.com/2073-431x/15/7/430>
31. EASE: Practical and Efficient Safety Alignment for Small Language Models — arXiv. <https://arxiv.org/html/2511.06512v1>
