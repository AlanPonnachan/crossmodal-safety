
# Limits of Linear Cross-Modal Steering: Investigating Safety Representations in Vision-Language Models

This repository contains the full code, experimental pipelines, raw CSV logs, and visual diagnostics for investigating cross-modal safety representation alignment and causal activation steering in Vision-Language Models (VLMs).

**Target Model:** `Qwen/Qwen2.5-VL-7B-Instruct`  
**Evaluation Judge:** `gpt-5.1` (LLM-as-a-Judge on 1–5 policy compliance)  
**Dataset:** 64 category-stratified counterfactual pairs derived from `MM-SafetyBench++`

---

##  Core Empirical Findings

1. **Category-Driven Alignment:** Paired cross-modal cosine similarity rises in deep layers ($C_{\text{paired}} \approx 0.137$ at L20), but is largely accounted for by category-level background geometry ($C_{\text{perm}} \approx 0.144$). Excess pair-specific alignment ($\Delta = +0.0213$ at Layer 2) does not survive single-step max-statistic FWER correction ($p_{\text{adj}} = 0.161$).
2. **Modality-Associated Geometry:** 2D PCA on Layer 27 normalized displacement vectors reveals that PC1 explains **37.6% of variance**, cleanly separating text- and vision-derived displacements into distinct regions.
3. **Latent Perturbation Without Behavioral Transfer:** Injecting 1D text vectors ($V_U - \alpha \mathbf{v}$) at $p_{\text{gen}}$ perturbs internal distributions ($D_{\text{JS}} = 0.0209$ vs. $0.0021$ null), but fails to improve behavioral safety ($\Delta B = -0.1823$), performing indistinguishably from random Gaussian noise ($\Delta B = -0.1719$).
4. **Failure of Native Positive Controls:** Steering with the modality's *own* visual safety vector ($d_V$) also fails ($\Delta B = -0.1589$).
5. **Spatial Scope Attenuation:** Broadcasting additive vectors across all visual patch tokens (`visual_tokens`) yields near-zero next-token distribution shift ($D_{\text{JS}} \le 0.0002$), compared to interventions at $p_{\text{gen}}$ ($D_{\text{JS}} = 0.0367$).

---

##  Repository Structure & Artifacts

```text
├── build_dataset.py          # Builds N=64 balanced counterfactual pairs from MM-SafetyBench++
├── geometric_alignment.py    # Cross-modal cosine extraction & B=100k FWER permutation test
├── causal_interventions.py   # Multi-layer causal patching sweep (Layers 2, 18, 20, 22, 25, 27)
├── matrix_ablation.py       # 1-Day Ablation Matrix (Scope vs. Representation & Ridge W*dT)
├── behavioral_eval.py        # Parallelized GPT-5.1 behavioral evaluation (1-5 policy scale)
├── plot_results.py           # Publication figures (Figs 1–4) & final causal summary
├── plot_ablation.py         # Scope ablation heatmaps (Figs A & B)
├── plot_diagnostics.py       # Cosine matrix heatmap (Fig 5) & 2D PCA Modality Gap (Fig 6)
├── utils.py                     # Qwen2.5-VL loader, ChatML formatting, p_gen locator
├── pyproject.toml               # Python dependencies
└── results/
    ├── geometry_report.md       # Layer-wise alignment stats, p-values, LOCO diagnostics
    ├── final_causal_report.md   # Behavioral transfer (ΔB) vs. Latent shift (D_JS) at α=1.0
    ├── ablation_matrix_report.md# Layer-by-layer scope ablation table
    ├── ablation_final_summary.md# Aggregated 2x4 ablation matrix summary
    ├── geometry_metrics.csv     # Full layer-by-layer raw geometric and norm statistics
    ├── causal_results.csv       # Generation completions & D_JS across layers and alphas
    ├── behavioral_results.csv   # GPT-5.1 safety scores (1-5) and justification logs
    ├── ablation_matrix_results.csv # Raw generations and D_JS for scope ablations
    ├── ablation_behavior.csv    # Evaluated ablation generations with GPT-5.1 scores
    ├── fig1_geometric_alignment.png    # Layer-wise alignment trajectory (C_paired vs C_perm)
    ├── fig2_latent_js_shift.png        # Latent D_JS distribution shift heatmap across layers
    ├── fig3_behavioral_transfer.png    # Behavioral transfer (ΔB) bar chart vs. controls
    ├── fig4_dose_response.png          # Safety score vs. intervention multiplier (α)
    ├── fig5_cosine_matrix_blocks.png   # 64x64 Cross-modal cosine similarity matrix (Layer 27)
    ├── fig6_pca_modality_gap.png       # 2D PCA projection of safety displacement vectors
    ├── figA_ablation_latent_matrix.png # Scope vs. Representation latent shift (D_JS) heatmap
    └── figB_ablation_behavior_matrix.png # Scope vs. Representation behavioral transfer (ΔB)
```

---

##  Reproduction & Pipeline Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/AlanPonnachan/crossmodal-safety.git
cd crossmodal-safety

# Install dependencies using uv
uv sync
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory for the behavioral judge:
```bash
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
```

### 3. Step-by-Step Execution

```bash
# Step 1: Extract 64 balanced counterfactual multimodal pairs
uv run python build_dataset.py --num_pairs 64

# Step 2: Run geometric alignment & B=100,000 permutation testing
uv run python geometric_alignment.py --permutations 100000 --seed 42

# Step 3: Run multi-layer causal patching sweep across decoder layers
uv run python causal_interventions.py --layers 2 18 20 22 25 27 --alphas 0.5 1.0 2.0 --max_tokens 75

# Step 4: Run spatial scope and linear mapping ablation matrix
uv run python matrix_ablation.py --layers 18 20 22 25 27 --alpha 1.0 --max_tokens 75

# Step 5: Score generations with GPT-5.1 LLM-as-a-judge
uv run python behavioral_eval.py --input_csv results/causal_results.csv --output_csv results/behavioral_results.csv --concurrency 10
uv run python behavioral_eval.py --input_csv results/ablation_matrix_results.csv --output_csv results/ablation_behavior.csv --concurrency 10

# Step 6: Generate all publication figures and diagnostic plots
uv run python plot_results.py
uv run python plot_ablation.py
uv run python plot_diagnostics.py --layer 27
```

