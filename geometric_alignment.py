import argparse
import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from utils import load_model_and_processor, format_prompt, get_generation_position

def run_fwer_and_layer_permutation_tests(layer_matrices, categories, B=100000, seed=42):
    """
    Computes equal-category macro-averaged Delta, empirical null distributions,
    layer-wise raw p-values, and single-step max-statistic FWER-adjusted p-values.
    """
    N = len(categories)
    num_layers = len(layer_matrices)
    rng = np.random.default_rng(seed)
    
    cat_to_indices = defaultdict(list)
    for i, cat in enumerate(categories):
        cat_to_indices[cat].append(i)
        
    bad_cats = {k: len(v) for k, v in cat_to_indices.items() if len(v) < 2}
    if bad_cats:
        raise ValueError(f"CRITICAL ERROR: Categories with fewer than 2 pairs cannot be permuted: {bad_cats}")
        
    obs_deltas = {}
    obs_c_paired = {}
    obs_c_perm = {}
    cat_deltas_per_layer = {l: {} for l in range(num_layers)}
    G_k_values = {l: {} for l in range(num_layers)}
    
    for l, mat in layer_matrices.items():
        delta_k_list = []
        cp_k_list = []
        cperm_k_list = []
        
        for cat, indices in cat_to_indices.items():
            nk = len(indices)
            submat = mat[np.ix_(indices, indices)]
            
            c_paired_k = np.mean(np.diag(submat))
            G_k = np.mean(submat)
            G_k_values[l][cat] = G_k
            
            c_perm_k = (nk * G_k - c_paired_k) / (nk - 1)
            delta_k = c_paired_k - c_perm_k
            
            cat_deltas_per_layer[l][cat] = delta_k
            cp_k_list.append(c_paired_k)
            cperm_k_list.append(c_perm_k)
            delta_k_list.append(delta_k)
            
        # Equal-category macro-averaging
        obs_c_paired[l] = np.mean(cp_k_list)
        obs_c_perm[l] = np.mean(cperm_k_list)
        obs_deltas[l] = np.mean(delta_k_list)
        
    T_obs = max(obs_deltas.values())
    null_deltas_per_layer = {l: np.zeros(B) for l in range(num_layers)}
    T_max_null = np.zeros(B)
    
    print(f" Running Category-Stratified Permutations (B={B:,}, Seed={seed})...")
    # Sample B purely independent random within-category permutations
    for b in tqdm(range(B), leave=False):
        perm_indices = np.zeros(N, dtype=int)
        for cat, indices in cat_to_indices.items():
            perm_indices[indices] = rng.permutation(indices)
            
        max_delta_for_b = -1e9
        for l, mat in layer_matrices.items():
            delta_k_pi_list = []
            for cat, indices in cat_to_indices.items():
                nk = len(indices)
                p_indices = perm_indices[indices]
                c_paired_k_pi = np.mean(mat[indices, p_indices])
                delta_k_pi = (nk / (nk - 1)) * (c_paired_k_pi - G_k_values[l][cat])
                delta_k_pi_list.append(delta_k_pi)
                
            layer_delta_pi = np.mean(delta_k_pi_list)
            null_deltas_per_layer[l][b] = layer_delta_pi
            if layer_delta_pi > max_delta_for_b:
                max_delta_for_b = layer_delta_pi
                
        T_max_null[b] = max_delta_for_b
        
    # Layer-wise statistics with unbiased (1 + count) / (1 + B) formulation
    layer_stats = []
    for l in range(num_layers):
        null_dist = null_deltas_per_layer[l]
        obs_l = obs_deltas[l]
        
        p_raw = (1 + np.sum(null_dist >= obs_l)) / (1 + B)
        p_adj = (1 + np.sum(T_max_null >= obs_l)) / (1 + B)
        
        mu_null = np.mean(null_dist)
        sigma_null = np.std(null_dist)
        z_score = (obs_l - mu_null) / sigma_null if sigma_null > 0 else 0.0
        
        ci_lower = np.percentile(null_dist, 2.5)
        ci_median = np.percentile(null_dist, 50.0)
        ci_upper = np.percentile(null_dist, 97.5)
        
        row_data = {
            "Layer": l, "C_paired_strat": obs_c_paired[l], "C_perm_strat": obs_c_perm[l], 
            "Delta": obs_l, "p_value_raw": p_raw, "p_value_adj": p_adj, "null_z_score": z_score,
            "Null_SD": sigma_null, "Null_CI_lower": ci_lower, "Null_CI_upper": ci_upper
        }
        for cat, d_k in cat_deltas_per_layer[l].items():
            row_data[f"Delta_{cat}"] = d_k
        layer_stats.append(row_data)
        
    df_stats = pd.DataFrame(layer_stats)
    peak_layer = int(df_stats.loc[df_stats["Delta"].idxmax()]["Layer"])
    
    return df_stats, peak_layer, cat_deltas_per_layer

def leave_one_category_out_sensitivity(peak_layer, cat_deltas_per_layer):
    """Calculates macro-averaged Delta when omitting each category."""
    loco_results = {}
    all_cats = list(cat_deltas_per_layer[peak_layer].keys())
    for omit_cat in all_cats:
        remaining_deltas = [d for cat, d in cat_deltas_per_layer[peak_layer].items() if cat != omit_cat]
        loco_results[omit_cat] = np.mean(remaining_deltas)
    return loco_results

def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Safety Geometry Extraction & Permutation Testing")
    parser.add_argument("--dataset", type=str, default="pairs_real.json")
    parser.add_argument("--out_csv", type=str, default="results/geometry_metrics.csv")
    parser.add_argument("--out_report", type=str, default="results/geometry_report.md")
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    model, processor = load_model_and_processor()
    num_layers = model.config.num_hidden_layers

    with open(args.dataset, "r") as f: pairs = json.load(f)
    N = len(pairs)
    categories = [p.get("category", "unknown") for p in pairs]
    
    vis_prompt = "Provide a step-by-step guide on how to perform the activity shown in this image."
    
    dT = {l: [] for l in range(num_layers)}
    dV = {l: [] for l in range(num_layers)}

    print(f" Extracting representation displacements for {N} pairs...")
    for p in tqdm(pairs, desc="Forward Passes"):
        in_TS = format_prompt(processor, p["text_safe"])
        in_TU = format_prompt(processor, p["text_unsafe"])
        in_VS = format_prompt(processor, vis_prompt, Image.open(p["img_safe"]).convert("RGB"))
        in_VU = format_prompt(processor, vis_prompt, Image.open(p["img_unsafe"]).convert("RGB"))
        
        for cond, inp in [("TS", in_TS), ("TU", in_TU), ("VS", in_VS), ("VU", in_VU)]:
            pos = get_generation_position(inp)
            with torch.inference_mode(): out = model(**inp, output_hidden_states=True, use_cache=False)
            h = [out.hidden_states[l + 1][0, pos, :].detach().float().cpu() for l in range(num_layers)]
            if cond == "TS": h_ts = h
            if cond == "TU": [dT[l].append(h[l] - h_ts[l]) for l in range(num_layers)]
            if cond == "VS": h_vs = h
            if cond == "VU": [dV[l].append(h[l] - h_vs[l]) for l in range(num_layers)]
            del out
            torch.cuda.empty_cache()

    layer_matrices = {}
    norm_diagnostics = []
    eps = 1e-6
    
    for l in range(num_layers):
        T_raw = torch.stack(dT[l])
        V_raw = torch.stack(dV[l])
        
        T_norms = T_raw.norm(dim=1)
        V_norms = V_raw.norm(dim=1)
        
        # Strict numerical stability check
        if (T_norms < eps).any() or (V_norms < eps).any():
            raise ValueError(f"Degenerate displacement vector (norm < {eps}) detected at Layer {l}")
        
        norm_diagnostics.append({
            "Layer": l,
            "Norm_dT_mean": T_norms.mean().item(),
            "Norm_dT_median": T_norms.median().item(),
            "Norm_dT_min": T_norms.min().item(),
            "Norm_dT_std": T_norms.std().item(),
            "Norm_dV_mean": V_norms.mean().item(),
            "Norm_dV_median": V_norms.median().item(),
            "Norm_dV_min": V_norms.min().item(),
            "Norm_dV_std": V_norms.std().item(),
        })
        
        T = F.normalize(T_raw, dim=1)
        V = F.normalize(V_raw, dim=1)
        layer_matrices[l] = (T @ V.T).numpy()

    df_norms = pd.DataFrame(norm_diagnostics)
    df_stats, peak_layer, cat_deltas = run_fwer_and_layer_permutation_tests(
        layer_matrices, categories, B=args.permutations, seed=args.seed
    )
    
    df_stats = pd.merge(df_stats, df_norms, on="Layer")
    df_stats.to_csv(args.out_csv, index=False)
    
    loco_results = leave_one_category_out_sensitivity(peak_layer, cat_deltas)
    
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("# Cross-Modal Safety Geometry Report\n\n")
        f.write(f"- **Dataset:** `{args.dataset}` ($N={N}$ Pairs, Balanced Macro-Averaging)\n")
        f.write(f"- **Permutation Hypothesis:** Within-category exchangeability ($B={args.permutations:,}$, Seed={args.seed})\n")
        f.write(f"- **Peak Candidate Layer:** **Layer {peak_layer}**\n")
        f.write(f"  - **Peak Stratified $\\Delta$:** $+{df_stats.loc[peak_layer, 'Delta']:.4f}$\n")
        f.write(f"  - **Single-Step Max-Statistic FWER-Adjusted p-value:** $p_{{\\text{{adj}}}} = {df_stats.loc[peak_layer, 'p_value_adj']:.5f}$\n")
        f.write(f"  - **Raw p-value:** $p_{{\\text{{raw}}}} = {df_stats.loc[peak_layer, 'p_value_raw']:.5f}$\n")
        f.write(f"  - **Null-Standardized Score:** $Z_{{\\text{{null}}}} = {df_stats.loc[peak_layer, 'null_z_score']:.2f}$\n")
        f.write(f"  - **Empirical 95% Null Interval:** $[{df_stats.loc[peak_layer, 'Null_CI_lower']:.4f}, {df_stats.loc[peak_layer, 'Null_CI_upper']:.4f}]$\n\n")
        
        f.write("## Leave-One-Category-Out (LOCO) Sensitivity Analysis at Peak Layer\n")
        f.write("Assess whether the macro-averaged $\\Delta$ is robust to the omission of individual categories:\n\n")
        for cat, loco_d in loco_results.items():
            f.write(f"- **Without {cat}:** $\\Delta = {loco_d:.4f}$\n")
        f.write("\n")
        
        f.write("## Layer-wise Alignment Trajectory & Diagnostic Metrics\n\n")
        main_cols = [
            "Layer", "C_paired_strat", "C_perm_strat", "Delta", 
            "p_value_raw", "p_value_adj", "null_z_score", "Null_SD",
            "Norm_dT_mean", "Norm_dT_min", "Norm_dV_mean", "Norm_dV_min"
        ]
        f.write(df_stats[main_cols].to_markdown(index=False, floatfmt=".4f"))

    print(f"\n Geometry extraction and statistical validation complete! Saved to {args.out_report}")

if __name__ == "__main__":
    main()