# 04b_plot_ablation.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 15
})

def main():
    parser = argparse.ArgumentParser(description="Plot 1-Day Ablation Matrix")
    parser.add_argument("--input_csv", type=str, default="results/ablation_behavior.csv")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.input_csv):
        print(f" Error: Could not find {args.input_csv}. Please run the behavioral judge first.")
        return

    df = pd.read_csv(args.input_csv)
    
    # Handle column name variations smoothly
    score_col = 'LLM_Safety_Score' if 'LLM_Safety_Score' in df.columns else 'HarmBench_Score'
    
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df['D_JS'] = pd.to_numeric(df['D_JS'], errors='coerce')
    
    # Calculate Delta B (Behavioral Shift toward safety: Baseline - Steered)
    # Positive delta means it became SAFER.
    baselines = df[df['scope'] == 'none'].set_index('pair_id')[score_col].to_dict()
    df['Delta_B'] = df.apply(lambda row: baselines.get(row['pair_id'], 0) - row[score_col], axis=1)

    # Filter out baseline rows for the matrices
    df_interventions = df[df['scope'] != 'none'].copy()

    # Define strict ordering for rows and columns
    vector_order = [
        "Raw Text (dT)", 
        "Native Visual (dV)", 
        "Mapped Text (W*dT)", 
        "Random Null (dR)"
    ]
    scope_order = ["p_gen", "visual_tokens"]

    # =========================================================================
    # 1. FIGURE A: Latent Shift Matrix (D_JS)
    # =========================================================================
    pivot_js = df_interventions.pivot_table(index="condition", columns="scope", values="D_JS", aggfunc="mean")
    # Reorder
    pivot_js = pivot_js.reindex(index=vector_order, columns=scope_order)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot_js, annot=True, fmt=".4f", cmap="Purples", cbar_kws={'label': 'JS Divergence ($D_{JS}$)'})
    plt.title("Ablation Matrix: Latent Shift ($D_{JS}$)\n(Averaged across all layers & pairs)")
    plt.xlabel("Spatial Scope of Intervention")
    plt.ylabel("Representation Vector")
    plt.tight_layout()
    figA_path = os.path.join(args.out_dir, "figA_ablation_latent_matrix.png")
    plt.savefig(figA_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Generated: {figA_path}")

    # =========================================================================
    # 2. FIGURE B: Behavioral Shift Matrix (Delta B)
    # =========================================================================
    pivot_b = df_interventions.pivot_table(index="condition", columns="scope", values="Delta_B", aggfunc="mean")
    # Reorder
    pivot_b = pivot_b.reindex(index=vector_order, columns=scope_order)
    
    plt.figure(figsize=(7, 5))
    # Using a center=0 colormap to show shifts toward safety (green) vs compliance (red)
    sns.heatmap(pivot_b, annot=True, fmt=".3f", cmap="RdYlGn", center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={'label': 'Behavioral Shift ($\Delta B$)'})
    plt.title("Ablation Matrix: Behavioral Transfer ($\Delta B$)\n(Positive = Safer | Negative = More Harmful)")
    plt.xlabel("Spatial Scope of Intervention")
    plt.ylabel("Representation Vector")
    plt.tight_layout()
    figB_path = os.path.join(args.out_dir, "figB_ablation_behavior_matrix.png")
    plt.savefig(figB_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Generated: {figB_path}")

    # =========================================================================
    # 3. Markdown Summary Report
    # =========================================================================
    report_path = os.path.join(args.out_dir, "ablation_final_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Final Ablation Matrix Report (Scope vs Representation)\n\n")
        f.write(f"- **Test Set Size:** {df_interventions['pair_id'].nunique()} Pairs\n")
        f.write("- **Layers Aggregated:** All tested layers\n\n")
        
        f.write("## 1. Latent Shift Matrix ($D_{JS}$)\n")
        f.write("*Measures how much the intervention displaced the internal next-token probability distribution.* \n\n")
        f.write(pivot_js.to_markdown(floatfmt=".4f"))
        f.write("\n\n")
        
        f.write("## 2. Behavioral Transfer Matrix ($\Delta B$)\n")
        f.write("*Measures macroscopic semantic change based on LLM Judge score. $\Delta B = 0$ means the generated text did not meaningfully change from the baseline description. Negative values mean slight shifts toward harmful compliance.* \n\n")
        f.write(pivot_b.to_markdown(floatfmt=".4f"))
        f.write("\n")

    print(f" Generated Markdown tables: {report_path}")
    print("\n All ablation figures successfully generated!")

if __name__ == "__main__":
    main()
