# 04_plot_results.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 15
})

CLEAN_CONDITION_NAMES = {
    "Baseline_VU": "Baseline (Unsteered VU)",
    "Baseline_VS": "Baseline (Unsteered VS)",
    "Positive": "Positive Control (dV)",
    "Primary": "LOO Mean (Primary)",
    "Robustness": "LOO PCA (Robustness)",
    "Specificity": "Wrong Vector (Specificity)",
    "Null": "Random Vector (Null)"
}

def clean_label(cond_str):
    for key, clean_name in CLEAN_CONDITION_NAMES.items():
        if f"({key})" in cond_str or cond_str == key:
            return clean_name
    return cond_str

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures dynamically.")
    parser.add_argument("--geom_csv", type=str, default="results/geometry_metrics.csv")
    parser.add_argument("--judge_csv", type=str, default="results/behavioral_results.csv")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # =========================================================================
    # 1. FIGURE 1: Geometric Alignment Trajectory
    # =========================================================================
    if os.path.exists(args.geom_csv):
        df_geom = pd.read_csv(args.geom_csv)
        plt.figure(figsize=(8.5, 4.5))
        plt.plot(df_geom['Layer'], df_geom['C_paired_strat'], label=r'Paired Concept Alignment ($C_{paired}$)', 
                 color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
        plt.plot(df_geom['Layer'], df_geom['C_perm_strat'], label=r'Category-Matched Null ($C_{perm}$)', 
                 color='#7f7f7f', linewidth=2, linestyle='--')
        
        plt.axvspan(24, 27, color='#2ca02c', alpha=0.15, label='Peak Decoupling Region (L24–L27)')
        plt.title("Cross-Modal Geometric Alignment Trajectory")
        plt.xlabel("Transformer Decoder Layer Index")
        plt.ylabel("Cosine Similarity")
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        fig1_path = os.path.join(args.out_dir, "fig1_geometric_alignment.png")
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Generated: {fig1_path}")

    # =========================================================================
    # Load and Clean Behavioral Data
    # =========================================================================
    if not os.path.exists(args.judge_csv):
        print(f" Error: Could not find {args.judge_csv}.")
        return

    df_judge = pd.read_csv(args.judge_csv)
    N_pairs = df_judge['pair_id'].nunique()
    
    df_judge['LLM_Safety_Score'] = pd.to_numeric(df_judge['LLM_Safety_Score'], errors='coerce')
    df_judge['D_JS'] = pd.to_numeric(df_judge['D_JS'], errors='coerce')
    df_judge['alpha'] = pd.to_numeric(df_judge['alpha'], errors='coerce')
    
    df_judge['clean_condition'] = df_judge['condition'].apply(clean_label)
    
    # Delta B = Baseline_VU Score - Intervention Score (Positive = Shift Toward Safety)
    base_scores = df_judge[df_judge['condition'] == 'Baseline_VU'].set_index('pair_id')['LLM_Safety_Score'].to_dict()
    df_judge['Delta_B'] = df_judge.apply(lambda row: base_scores.get(row['pair_id'], 0) - row['LLM_Safety_Score'], axis=1)

    intervention_conditions = [c for c in df_judge['condition'].unique() if "VU - a" in c]
    df_interventions = df_judge[df_judge['condition'].isin(intervention_conditions)].copy()
    df_alpha1 = df_interventions[df_interventions['alpha'] == 1.0].copy()

    if not df_alpha1.empty:
        # =====================================================================
        # 2. FIGURE 2: Latent Shift Heatmap (JS Divergence at Alpha = 1.0)
        # =====================================================================
        pivot_JS = df_alpha1.pivot_table(index='clean_condition', columns='layer', values='D_JS', aggfunc='mean')
        
        plt.figure(figsize=(9.5, 4.5))
        sns.heatmap(pivot_JS, annot=True, fmt=".3f", cmap="Purples", cbar_kws={'label': 'JS Divergence ($D_{JS}$)'})
        plt.title(f"Latent Distribution Shift ($D_{{JS}}$) at $\\alpha=1.0$ (N={N_pairs})")
        plt.xlabel("Transformer Decoder Layer Index")
        plt.ylabel("Intervention Condition")
        plt.tight_layout()
        fig2_path = os.path.join(args.out_dir, "fig2_latent_js_shift.png")
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Generated: {fig2_path}")

        # =====================================================================
        # 3. FIGURE 3: Behavioral Transfer Bar Chart (Delta B)
        # =====================================================================
        behav_pivot = df_alpha1.groupby('clean_condition')['Delta_B'].mean().sort_values()
        
        plt.figure(figsize=(9, 4.5))
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in behav_pivot.values]
        bars = plt.barh(behav_pivot.index, behav_pivot.values, color=colors, alpha=0.85, edgecolor='black')
        
        min_x = min(behav_pivot.values.min(), -0.05)
        max_x = max(behav_pivot.values.max(), 0.05)
        span = max_x - min_x
        plt.xlim(min_x - span * 0.25, max_x + span * 0.25)
        
        for bar in bars:
            val = bar.get_width()
            if val < 0:
                plt.text(val - (span * 0.02), bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                         ha='right', va='center', fontweight='bold', color='#b2182b')
            else:
                plt.text(val + (span * 0.02), bar.get_y() + bar.get_height()/2, f'+{val:.2f}', 
                         ha='left', va='center', fontweight='bold', color='#1b7837')

        plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        plt.title(f"Behavioral Transfer ($\Delta B$) at $\\alpha=1.0$ (N={N_pairs})\n(Positive = Shift Toward Safety | Negative = Shift Away from Safety)")
        plt.xlabel("Mean Shift in Behavioral Safety Score ($\Delta B$)")
        plt.ylabel("")
        plt.tight_layout()
        fig3_path = os.path.join(args.out_dir, "fig3_behavioral_transfer.png")
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Generated: {fig3_path}")

        # =====================================================================
        # 4. FIGURE 4: Dose-Response Line Plot (Safety Score vs Alpha)
        # =====================================================================
        df_alphas = df_interventions.groupby(['alpha', 'clean_condition'])['LLM_Safety_Score'].mean().reset_index()
        
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=df_alphas, x='alpha', y='LLM_Safety_Score', hue='clean_condition', 
                     marker='o', markersize=7, linewidth=2.5)
        
        plt.title(f"Dose-Response Trajectory across Steering Multipliers (N={N_pairs})")
        plt.xlabel(r"Intervention Norm Multiplier ($\alpha$)")
        plt.ylabel("Behavioral Safety Score (1 = Safe, 5 = Harmful)")
        plt.ylim(1.0, 5.0)
        plt.axhline(2.0, color='gray', linestyle=':', label='Descriptive Baseline Region (~2.0)')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        fig4_path = os.path.join(args.out_dir, "fig4_dose_response.png")
        plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Generated: {fig4_path}")

        # =====================================================================
        # 5. Markdown Report
        # =====================================================================
        report_path = os.path.join(args.out_dir, "final_causal_report.md")
        with open(report_path, "w") as f:
            f.write(f"# Final Causal Transfer Report (N={N_pairs} Pairs)\n\n")
            f.write("## Behavioral Shift ($\Delta B$) vs. Latent Shift ($D_{JS}$) at Alpha = 1.0\n")
            f.write("*(Note: $\Delta B = 0$ indicates no macroscopic behavioral change relative to the unsteered baseline; scores evaluate policy compliance on a 1–5 scale).* \n\n")
            summary_df = df_alpha1.groupby('clean_condition')[['D_JS', 'Delta_B']].mean().reset_index()
            f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n")
        print(f" Generated: {report_path}")

    print("\n All publication figures successfully generated and saved to results/ directory!")

if __name__ == "__main__":
    main()