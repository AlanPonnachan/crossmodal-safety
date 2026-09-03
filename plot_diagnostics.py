import os
import json
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import load_model_and_processor, format_prompt, get_generation_position

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.titlesize": 15
})

def main():
    parser = argparse.ArgumentParser(description="Generate Geometric Diagnostic Plots (Heatmap & PCA)")
    parser.add_argument("--dataset", type=str, default="pairs_real.json")
    parser.add_argument("--layer", type=int, default=27, help="Peak layer to analyze")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.dataset, "r") as f:
        pairs = json.load(f)

    # Sort pairs by category so the heatmap shows distinct blocks
    pairs = sorted(pairs, key=lambda x: x.get("category", ""))
    N = len(pairs)
    categories = [p.get("category", "unknown") for p in pairs]

    print(f" Loading model to extract Layer {args.layer} vectors for {N} pairs...")
    model, processor = load_model_and_processor()
    vis_prompt = "Provide a step-by-step guide on how to perform the activity shown in this image."
    
    dT_list = []
    dV_list = []

    for p in tqdm(pairs, desc="Extracting Vectors"):
        in_TS = format_prompt(processor, p["text_safe"])
        in_TU = format_prompt(processor, p["text_unsafe"])
        in_VS = format_prompt(processor, vis_prompt, Image.open(p["img_safe"]).convert("RGB"))
        in_VU = format_prompt(processor, vis_prompt, Image.open(p["img_unsafe"]).convert("RGB"))
        
        hiddens = {}
        for cond, inp in [("TS", in_TS), ("TU", in_TU), ("VS", in_VS), ("VU", in_VU)]:
            pos = get_generation_position(inp)
            with torch.inference_mode():
                out = model(**inp, output_hidden_states=True, use_cache=False)
            hiddens[cond] = out.hidden_states[args.layer + 1][0, pos, :].detach().float().cpu()
            del out
            torch.cuda.empty_cache()
            
        dT_list.append(hiddens["TU"] - hiddens["TS"])
        dV_list.append(hiddens["VU"] - hiddens["VS"])

    # Convert to normalized tensors
    T_raw = torch.stack(dT_list)
    V_raw = torch.stack(dV_list)
    T = F.normalize(T_raw, dim=1).numpy()
    V = F.normalize(V_raw, dim=1).numpy()

    print(" Generating Visualizations...")

    # =========================================================================
    # 1. FIGURE 5: N x N Cross-Modal Cosine Matrix Heatmap
    # =========================================================================
    cosine_matrix = T @ V.T

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cosine_matrix, cmap="viridis", vmin=-0.1, vmax=0.3, 
                     cbar_kws={'label': 'Cosine Similarity'})
    
    # Calculate ordered counts per category in pure Python
    unique_ordered_cats = list(dict.fromkeys(categories))
    cat_counts = [categories.count(cat) for cat in unique_ordered_cats]
    
    current_idx = 0
    for count in cat_counts:
        # Draw a rectangle around each category block
        ax.add_patch(plt.Rectangle((current_idx, current_idx), count, count, 
                                   fill=False, edgecolor='red', lw=2, linestyle='--'))
        current_idx += count

    plt.title(f"Cross-Modal Safety Geometry (Layer {args.layer})\nRed boxes = Within-Category Similarity Blocks")
    plt.xlabel(r"Visual Safety Vectors ($d_V$)")
    plt.ylabel(r"Text Safety Vectors ($d_T$)")
    
    ax.set_xticks([])
    ax.set_yticks([])

    fig5_path = os.path.join(args.out_dir, "fig5_cosine_matrix_blocks.png")
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f" Generated: {fig5_path}")

    # =========================================================================
    # 2. FIGURE 6: PCA Projection (The Modality Gap)
    # =========================================================================
    all_vectors = np.vstack([T_raw.numpy(), V_raw.numpy()])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_vectors)

    df_pca = pd.DataFrame({
        "PC1": pca_result[:, 0],
        "PC2": pca_result[:, 1],
        "Modality": ["Text"] * N + ["Vision"] * N,
        "Category": categories + categories
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", 
        hue="Category", style="Modality", 
        markers={"Text": "o", "Vision": "X"},
        s=100, alpha=0.8, edgecolor="k"
    )

    plt.title(f"PCA Projection of Safety Displacement Vectors (Layer {args.layer})\nDemonstrating the 'Modality Gap'")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    fig6_path = os.path.join(args.out_dir, "fig6_pca_modality_gap.png")
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Generated: {fig6_path}")

    print("\n Diagnostics complete! Figures saved to results/ directory.")

if __name__ == "__main__":
    main()