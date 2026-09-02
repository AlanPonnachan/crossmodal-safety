import os
import json
import shutil
import argparse
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

VALID_CATEGORIES = [
    "01-Illegal_Activity", "02-HateSpeech", "03-Malware_Generation",
    "04-Physical_Harm", "06-Fraud", "07-Sex"
]

def fetch_category_jsons(repo_id, category_name):
    unsafe_path = f"processed_questions/{category_name}.json"
    safe_path = f"processed_questions_safe/{category_name}.json"
    try:
        u_local = hf_hub_download(repo_id=repo_id, filename=unsafe_path, repo_type="dataset")
        s_local = hf_hub_download(repo_id=repo_id, filename=safe_path, repo_type="dataset")
        with open(u_local, "r", encoding="utf-8") as f:
            u_data = json.load(f)
        with open(s_local, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        return u_data, s_data
    except Exception as e:
        print(f"  Could not fetch JSONs for {category_name}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Build a balanced safety dataset.")
    parser.add_argument("--num_pairs", type=int, default=64, help="Exact total pairs to extract")
    parser.add_argument("--out_file", type=str, default="pairs_real.json")
    parser.add_argument("--image_dir", type=str, default="images_real")
    parser.add_argument("--mode", type=str, default="Gen", choices=["Gen", "GenOCR"])
    args = parser.parse_args()

    repo_id = "EchoSafe-MLLM/MM-SafetyBench-plus-plus"
    
    if os.path.exists(args.image_dir):
        shutil.rmtree(args.image_dir)
    os.makedirs(args.image_dir, exist_ok=True)
    
    print(f" Loading dataset from Hugging Face ({repo_id})...")
    hf_ds = load_dataset(repo_id, split="test")
    
    dataset = []
    
    # Exact Distribution Math (Handles remainders so you get exactly args.num_pairs)
    num_cats = len(VALID_CATEGORIES)
    base_per_cat = args.num_pairs // num_cats
    remainder = args.num_pairs % num_cats
    
    total_extracted = 0

    for i, cat in enumerate(VALID_CATEGORIES):
        target_for_this_cat = base_per_cat + (1 if i < remainder else 0)
        if target_for_this_cat == 0:
            continue
            
        print(f" Processing Category: {cat} (Target: {target_for_this_cat} pairs)...")
        u_json, s_json = fetch_category_jsons(repo_id, cat)
        if u_json is None or s_json is None:
            continue
            
        clean_cat = cat.split("-")[-1]
        
        cat_rows = [row for row in hf_ds if row['category'].lower() == clean_cat.lower() and row.get('mode', '') == args.mode]
        unsafe_rows = [r for r in cat_rows if r['label'].lower() == 'unsafe']
        safe_rows = [r for r in cat_rows if r['label'].lower() == 'safe']
        
        matched_keys = sorted([k for k in u_json.keys() if k in s_json], key=lambda x: int(x))
        
        cat_extracted = 0
        for idx_str in matched_keys:
            if cat_extracted >= target_for_this_cat or total_extracted >= args.num_pairs:
                break
                
            idx = int(idx_str)
            if idx >= len(unsafe_rows) or idx >= len(safe_rows):
                continue
                
            text_u = u_json[idx_str].get("Changed Question")
            text_s = s_json[idx_str].get("Changed Question")
            
            if not text_u or not text_s:
                continue
                
            pair_id = f"pair_{total_extracted + 1}_{clean_cat.lower()}"
            img_safe_path = os.path.join(args.image_dir, f"{pair_id}_safe.jpg")
            img_unsafe_path = os.path.join(args.image_dir, f"{pair_id}_unsafe.jpg")
            
            try:
                safe_rows[idx]['image'].convert("RGB").save(img_safe_path)
                unsafe_rows[idx]['image'].convert("RGB").save(img_unsafe_path)
            except Exception as e:
                continue
                
            dataset.append({
                "id": pair_id, "category": clean_cat,
                "text_safe": text_s, "text_unsafe": text_u,
                "img_safe": img_safe_path, "img_unsafe": img_unsafe_path
            })
            cat_extracted += 1
            total_extracted += 1

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n dataset build complete! Successfully saved exactly {len(dataset)} balanced pairs.")

if __name__ == "__main__":
    main()