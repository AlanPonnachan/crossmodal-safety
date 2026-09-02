import os
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    local_path = os.path.abspath(f"scratch/models/{model_path.split('/')[-1]}")
    path = local_path if os.path.exists(local_path) else model_path
    print(f"Loading model from: {path}")
    processor = AutoProcessor.from_pretrained(path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, processor

def format_prompt(processor, text_content, image_obj=None):
    if image_obj is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": text_content}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_content}]}]
        
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_obj is None:
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], images=[image_obj], padding=True, return_tensors="pt")
    return inputs.to("cuda")

def get_generation_position(inputs):
    return inputs.input_ids.shape[1] - 1

def compute_refusal_score(logits, processor):
    """
    Implements Aggregated Next-Token Refusal Probability (Arditi et al., 2024).
    Sums probability mass over canonical refusal initiation tokens for Qwen.
    """
    probs = torch.softmax(logits, dim=-1)
    refusal_keywords = ["I", "Sorry", "As", "Unfortunately", "My", "Please", "Cannot", "I'm", "unable"]
    
    refusal_token_ids = []
    for s in refusal_keywords:
        ids = processor.tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            refusal_token_ids.append(ids[0])
        space_ids = processor.tokenizer(" " + s, add_special_tokens=False).input_ids
        if space_ids:
            refusal_token_ids.append(space_ids[0])
            
    refusal_token_ids = list(set(refusal_token_ids))
    return sum(probs[tok_id].item() for tok_id in refusal_token_ids)

def compute_js_divergence(logits_p, logits_q):
    """Computes symmetric Jensen-Shannon Divergence stably in log-space."""
    log_p = F.log_softmax(logits_p, dim=-1).to(torch.float64)
    log_q = F.log_softmax(logits_q, dim=-1).to(torch.float64)
    
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    log_m = torch.log(m)
    
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).item()

def print_top_tokens(logits, processor, k=5):
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    tokens = []
    for p, idx in zip(top_probs, top_indices):
        tok_str = processor.tokenizer.decode([idx.item()]).replace('\n', '\\n')
        tokens.append(f"'{tok_str}': {p.item():.3f}")
    return " | ".join(tokens)