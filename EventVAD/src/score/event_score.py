import os
import csv
import argparse
import re
import traceback
os.environ.setdefault("PYTORCH_NO_CUDA_NVML", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

NUM_FRAMES = 16
BATCH_SIZE = 4

def init_model(model_path, quantization="4bit"):
    disable_torch_init()
    from transformers.models.siglip.modeling_siglip import SiglipAttention

    # SigLIP 4.40 uses an eager fp32 attention matrix; SDPA computes the same
    # attention with much lower peak memory, which matters for 16 video frames.
    def siglip_sdpa_forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch, length, _ = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.out_proj(output), None

    SiglipAttention.forward = siglip_sdpa_forward
    if quantization == "4bit":
        load_args = {"device": "cuda:0", "load_4bit": True}
    elif quantization == "none":
        load_args = {"device": "cuda:0"}
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    model, processor, tokenizer = model_init(model_path, **load_args)
    return model.eval(), processor, tokenizer

def parse_score(output):
    """Accept a bare scalar or a short answer containing one 0-1 score."""
    numbers = re.findall(r"(?<![\w.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\w.])", output)
    if not numbers:
        raise ValueError(f"Model did not return a score in [0, 1]: {output!r}")
    return float(numbers[-1])

def process_files(video_files, output_csv, model_path, quantization="4bit"):
    model, processor, tokenizer = init_model(model_path, quantization)

    abnormal_prompt = (
        "Assess whether this surveillance video segment contains anomalous or criminal activity. "
        "Return only a single anomaly score from 0 (normal) to 1 (clearly anomalous)."
    )
    
    details_csv = f"{output_csv}.details.csv"
    with open(details_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'score', 'status', 'response'])
        
        for path_str in tqdm(video_files, desc="Processing"):
            path = Path(path_str)
            if not path.exists():
                writer.writerow([str(path), -1, "Error: File not found", ""])
                continue
            try:
                video_input = processor['video'](str(path))
                if torch.cuda.is_available():
                    video_input = video_input.to("cuda:0")
                
                with torch.inference_mode():
                    output = mm_infer(
                        video_input,
                        abnormal_prompt,
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=False,
                        max_new_tokens=32,
                        modal='video'
                    )
                
                try:
                    score = parse_score(output)
                    status = "success"
                except ValueError as e:
                    score = -1
                    status = f"Error: {e}"
                
                writer.writerow([str(path), score, status, output])
            except Exception as e:
                writer.writerow([str(path), -1, f"Error: {e}\n{traceback.format_exc()}", ""])
            finally:
                # empty_cache() can assert in CUDA MIG environments; normal
                # tensor lifetime management releases per-clip allocations.
                pass

def main(input_csv, output_csv, model_path, quantization="4bit"):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Manifest文件不存在: {input_csv}")
    
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    original_entries = []
    video_files = []
    with open(input_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                path = parts[0]
                start = parts[1]
                end = parts[2]
                video_files.append(path)
                original_entries.append({
                    'path': path,
                    'start': start,
                    'end': end,
                    'score': None
                })
    
    process_files(video_files, output_csv, model_path, quantization)
    
    scores_dict = {}
    with open(f"{output_csv}.details.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[0]
            score = row[1]
            scores_dict[path] = float(score) if row[2] == "success" else -1

    with open(output_csv, 'w') as f:
        for entry in original_entries:
            line = f"{entry['path']} {entry['start']} {entry['end']} {scores_dict.get(entry['path'], -1)}\n"
            f.write(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model", default=os.environ.get(
        "EVENTVAD_MODEL", "models/VideoLLaMA2.1-7B-16F"))
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit",
                        help="Runtime quantization for the frozen checkpoint.")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.model,
         quantization=args.quantization)



