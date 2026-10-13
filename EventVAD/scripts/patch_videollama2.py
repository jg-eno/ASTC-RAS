"""Apply the Transformers 4.40 SigLIP attention compatibility fix upstream needs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="VideoLLaMA2 checkout or package directory")
    root = parser.parse_args().source
    encoder = root / "videollama2" / "model" / "encoder.py"
    source = encoder.read_text()
    old_single = "config._attn_implementation = 'flash_attention_2'"
    old_double = 'config._attn_implementation = "flash_attention_2"'
    if (old_single not in source and old_double not in source
            and "config._attn_implementation = 'eager'" not in source
            and 'config._attn_implementation = "eager"' not in source):
        raise SystemExit(f"Expected upstream attention setting not found in {encoder}")
    source = source.replace(old_single, "config._attn_implementation = 'eager'")
    source = source.replace(old_double, "config._attn_implementation = \"eager\"")
    full_batch = """            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)"""
    chunked_batch = """            chunk_size = int(os.environ.get("VIDEOLLAMA2_VISION_BATCH", "1"))
            if images.shape[0] > chunk_size:
                image_features = []
                for image_chunk in images.split(chunk_size):
                    image_forward_outs = self.vision_tower(image_chunk, output_hidden_states=True)
                    image_features.append(self.feature_select(image_forward_outs).to(images.dtype))
                image_features = torch.cat(image_features, dim=0)
            else:
                image_forward_outs = self.vision_tower(images, output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs)"""
    if full_batch in source:
        source = source.replace(full_batch, chunked_batch)
    encoder.write_text(source)

    loader = root / "videollama2" / "model" / "__init__.py"
    source = loader.read_text()
    old_quant = """            bnb_4bit_quant_type='nf4'
        )"""
    new_quant = """            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=['vision_tower', 'mm_projector', 'lm_head']
        )"""
    if old_quant in source:
        source = source.replace(old_quant, new_quant, 1)
        loader.write_text(source)

    projector = root / "videollama2" / "model" / "projector.py"
    source = projector.read_text()
    old_cache_import = "from transformers import TRANSFORMERS_CACHE"
    new_cache_import = "from huggingface_hub.constants import HF_HUB_CACHE"
    if old_cache_import in source:
        source = source.replace(old_cache_import, new_cache_import, 1)
        source = source.replace("cache_dir = TRANSFORMERS_CACHE", "cache_dir = HF_HUB_CACHE", 1)
        projector.write_text(source)
    elif new_cache_import not in source:
        raise SystemExit(f"Expected upstream cache import not found in {projector}")
    print(f"Patched SigLIP/CLIP vision attention to eager: {encoder}")


if __name__ == "__main__":
    main()
