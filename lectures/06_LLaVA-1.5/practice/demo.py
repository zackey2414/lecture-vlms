"""LLaVA-1.5 最小推論デモ（HF transformers）。

画像1枚とプロンプトを与え、`LlavaForConditionalGeneration` で言語生成を得る。
仕組み: CLIP 視覚特徴 → MLP projector → LLM(Vicuna) の埋め込み空間 → 生成。
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "sample.jpg"
DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"  # サイズ選択肢は README 参照


def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        import requests

        return Image.open(requests.get(src, stream=True, timeout=30).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaVA-1.5 minimal inference demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id（README のサイズ選択肢参照）")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="入力画像のパス or URL")
    parser.add_argument(
        "--prompt", default="この画像には何が写っていますか？主な物体を列挙してください。"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    image = load_image(args.image)

    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )

    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device, torch.float16)

    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\n[model]  {args.model}")
    print(f"[image]  {args.image}")
    print(f"[prompt] {args.prompt}")
    print(f"[answer] {answer}")


if __name__ == "__main__":
    main()
