"""Qwen2.5-VL 最小推論デモ（HF transformers + qwen-vl-utils）。

画像1枚とプロンプトを与え、`Qwen2_5_VLForConditionalGeneration` で言語生成を得る。
ポイント: window attention ViT・動的解像度・絶対時間エンコーディング（長尺動画）。
"""

import argparse
from pathlib import Path

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "sample.jpg"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # サイズ選択肢は README 参照


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL minimal inference demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id（README のサイズ選択肢参照）")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="入力画像のパス or URL")
    parser.add_argument(
        "--prompt", default="この画像には何が写っていますか？主な物体を列挙してください。"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"\n[model]  {args.model}")
    print(f"[image]  {args.image}")
    print(f"[prompt] {args.prompt}")
    print(f"[answer] {answer}")


if __name__ == "__main__":
    main()
