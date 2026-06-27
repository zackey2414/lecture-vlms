"""Qwen-VL(-Chat) 最小推論デモ（HF transformers, レガシー custom code）。

画像1枚とプロンプトを与え、`model.chat()` で言語生成を得る。
仕組み: ViT → Position-aware VL Adapter（cross-attn で 256 クエリへ圧縮）→ LLM。

注意: Qwen-VL-Chat は `trust_remote_code=True` 必須の **レガシー custom code** 系で、
transformers のバージョン整合がシビア（README 参照）。確実に動く Qwen デモは
[11 Qwen2-VL](../../11_Qwen2-VL/) / [12 Qwen2.5-VL](../../12_Qwen2.5-VL/) を推奨。
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "sample.jpg"
DEFAULT_MODEL = "Qwen/Qwen-VL-Chat"


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-VL-Chat minimal inference demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="入力画像のパス or URL")
    parser.add_argument(
        "--prompt", default="この画像には何が写っていますか？主な物体を列挙してください。"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    ).eval()

    # Qwen-VL は画像をパス/URL のトークンとして埋め込む独自フォーマットを使う
    query = tokenizer.from_list_format(
        [{"image": args.image}, {"text": args.prompt}]
    )
    answer, _history = model.chat(tokenizer, query=query, history=None)

    print(f"\n[model]  {args.model}")
    print(f"[image]  {args.image}")
    print(f"[prompt] {args.prompt}")
    print(f"[answer] {answer.strip()}")


if __name__ == "__main__":
    main()
