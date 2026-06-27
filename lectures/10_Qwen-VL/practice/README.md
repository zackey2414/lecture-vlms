# 10 Qwen-VL — practice（最小推論デモ）

初代 Qwen-VL（ViT → Position-aware VL Adapter → LLM）を `model.chat()` で動かす最小デモ。

## ⚠ レガシー custom code につき注意

- `Qwen/Qwen-VL-Chat` は **`trust_remote_code=True` 必須**の独自実装で、`tiktoken` /
  `einops` / `transformers-stream-generator` 等に依存し、**transformers のバージョン整合がシビア**。
- 本 practice は `transformers==4.37.2` を目安に固定（動作実績のある組み合わせ）。新しすぎる
  transformers では custom code が壊れることがある。
- うまく動かない場合は、**確実に動く Qwen デモである [11 Qwen2-VL](../../11_Qwen2-VL/) /
  [12 Qwen2.5-VL](../../12_Qwen2.5-VL/) を先に試す**ことを推奨。この章は主に**論文（materials）で
  3 段階学習・grounding・OCR の設計を学ぶ**ことに重点を置く。

## モデル（`--model`）

| variant | HF model id | 目安 VRAM(fp16) |
|---|---|---|
| 既定 | `Qwen/Qwen-VL-Chat` | 約 18–20 GB |

## ローカルで実行（uv）

```bash
cd lectures/10_Qwen-VL/practice
uv run demo.py
uv run demo.py --prompt "画像中の物体の位置を説明して"
```

## Docker で実行

```bash
cd lectures/10_Qwen-VL/practice
docker build -t qwenvl-demo .
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(cd ../../assets && pwd):/assets" \
  qwenvl-demo --image /assets/sample.jpg
```

## 引数

| 引数 | 既定 | 説明 |
|---|---|---|
| `--model` | `Qwen/Qwen-VL-Chat` | HF model id |
| `--image` | `../../assets/sample.jpg` | 画像パス or URL |
| `--prompt` | 「主な物体を列挙」 | 質問文 |

## トラブルシュート

- `ImportError` / API 不整合 → transformers のバージョンを 4.37.2 前後で調整。
- `tiktoken` 関連エラー → `tiktoken` が入っているか確認（pyproject に記載済み）。
- それでも不安定なら 04 / 05 で Qwen の挙動を確認し、本章は論文読解に充てる。
- 仕組みの詳細は [`../materials/README.md`](../materials/README.md)。
