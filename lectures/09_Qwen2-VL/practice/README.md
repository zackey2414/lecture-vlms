# 04 Qwen2-VL — practice（最小推論デモ）

任意解像度を可変トークン化する **naive dynamic resolution** と **M-RoPE** を持つ Qwen2-VL の最小デモ。

## モデルサイズ選択肢（`--model`）

| variant | HF model id | 目安 VRAM(bf16) |
|---|---|---|
| **2B（既定）** | `Qwen/Qwen2-VL-2B-Instruct` | 約 6–8 GB |
| 7B | `Qwen/Qwen2-VL-7B-Instruct` | 約 18–20 GB |

初回実行時に HuggingFace Hub から重みが DL される。

## ローカルで素早く実行（uv）

```bash
cd lectures/09_Qwen2-VL/practice
uv run demo.py                                    # 既定: 2B + ../../assets/sample.jpg
uv run demo.py --prompt "画像内の物体をすべて挙げて"
uv run demo.py --model Qwen/Qwen2-VL-7B-Instruct  # 7B に切替
uv run demo.py --image https://example.com/x.jpg  # URL も可
```

## Docker で実行（CLAUDE.md 準拠: コンテナ内 uv）

```bash
cd lectures/09_Qwen2-VL/practice
docker build -t qwen2vl-demo .

docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(cd ../../assets && pwd):/assets" \
  qwen2vl-demo --image /assets/sample.jpg --prompt "主な物体を列挙して"
```

- `--gpus all` にはホストの **nvidia-container-toolkit** が必要。

## 引数

| 引数 | 既定 | 説明 |
|---|---|---|
| `--model` | `Qwen/Qwen2-VL-2B-Instruct` | HF model id |
| `--image` | `../../assets/sample.jpg` | 画像パス or URL |
| `--prompt` | 「主な物体を列挙」 | 質問文 |
| `--max-new-tokens` | 256 | 生成上限 |

## 補足

- `qwen-vl-utils` が画像/動画入力の前処理（`process_vision_info`）を担う。動画を渡す場合は
  messages の content に `{"type": "video", "video": ...}` を追加する（本デモは画像のみ）。
- 仕組みの詳細は [`../materials/README.md`](../materials/README.md)。
