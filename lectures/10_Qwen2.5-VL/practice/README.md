# 10 Qwen2.5-VL — practice（最小推論デモ）

window attention ViT・動的解像度・絶対時間（長尺動画）を持つ Qwen2.5-VL の最小デモ。
大規模で高機能な MLLM の単体推論を、最小コードで体感できます。

## モデルサイズ選択肢（`--model`）

| variant | HF model id | 目安 VRAM(bf16) |
|---|---|---|
| **3B（既定）** | `Qwen/Qwen2.5-VL-3B-Instruct` | 約 8–10 GB |
| 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 約 18–20 GB |

初回実行時に HuggingFace Hub から重みが DL される。

## ローカルで素早く実行（uv）

```bash
cd lectures/10_Qwen2.5-VL/practice
uv run demo.py                                      # 既定: 3B + ../../assets/sample.jpg
uv run demo.py --prompt "画像内の物体をすべて挙げて"
uv run demo.py --model Qwen/Qwen2.5-VL-7B-Instruct  # 7B に切替
```

## Docker で実行（CLAUDE.md 準拠: コンテナ内 uv）

```bash
cd lectures/10_Qwen2.5-VL/practice
docker build -t qwen25vl-demo .

docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(cd ../../assets && pwd):/assets" \
  qwen25vl-demo --image /assets/sample.jpg --prompt "主な物体を列挙して"
```

- `--gpus all` にはホストの **nvidia-container-toolkit** が必要。

## 引数

| 引数 | 既定 | 説明 |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF model id |
| `--image` | `../../assets/sample.jpg` | 画像パス or URL |
| `--prompt` | 「主な物体を列挙」 | 質問文 |
| `--max-new-tokens` | 256 | 生成上限 |

## 補足

- `Qwen2_5_VLForConditionalGeneration` には **transformers>=4.49** が必要（pyproject で指定済み）。
- 仕組みの詳細は [`../materials/README.md`](../materials/README.md)。
