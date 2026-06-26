# 06 LLaVA-1.5 — practice（最小推論デモ）

CLIP 視覚特徴 →（MLP projector）→ LLM(Vicuna) という LLaVA-1.5 を、画像1枚で動かす最小デモ。

## モデルサイズ選択肢（`--model`）

| variant | HF model id | 目安 VRAM(fp16) |
|---|---|---|
| **7B（既定）** | `llava-hf/llava-1.5-7b-hf` | 約 14–16 GB |
| 13B | `llava-hf/llava-1.5-13b-hf` | 約 26–30 GB |

初回実行時に HuggingFace Hub から重みが DL される（7B で約 14GB）。

## ローカルで素早く実行（uv）

```bash
cd lectures/06_LLaVA-1.5/practice
uv run demo.py                                   # 既定: 7B + ../../assets/sample.jpg
uv run demo.py --prompt "テーブルの上にある果物は何ですか？"
uv run demo.py --model llava-hf/llava-1.5-13b-hf # 13B に切替
uv run demo.py --image /path/to/your.jpg
```

## Docker で実行（CLAUDE.md 準拠: コンテナ内 uv）

```bash
cd lectures/06_LLaVA-1.5/practice
docker build -t llava15-demo .

docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(cd ../../assets && pwd):/assets" \
  llava15-demo --image /assets/sample.jpg --prompt "主な物体を列挙して"
```

- `--gpus all` にはホストの **nvidia-container-toolkit** が必要。
- HF キャッシュをマウントするとモデル重みの再 DL を避けられる。

## 引数

| 引数 | 既定 | 説明 |
|---|---|---|
| `--model` | `llava-hf/llava-1.5-7b-hf` | HF model id |
| `--image` | `../../assets/sample.jpg` | 画像パス or URL |
| `--prompt` | 「主な物体を列挙」 | 質問文 |
| `--max-new-tokens` | 256 | 生成上限 |

## 補足

- torch は Linux 既定の PyPI wheel が CUDA(cu12x) ビルド。万一 CPU 版が入る場合は
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` で差し替える。
- 仕組みの詳細は [`../materials/README.md`](../materials/README.md)。
