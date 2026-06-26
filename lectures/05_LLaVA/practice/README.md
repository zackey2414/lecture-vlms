# 05 LLaVA — practice（最小推論デモ）

LLaVA の原型「CLIP 視覚特徴 →（**線形** projector）→ LLM(Vicuna) → 生成」を動かす最小デモ。

## ⚠ 原典 LLaVA の重みについて

- 原典 LLaVA(v0/v1) の公開重みは **Vicuna への delta** 形式で配布されており、HF transformers の
  `from_pretrained` で**直接ロードできる公式 checkpoint がない**。
- そこで本デモは、アーキテクチャ（CLIP+projector+LLM）が共通な `llava-hf` 系 checkpoint で
  *仕組み* を動かす。**原典固有のレシピ（線形 projector・学習データ・解像度）は
  [`../materials/README.md`](../materials/README.md) と論文で学ぶ**前提。
- 原典の重みを実際に使う場合は、[LLaVA 公式リポジトリ](https://github.com/haotian-liu/LLaVA) の手順で
  base(Vicuna) ＋ delta を merge する必要がある（本デモのスコープ外）。
- 「線形 → MLP projector・データ強化・解像度 336」という**差分**は次章 [02 LLaVA-1.5](../../06_LLaVA-1.5/) で確認。

## モデル（`--model`）

| variant | HF model id | 目安 VRAM(fp16) | 備考 |
|---|---|---|---|
| 既定 | `llava-hf/llava-1.5-7b-hf` | 約 14–16 GB | 共通アーキで動かす代替 |
| （原典） | delta merge が必要 | — | 公式 HF checkpoint なし |

## ローカルで実行（uv）

```bash
cd lectures/05_LLaVA/practice
uv run demo.py
uv run demo.py --prompt "画像の状況を説明して"
```

## Docker で実行

```bash
cd lectures/05_LLaVA/practice
docker build -t llava-demo .
docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(cd ../../assets && pwd):/assets" \
  llava-demo --image /assets/sample.jpg
```

- `--gpus all` にはホストの **nvidia-container-toolkit** が必要。
- torch が CPU 版になった場合は `--index-url https://download.pytorch.org/whl/cu124` で差し替え。
