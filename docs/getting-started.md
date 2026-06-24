# はじめ方

この教材は、LLM をバックボーンに持つ **VLM（論文表現では MLLM: 画像＋言語を入力し、LLM により言語生成を得るモデル）** を、論文から構造・仕組みまで理解するためのものです。各回は次の2層で構成されます。

- **教科書ページ（このサイト）** — 丁寧な地の文と SVG 図で、論文の仕組みを解説します。
- **最小推論デモ（`practice/demo.py`）** — HF transformers の `from_pretrained` で実際にモデルを動かします。

## 読む順番

番号順（`01_LLaVA` → `05_Qwen2.5-VL`）が基本ですが、各回には「前提」があります。[学習順序グラフ](graph.html) で依存関係（DAG）を確認できます。

- **パラダイム原点**: `01_LLaVA` / `02_LLaVA-1.5` — LLM＋projector＋visual instruction tuning の雛形。
- **Qwen系**: `03_Qwen-VL` / `04_Qwen2-VL` / `05_Qwen2.5-VL` — 3段階学習・動的解像度・M-RoPE・長尺動画へと縦に深掘り。

## 読むときの5観点（比較メモのテンプレ）

各回を同じ観点で比較すると、モデルの差分が立体的に見えます。

1. **コネクタ設計** — Q-Former / MLP projector / cross-attention / native fusion
2. **解像度処理** — 固定 / 動的解像度 / タイリング
3. **学習段階** — pretraining → SFT → preference/RL の構成
4. **動画対応** — フレーム処理・時間符号化（M-RoPE・絶対時間など）
5. **後処理アライメント** — DPO / MPO / GRPO 等

### 横断比較表

| 観点 | LLaVA | LLaVA-1.5 | Qwen-VL | Qwen2-VL | Qwen2.5-VL |
|---|---|---|---|---|---|
| コネクタ | 線形projector | MLP projector | Position-aware Adapter (cross-attn) | MLP (動的解像度) | MLP (window attn ViT) |
| 解像度 | 固定 | 固定336 | 固定448 | naive dynamic resolution | dynamic + 絶対時間 |
| 学習段階 | 2段階 | 2段階(強化) | 3段階 | 3段階 | 3段階 |
| 動画 | × | × | × | ○ (M-RoPE) | ○ (絶対時間・長尺) |

## 実装デモを動かす

各回の `practice/` に最小デモがあります。CLAUDE.md の方針に従い **Docker コンテナ内で uv** が Python 依存を管理しますが、ローカルでも素早く動かせます。

```bash
# 例: Qwen2-VL 2B（最小・クリーンに動く）
cd lectures/04_Qwen2-VL/practice
uv run demo.py
```

GPU（CUDA）が必要です。初回実行時に HuggingFace Hub からモデル重みが DL されます。サイズ選択肢や Docker 実行手順は各 `practice/README.md` を参照してください。

## このサイトについて

- 教材ソースは `lectures/<id>/README.md`（Markdown＋インライン SVG）。`tools/build_site.py` が静的 HTML に変換します。
- 論文 PDF は教材ソースとしてローカル参照しますが、再配布しないため `*.pdf` は `.gitignore` 対象です。各回の論文は arXiv で確認してください。
