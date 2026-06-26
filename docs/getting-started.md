# はじめ方

この教材は、LLM をバックボーンに持つ **VLM（論文表現では MLLM: 画像＋言語を入力し、LLM により言語生成を得るモデル）** を、論文から構造・仕組みまで理解するためのものです。各回は次の構成です。

- **教科書ページ（このサイト）** — 丁寧な地の文と SVG 図で、論文の仕組みを解説します（全回）。
- **最小推論デモ（`practice/demo.py`）** — HF transformers の `from_pretrained` で実際にモデルを動かします（一部の回）。

## 読む順番（4系統・全13回）

番号は安定した ID で、実際に学ぶ順番は各回の「前提」をたどる [学習順序グラフ](graph.html)（DAG）が正です。大きく4系統に分かれます。

- **黎明期・原点**: `01_Flamingo` / `02_BLIP-2` / `03_InstructBLIP` / `04_MiniGPT-4`
  — 凍結エンコーダ＋凍結LLM を Perceiver / Q-Former / 線形射影で橋渡しする、MLLM の出発点。
- **LLaVA系**: `05_LLaVA` / `06_LLaVA-1.5` / `07_LLaVA-OneVision`
  — visual instruction tuning の雛形から、画像・複数画像・動画の統合まで。
- **Qwen系**: `08_Qwen-VL` / `09_Qwen2-VL` / `10_Qwen2.5-VL` / `11_Qwen3-VL`
  — 3段階学習・動的解像度・M-RoPE・長尺動画・MoE/Thinking へと縦に深掘り。
- **InternVL系**: `12_InternVL` / `13_InternVL3`
  — 大規模視覚エンコーダ（InternViT）と native multimodal pretraining の別系統。

## 読むときの5観点（比較メモのテンプレ）

各回を同じ観点で比較すると、モデルの差分が立体的に見えます。

1. **コネクタ設計** — Perceiver Resampler / Q-Former / 線形・MLP projector / native fusion
2. **解像度処理** — 固定 / 動的解像度 / タイリング（AnyRes）
3. **学習段階** — pretraining → SFT → preference/RL の構成
4. **動画対応** — フレーム処理・時間符号化（M-RoPE・絶対時間など）
5. **後処理アライメント** — DPO / MPO / GRPO 等

### コネクタ設計の系譜（ざっくり）

| 系統 | 代表 | コネクタ |
|---|---|---|
| 黎明期 | Flamingo | Perceiver Resampler ＋ ゲート付き cross-attn |
| 黎明期 | BLIP-2 / InstructBLIP | Q-Former（学習可能クエリの Transformer） |
| 黎明期 | MiniGPT-4 | Q-Former ＋ 単一線形射影 |
| LLaVA系 | LLaVA → 1.5 | 線形 → 2層 MLP projector |
| Qwen系 | Qwen-VL | Position-aware VL Adapter（cross-attn） |
| Qwen系 | Qwen2-VL〜 | MLP merger ＋ 動的解像度・M-RoPE |
| InternVL系 | InternVL | 大規模 InternViT ＋ 段階的整列 |

## 実装デモを動かす

LLaVA系・Qwen系の各回には `practice/` に最小デモがあります。CLAUDE.md の方針に従い **Docker コンテナ内で uv** が Python 依存を管理しますが、ローカルでも素早く動かせます。

```bash
# 例: Qwen2-VL 2B（最小・クリーンに動く）
cd lectures/09_Qwen2-VL/practice
uv run demo.py
```

GPU（CUDA）が必要です。初回実行時に HuggingFace Hub からモデル重みが DL されます。サイズ選択肢や Docker 実行手順は各 `practice/README.md` を参照してください。黎明期・InternVL 系は現状「教科書ページ」中心です（デモは今後追加予定）。

## このサイトについて

- 教材ソースは `lectures/<id>/README.md`（Markdown＋インライン SVG）。`tools/build_site.py` が静的 HTML に変換します。
- 論文 PDF は教材ソースとしてローカル参照しますが、再配布しないため `*.pdf` は `.gitignore` 対象です。各回の論文は arXiv で確認してください。
