# はじめ方

この教材は、**視覚言語モデル（VLM）の基盤から最新の MLLM まで**を、論文から構造・仕組みまで理解するためのものです。各回は次の構成です。

- **教科書ページ（このサイト）** — 丁寧な地の文と SVG 図・数式で、論文の仕組みを解説します（全回）。
- **最小推論デモ（`practice/demo.py`）** — HF transformers の `from_pretrained` で実際にモデルを動かします（LLaVA系・Qwen系の一部）。

## 読む順番（8系統・全26回）

番号は安定した ID で、実際に学ぶ順番は各回の「前提」をたどる [学習順序グラフ](graph.html)（DAG）が正です。大きく8系統に分かれます。

- **視覚言語の基盤**: `01_CLIP` / `02_SigLIP` — 対照学習の二重エンコーダ。ほぼ全ての MLLM の視覚側の土台。
- **黎明期・原点**: `03_Flamingo` / `04_BLIP-2` / `05_InstructBLIP` / `06_MiniGPT-4` — 凍結エンコーダ＋凍結LLM を Perceiver / Q-Former / 線形射影で橋渡しする MLLM の出発点。
- **LLaVA系**: `07_LLaVA` / `08_LLaVA-1.5` / `09_LLaVA-OneVision` — visual instruction tuning の雛形から画像・複数画像・動画の統合まで。
- **Qwen系**: `10_Qwen-VL` 〜 `13_Qwen3-VL` — 3段階学習・動的解像度・M-RoPE・長尺動画・MoE/Thinking へ縦に深掘り。
- **InternVL系**: `14_InternVL` / `15_InternVL3` — 大規模 InternViT と native multimodal pretraining。
- **DeepSeek系**: `16_DeepSeek-VL` / `17_DeepSeek-VL2` / `18_Janus` / `19_Janus-Pro` — 効率設計・MoE、そして理解＋生成の統合。
- **Google系**: `20_PaLI` / `21_PaliGemma` / `22_PaliGemma-2` / `23_Gemma-3` — スケール志向と SigLIP＋Gemma のオープン VLM。
- **別アーキ・統合**: `24_Chameleon` / `25_CogVLM` / `26_Emu3` — early-fusion トークン統一・deep-fusion・次トークン統一生成。

## 読むときの5観点（比較メモのテンプレ）

各回を同じ観点で比較すると、モデルの差分が立体的に見えます。

1. **視覚側** — 対照学習エンコーダ（CLIP / SigLIP）/ 大規模 ViT / 離散トークン化（VQ）
2. **コネクタ設計** — Perceiver / Q-Former / 線形・MLP projector / visual expert / early-fusion（コネクタなし）
3. **解像度処理** — 固定 / 動的解像度 / タイリング（AnyRes）
4. **学習段階と統合** — pretraining → SFT → preference/RL、理解のみ / 理解＋生成
5. **動画・時間** — フレーム処理・時間符号化（M-RoPE・絶対時間など）

### コネクタ設計の系譜（ざっくり）

| 系統 | 代表 | 視覚→言語の繋ぎ方 |
|---|---|---|
| 基盤 | CLIP / SigLIP | （生成はせず）対照学習で同一空間に整列 |
| 黎明期 | Flamingo | Perceiver Resampler ＋ ゲート付き cross-attn |
| 黎明期 | BLIP-2 / InstructBLIP / MiniGPT-4 | Q-Former（＋線形射影） |
| LLaVA系 | LLaVA → 1.5 | 線形 → 2層 MLP projector |
| Qwen系 | Qwen-VL〜 | Adapter / MLP merger ＋ 動的解像度・M-RoPE |
| Google系 | PaliGemma | SigLIP ＋ 線形 projector ＋ Gemma |
| 別アーキ | CogVLM | visual expert module（deep fusion） |
| 別アーキ | Chameleon / Emu3 | 画像も離散トークン化し early-fusion／次トークン予測で統一 |

## 実装デモを動かす

LLaVA系・Qwen系の各回には `practice/` に最小デモがあります。CLAUDE.md の方針に従い **Docker コンテナ内で uv** が Python 依存を管理しますが、ローカルでも素早く動かせます。

```bash
# 例: Qwen2-VL 2B（最小・クリーンに動く）
cd lectures/11_Qwen2-VL/practice
uv run demo.py
```

GPU（CUDA）が必要です。初回実行時に HuggingFace Hub からモデル重みが DL されます。サイズ選択肢や Docker 実行手順は各 `practice/README.md` を参照。その他の系統は現状「教科書ページ」中心です（デモは今後追加予定）。

## このサイトについて

- 教材ソースは `lectures/<id>/README.md`（Markdown＋インライン SVG＋TeX 数式）。`tools/build_site.py` が静的 HTML に変換し、数式は MathJax で描画します。
- 論文 PDF は教材ソースとしてローカル参照しますが、再配布しないため `*.pdf` は `.gitignore` 対象です。各回の論文は arXiv で確認してください。
