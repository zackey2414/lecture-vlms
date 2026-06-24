# ロードマップ

LLM ベース VLM(MLLM) を学習の流れ順に読む教材です。現在は **ステージ0+1（パラダイム原点＋Qwen系）** の5本を収録しています。

## 収録済み（ステージ0+1）

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `01_LLaVA` | LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) | LLM＋線形projector＋visual instruction tuning の原点（最重要） |
| `02_LLaVA-1.5` | LLaVA-1.5 | [2310.03744](https://arxiv.org/abs/2310.03744) | MLP化・解像度336・データ強化で標準ベースライン化 |
| `03_Qwen-VL` | Qwen-VL | [2308.12966](https://arxiv.org/abs/2308.12966) | 3段階学習・grounding・OCR を備えた初代 |
| `04_Qwen2-VL` | Qwen2-VL | [2409.12191](https://arxiv.org/abs/2409.12191) | naive dynamic resolution・M-RoPE・動画対応 |
| `05_Qwen2.5-VL` | Qwen2.5-VL | [2502.13923](https://arxiv.org/abs/2502.13923) | window attention ViT・絶対時間・長尺動画（ACCベースライン） |

## 今後の拡張候補（ステージ2以降）

VLM 領域は更新が速いため、最新版・追加モデルは各 GitHub・arXiv で都度確認してください。

- **InternVL 系** — ネイティブ・マルチモーダルの別解（InternViT、動的高解像度、native multimodal pretraining）
- **DeepSeek-VL / Janus / Emu3** — 効率設計・理解と生成の統合・次トークン予測統一
- **統合 I/O 最前線** — BAGEL ほか（統合トランスフォーマ＋画像生成ヘッド）
- **横断ファミリー** — CogVLM / GLM-4.5V / Kimi-VL / MiniCPM-V / Molmo / SmolVLM / Gemma 3 など

## 設計メモ

- 各回は `lectures/<id>/` に **教科書 `README.md`**（地の文＋SVG図）と **`practice/`**（最小推論デモ）を持つ。
- 論文 PDF は `lectures/<id>/materials/` にローカル配置（`*.pdf` は `.gitignore`、再配布しない）。
- モデルごとに `practice/` の uv 環境を分離し、transformers 等の依存衝突を回避する。
- サイトは `tools/build_site.py` でビルドし、GitHub Actions で GitHub Pages に配信する。
