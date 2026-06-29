# ロードマップ

視覚言語の基盤から最新 MLLM までを、学習の流れ順・高被引用モデル中心に読む教材です。現在 **全28回（9系統）** を収録しています（うち最後の系統はモデルではなく研究トピックの出典付きサーベイ）。

> 公開サイト: <https://zackey2414.github.io/lecture-vlms/>

## 収録済み（全26回）

### 視覚言語の基盤（contrastive）

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `01_CLIP` | CLIP | [2103.00020](https://arxiv.org/abs/2103.00020) | 対照学習の二重エンコーダ。ほぼ全ての VLM/MLLM の土台 |
| `02_SigLIP` | SigLIP | [2303.15343](https://arxiv.org/abs/2303.15343) | sigmoid 対照損失で効率化、現行 MLLM の視覚側で広く採用 |

### 黎明期・原点

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `03_Flamingo` | Flamingo | [2204.14198](https://arxiv.org/abs/2204.14198) | Perceiver Resampler＋ゲート付きcross-attn、in-context few-shot |
| `04_BLIP-2` | BLIP-2 | [2301.12597](https://arxiv.org/abs/2301.12597) | Q-Former で凍結エンコーダと凍結LLMを2段階に橋渡し |
| `05_InstructBLIP` | InstructBLIP | [2305.06500](https://arxiv.org/abs/2305.06500) | instruction-aware Q-Former でゼロショット指示追従 |
| `06_MiniGPT-4` | MiniGPT-4 | [2304.10592](https://arxiv.org/abs/2304.10592) | 単一線形射影で凍結LLMに会話能力を引き出す |

### LLaVA系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `07_LLaVA` | LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) | LLM＋線形projector＋visual instruction tuning の原点 |
| `08_LLaVA-1.5` | LLaVA-1.5 | [2310.03744](https://arxiv.org/abs/2310.03744) | MLP化・解像度336・データ強化で標準ベースライン化 |
| `09_LLaVA-OneVision` | LLaVA-OneVision | [2408.03326](https://arxiv.org/abs/2408.03326) | 単一画像・複数画像・動画を統合、能力を動画へ転移 |

### Qwen系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `10_Qwen-VL` | Qwen-VL | [2308.12966](https://arxiv.org/abs/2308.12966) | 3段階学習・grounding・OCR を備えた初代 |
| `11_Qwen2-VL` | Qwen2-VL | [2409.12191](https://arxiv.org/abs/2409.12191) | naive dynamic resolution・M-RoPE・動画対応 |
| `12_Qwen2.5-VL` | Qwen2.5-VL | [2502.13923](https://arxiv.org/abs/2502.13923) | window attention ViT・絶対時間・長尺動画 |
| `13_Qwen3-VL` | Qwen3-VL | [2511.21631](https://arxiv.org/abs/2511.21631) | Dense/MoE・Thinking・256K・GUI Agent・3D grounding |

### InternVL系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `14_InternVL` | InternVL | [2312.14238](https://arxiv.org/abs/2312.14238) | 大規模 InternViT と段階的整列でスケール |
| `15_InternVL3` | InternVL3 | [2504.10479](https://arxiv.org/abs/2504.10479) | native multimodal pretraining と test-time recipe |

### DeepSeek系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `16_DeepSeek-VL` | DeepSeek-VL | [2403.05525](https://arxiv.org/abs/2403.05525) | SigLIP-L＋SAM-B のハイブリッド視覚、実世界志向 |
| `17_DeepSeek-VL2` | DeepSeek-VL2 | [2412.10302](https://arxiv.org/abs/2412.10302) | 動的タイリング＋MoE で効率と高性能を両立 |
| `18_Janus` | Janus | [2410.13848](https://arxiv.org/abs/2410.13848) | 理解と生成で視覚エンコーディングを分離・統合 |
| `19_Janus-Pro` | Janus-Pro | [2501.17811](https://arxiv.org/abs/2501.17811) | データ・規模強化で統合理解・生成を底上げ |

### Google系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `20_PaLI` | PaLI | [2209.06794](https://arxiv.org/abs/2209.06794) | 大規模 ViT と多言語を共同スケール |
| `21_PaliGemma` | PaliGemma | [2407.07726](https://arxiv.org/abs/2407.07726) | SigLIP＋Gemma の転移しやすい 3B オープン VLM |
| `22_PaliGemma-2` | PaliGemma 2 | [2412.03555](https://arxiv.org/abs/2412.03555) | Gemma 2 へ更新、複数サイズ・解像度に対応 |
| `23_Gemma-3` | Gemma 3 | [2503.19786](https://arxiv.org/abs/2503.19786) | 視覚対応の軽量オープンモデル群、長コンテキスト |

### 別アーキ・統合

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `24_Chameleon` | Chameleon | [2405.09818](https://arxiv.org/abs/2405.09818) | 画像も離散トークン化し early-fusion で理解＋生成統一 |
| `25_CogVLM` | CogVLM | [2311.03079](https://arxiv.org/abs/2311.03079) | visual expert module で deep fusion |
| `26_Emu3` | Emu3 | [2409.18869](https://arxiv.org/abs/2409.18869) | 次トークン予測のみで理解＋生成を統一 |

> モデルは高被引用（Semantic Scholar 調べ）かつアーキ多様性で選定。番号は安定 ID で、学習順は[学習順序グラフ](graph.html)の prereqs が正。

### 研究トピック・最前線（出典付きサーベイ）

| # | トピック | 内容 |
|---|---|---|
| `27_Reasoning-Transfer` | 推論能力の抽出・移植 | CoT蒸留 / reasoning vector・cross-modal task vector / model merging / RL(R1系) を俯瞰 |
| `28_Edge-Efficient-MLLM` | エッジ向け小型・高速MLLM | 小型MLLM・視覚トークン削減・量子化・KV最適化・Jetson実機実測 を俯瞰 |

> 上記2回は単一モデルではなく、ディープリサーチ（敵対的検証）に基づく研究サーベイ。各論文の arXiv ID・確証度・反証/留保を本文に明記。

## 今後の拡張候補

VLM 領域は更新が速いため、最新版・追加モデルは各 GitHub・arXiv で都度確認してください。

- **横断ファミリー**: CoCa / Kosmos-2 / mPLUG-Owl / PaLM-E / Molmo / Pixtral / NVLM / MiniCPM-V / SmolVLM
- **統合 I/O 最前線**: BAGEL ほか（統合トランスフォーマ＋画像生成ヘッド）
- 既存回への **practice デモ追加**（基盤・黎明期・DeepSeek/Google/別アーキ 系）

## 設計メモ

- 各回は `lectures/<id>/` に **教科書 `README.md`**（地の文＋SVG図＋TeX数式）を持ち、LLaVA系・Qwen系は **`practice/`**（最小推論デモ）も持つ。
- 論文 PDF は `lectures/<id>/materials/` にローカル配置（`*.pdf` は `.gitignore`、再配布しない）。
- サイトは `tools/build_site.py` でビルドし、GitHub Actions で GitHub Pages に配信する。数式は MathJax で描画。
