# ロードマップ

LLM ベース VLM(MLLM) を学習の流れ順・高被引用モデル中心に読む教材です。現在 **全13回（4系統）** を収録しています。

> 公開サイト: <https://zackey2414.github.io/lecture-vlms/>

## 収録済み（全13回）

### 黎明期・原点

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `01_Flamingo` | Flamingo | [2204.14198](https://arxiv.org/abs/2204.14198) | Perceiver Resampler＋ゲート付きcross-attn、in-context few-shot |
| `02_BLIP-2` | BLIP-2 | [2301.12597](https://arxiv.org/abs/2301.12597) | Q-Former で凍結エンコーダと凍結LLMを2段階に橋渡し |
| `03_InstructBLIP` | InstructBLIP | [2305.06500](https://arxiv.org/abs/2305.06500) | instruction-aware Q-Former でゼロショット指示追従 |
| `04_MiniGPT-4` | MiniGPT-4 | [2304.10592](https://arxiv.org/abs/2304.10592) | 単一線形射影で凍結LLMに会話能力を引き出す |

### LLaVA系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `05_LLaVA` | LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) | LLM＋線形projector＋visual instruction tuning の原点 |
| `06_LLaVA-1.5` | LLaVA-1.5 | [2310.03744](https://arxiv.org/abs/2310.03744) | MLP化・解像度336・データ強化で標準ベースライン化 |
| `07_LLaVA-OneVision` | LLaVA-OneVision | [2408.03326](https://arxiv.org/abs/2408.03326) | 単一画像・複数画像・動画を統合、能力を動画へ転移 |

### Qwen系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `08_Qwen-VL` | Qwen-VL | [2308.12966](https://arxiv.org/abs/2308.12966) | 3段階学習・grounding・OCR を備えた初代 |
| `09_Qwen2-VL` | Qwen2-VL | [2409.12191](https://arxiv.org/abs/2409.12191) | naive dynamic resolution・M-RoPE・動画対応 |
| `10_Qwen2.5-VL` | Qwen2.5-VL | [2502.13923](https://arxiv.org/abs/2502.13923) | window attention ViT・絶対時間・長尺動画 |
| `11_Qwen3-VL` | Qwen3-VL | [2511.21631](https://arxiv.org/abs/2511.21631) | Dense/MoE・Thinking・256K・GUI Agent・3D grounding |

### InternVL系

| # | モデル | 論文 | 一言 |
|---|---|---|---|
| `12_InternVL` | InternVL | [2312.14238](https://arxiv.org/abs/2312.14238) | 大規模 InternViT と段階的整列でスケール |
| `13_InternVL3` | InternVL3 | [2504.10479](https://arxiv.org/abs/2504.10479) | native multimodal pretraining と test-time recipe |

> モデルは高被引用（Semantic Scholar 調べ）かつアーキ多様性で選定。番号は安定 ID で、学習順は[学習順序グラフ](graph.html)の prereqs が正。

## 今後の拡張候補

VLM 領域は更新が速いため、最新版・追加モデルは各 GitHub・arXiv で都度確認してください。

- **効率・統合**: DeepSeek-VL / VL2（MoE）、Janus / Janus-Pro（理解と生成の分離・統合）、Emu3（次トークン予測統一）
- **横断ファミリー**: CogVLM / GLM-4.5V / Kimi-VL / MiniCPM-V / Molmo / SmolVLM / Gemma 3 / Pixtral / NVLM
- **統合 I/O 最前線**: BAGEL ほか（統合トランスフォーマ＋画像生成ヘッド）
- 既存回への **practice デモ追加**（黎明期・InternVL 系）

## 設計メモ

- 各回は `lectures/<id>/` に **教科書 `README.md`**（地の文＋SVG図）を持ち、LLaVA系・Qwen系は **`practice/`**（最小推論デモ）も持つ。
- 論文 PDF は `lectures/<id>/materials/` にローカル配置（`*.pdf` は `.gitignore`、再配布しない）。
- サイトは `tools/build_site.py` でビルドし、GitHub Actions で GitHub Pages に配信する。
