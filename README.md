# lecture-vlms

LLM をバックボーンに持つ **VLM（論文表現では MLLM: 画像＋言語を入力し、LLM により言語生成を得るモデル）** を、
**論文から構造・仕組みまで理解し、最小推論デモで知見を得る**ためのサーベイ教材。

第一線の MLLM の構造・仕組みを論文ベースで理解し、知見を蓄えることを目的とする。

## 📖 教材サイト（GitHub Pages）

**https://zackey2414.github.io/lecture-vlms/**

丁寧な地の文＋SVG 図の「教科書ページ」を、ブラウザで読めます。

## 構成

```
lectures/
  <NN>_<MODEL>/
    README.md            # 教科書ページ（地の文＋インラインSVG図）→ HTML 化される
    materials/<id>.pdf   # 論文PDF（教材ソース・ローカル参照のみ。*.pdf は .gitignore）
    practice/            # 最小推論デモ
      demo.py            #   HF transformers の from_pretrained + 画像1枚で推論
      pyproject.toml     #   モデル専用 uv 環境
      Dockerfile         #   コンテナ内 uv（CLAUDE.md 準拠）
      README.md          #   実行手順・モデルサイズ選択肢
docs/                    # サイトの補助ページ・カリキュラム定義
  curriculum.json        #   各回のメタデータ（タイトル/系統/レベル/前提 等）
  getting-started.md, roadmap.md
tools/build_site.py      # Markdown(+SVG) → 静的 HTML サイト・ビルダー
.github/workflows/deploy-pages.yml  # GitHub Pages へ自動デプロイ
```

## 収録モデル（全26回・8系統）

高被引用（Semantic Scholar 調べ）かつアーキ多様性で選定。番号は安定 ID で、学習順は前提（DAG）が正。

| # | モデル | 論文 | 系統 |
|---|---|---|---|
| 01 | CLIP | [2103.00020](https://arxiv.org/abs/2103.00020) | 視覚言語の基盤 |
| 02 | SigLIP | [2303.15343](https://arxiv.org/abs/2303.15343) | 視覚言語の基盤 |
| 03 | Flamingo | [2204.14198](https://arxiv.org/abs/2204.14198) | 黎明期・原点 |
| 04 | BLIP-2 | [2301.12597](https://arxiv.org/abs/2301.12597) | 黎明期・原点 |
| 05 | InstructBLIP | [2305.06500](https://arxiv.org/abs/2305.06500) | 黎明期・原点 |
| 06 | MiniGPT-4 | [2304.10592](https://arxiv.org/abs/2304.10592) | 黎明期・原点 |
| 07 | LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) | LLaVA系 |
| 08 | LLaVA-1.5 | [2310.03744](https://arxiv.org/abs/2310.03744) | LLaVA系 |
| 09 | LLaVA-OneVision | [2408.03326](https://arxiv.org/abs/2408.03326) | LLaVA系 |
| 10 | Qwen-VL | [2308.12966](https://arxiv.org/abs/2308.12966) | Qwen系 |
| 11 | Qwen2-VL | [2409.12191](https://arxiv.org/abs/2409.12191) | Qwen系 |
| 12 | Qwen2.5-VL | [2502.13923](https://arxiv.org/abs/2502.13923) | Qwen系 |
| 13 | Qwen3-VL | [2511.21631](https://arxiv.org/abs/2511.21631) | Qwen系 |
| 14 | InternVL | [2312.14238](https://arxiv.org/abs/2312.14238) | InternVL系 |
| 15 | InternVL3 | [2504.10479](https://arxiv.org/abs/2504.10479) | InternVL系 |
| 16 | DeepSeek-VL | [2403.05525](https://arxiv.org/abs/2403.05525) | DeepSeek系 |
| 17 | DeepSeek-VL2 | [2412.10302](https://arxiv.org/abs/2412.10302) | DeepSeek系 |
| 18 | Janus | [2410.13848](https://arxiv.org/abs/2410.13848) | DeepSeek系 |
| 19 | Janus-Pro | [2501.17811](https://arxiv.org/abs/2501.17811) | DeepSeek系 |
| 20 | PaLI | [2209.06794](https://arxiv.org/abs/2209.06794) | Google系 |
| 21 | PaliGemma | [2407.07726](https://arxiv.org/abs/2407.07726) | Google系 |
| 22 | PaliGemma 2 | [2412.03555](https://arxiv.org/abs/2412.03555) | Google系 |
| 23 | Gemma 3 | [2503.19786](https://arxiv.org/abs/2503.19786) | Google系 |
| 24 | Chameleon | [2405.09818](https://arxiv.org/abs/2405.09818) | 別アーキ・統合 |
| 25 | CogVLM | [2311.03079](https://arxiv.org/abs/2311.03079) | 別アーキ・統合 |
| 26 | Emu3 | [2409.18869](https://arxiv.org/abs/2409.18869) | 別アーキ・統合 |

## サイトをローカルでビルド

```bash
uv run --group site python tools/build_site.py
# → site/ に静的 HTML を生成（site/index.html をブラウザで開く）
```

`site/` は CI でビルドして配信するため、リポジトリにはコミットしない（`.gitignore` 対象）。

## 推論デモを動かす

GPU（CUDA）が必要。初回実行時に HuggingFace Hub からモデル重みが DL される。

```bash
# 例: Qwen2-VL 2B（最小・クリーンに動く）
cd lectures/09_Qwen2-VL/practice
uv run demo.py
```

サイズ選択肢・Docker 実行は各 `practice/README.md` を参照。

## ライセンス・注記

- 論文 PDF は再配布しない（`*.pdf` は `.gitignore`）。各論文は arXiv で参照のこと。
- 教材本文・図・デモコードは本リポジトリのオリジナル。
