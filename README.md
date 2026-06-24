# lecture-vlms

LLM をバックボーンに持つ **VLM（論文表現では MLLM: 画像＋言語を入力し、LLM により言語生成を得るモデル）** を、
**論文から構造・仕組みまで理解し、最小推論デモで知見を得る**ためのサーベイ教材。

研究リポジトリ Adaptive Cluster-CLIP (ACC) の発展研究準備として、第一線の MLLM を押さえることを目的とする
（特に Qwen2.5-VL は ACC 論文の MLLM ベースライン比較対象）。

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

## 収録モデル（ステージ0+1）

| # | モデル | 論文 |
|---|---|---|
| 01 | LLaVA | [2304.08485](https://arxiv.org/abs/2304.08485) |
| 02 | LLaVA-1.5 | [2310.03744](https://arxiv.org/abs/2310.03744) |
| 03 | Qwen-VL | [2308.12966](https://arxiv.org/abs/2308.12966) |
| 04 | Qwen2-VL | [2409.12191](https://arxiv.org/abs/2409.12191) |
| 05 | Qwen2.5-VL | [2502.13923](https://arxiv.org/abs/2502.13923) |

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
cd lectures/04_Qwen2-VL/practice
uv run demo.py
```

サイズ選択肢・Docker 実行は各 `practice/README.md` を参照。

## ライセンス・注記

- 論文 PDF は再配布しない（`*.pdf` は `.gitignore`）。各論文は arXiv で参照のこと。
- 教材本文・図・デモコードは本リポジトリのオリジナル。
