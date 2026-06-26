# Qwen2.5-VL — Technical Report（window attention・絶対時間・長尺動画）

Qwen2.5-VL（arXiv:2502.13923）は、Qwen-VL / Qwen2-VL の系譜を継ぎつつ、ViT への **window attention** 導入、M-RoPE 時間成分の **絶対時間（秒）** への対応付け、**長尺動画**理解、そして HTML 形式での**文書・レイアウト解析**を一段押し進めたフラッグシップである。読者はすでに LLaVA / Qwen-VL / Qwen2-VL を踏まえているので、本章では「何が新しく、なぜ効くのか」を中心に腹落ちさせる。

---

## 全体像（まず一枚で）

Qwen2.5-VL は **(1) 再設計 ViT（vision encoder）→ (2) MLP merger → (3) Qwen2.5 LLM デコーダ**という、いまや見慣れた三段構成を取る。新しさは各段の中身にある。ViT は **2D-RoPE + window attention** をネイティブ解像度のまま処理できるよう再設計され、`RMSNorm` と `SwiGLU` を採用して LLM 側の設計思想に揃えてある。merger は空間的に隣接する **2×2＝4 個のパッチ特徴**をまとめて MLP で LLM 埋め込み次元へ射影し、視覚トークン列を圧縮する。動画では `Conv3D`(2×14×14) で連続 2 フレームをまとめ、トークン数をさらに削減する。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="Qwen2.5-VLの全体構成。再設計ViTからMLP mergerを経てQwen2.5 LLMデコーダへ流れる三段構成" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="20" y="40" width="150" height="280" rx="10" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="95" y="64" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">ネイティブ解像度入力</text>
<rect x="40" y="80" width="110" height="60" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
<text x="95" y="105" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">画像</text>
<text x="95" y="124" text-anchor="middle" font-size="11" fill="#71717a">任意のH×W</text>
<rect x="40" y="155" width="110" height="80" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="95" y="180" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">動画フレーム列</text>
<text x="95" y="199" text-anchor="middle" font-size="11" fill="#71717a">可変FPS</text>
<text x="95" y="216" text-anchor="middle" font-size="11" fill="#71717a">Conv3D 2フレーム束</text>
<text x="95" y="260" text-anchor="middle" font-size="11" fill="#71717a">パッチstride14</text>
<text x="95" y="278" text-anchor="middle" font-size="11" fill="#71717a">28の倍数にリサイズ</text>
<rect x="210" y="90" width="170" height="180" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="295" y="116" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">再設計ViT</text>
<text x="295" y="140" text-anchor="middle" font-size="11" fill="#15803d">window attention ×多数</text>
<text x="295" y="160" text-anchor="middle" font-size="11" fill="#15803d">full attention ×少数</text>
<text x="295" y="180" text-anchor="middle" font-size="11" fill="#15803d">2D-RoPE</text>
<text x="295" y="200" text-anchor="middle" font-size="11" fill="#15803d">RMSNorm / SwiGLU</text>
<text x="295" y="232" text-anchor="middle" font-size="11" fill="#71717a">計算量ほぼ線形</text>
<rect x="420" y="120" width="120" height="120" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="480" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">MLP merger</text>
<text x="480" y="192" text-anchor="middle" font-size="11" fill="#0e7490">2×2パッチを集約</text>
<text x="480" y="210" text-anchor="middle" font-size="11" fill="#71717a">トークン列を圧縮</text>
<rect x="580" y="90" width="120" height="180" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="640" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">Qwen2.5</text>
<text x="640" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">LLMデコーダ</text>
<text x="640" y="206" text-anchor="middle" font-size="11" fill="#71717a">M-RoPE（時間=絶対秒）</text>
<line x1="170" y1="180" x2="208" y2="180" stroke="#71717a" stroke-width="2"/>
<polygon points="208,180 198,175 198,185" fill="#71717a"/>
<line x1="380" y1="180" x2="418" y2="180" stroke="#71717a" stroke-width="2"/>
<polygon points="418,180 408,175 408,185" fill="#71717a"/>
<line x1="540" y1="180" x2="578" y2="180" stroke="#71717a" stroke-width="2"/>
<polygon points="578,180 568,175 568,185" fill="#71717a"/>
</svg><figcaption><b>三段構成</b>は LLaVA 以来の定石だが、Qwen2.5-VL は <b>ViT を window attention で軽量化</b>し、<b>merger で視覚トークンを圧縮</b>し、<b>LLM 側の M-RoPE 時間成分を絶対秒に揃える</b>点が要点。</figcaption></figure>

サイズは **3B / 7B / 72B** の三種で、エッジから高性能計算まで想定する。ViT の hidden は全サイズ共通の 1280、merger の出力次元だけが LLM の hidden（3B:2048 / 7B:3584 / 72B:8192）に合わせて変わる。事前学習トークンは Qwen2-VL の 1.2T から約 4.1T へ拡大している。

---

## window attention ViT による効率化

ネイティブ解像度で画像を処理すると、入力ごとにパッチ数（＝シーケンス長）が大きく変動する。標準的な ViT は全層が全結合の self-attention なので、計算量はパッチ数の **二次**で効いてしまい、高解像度の文書画像や長尺動画では推論コストが跳ね上がる。これが Qwen2-VL までの素朴な ViT のボトルネックだった。

Qwen2.5-VL の答えはシンプルで、**ほとんどの層を window attention（窓内のパッチだけで attention）にし、ごく少数の層だけ full attention を残す**というもの。論文の構成では **full attention を持つのはわずか 4 層**（Full Attention Block Indexes `{7, 15, 23, 31}`）で、残りの層はすべて最大 **112×112 ピクセル（＝8×8 パッチ）**の窓内に attention を閉じ込める。112×112 より小さい領域はパディングせずそのまま扱うので、ネイティブ解像度が保たれる。

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="全attentionと窓attentionの計算量対比。全attentionはパッチ数の二次、窓attentionはほぼ線形" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="180" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">全 attention（従来）</text>
<rect x="40" y="50" width="280" height="150" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<g stroke="#dc2626" stroke-width="1" opacity="0.5">
<line x1="70" y1="80" x2="290" y2="170"/><line x1="70" y1="80" x2="290" y2="80"/>
<line x1="70" y1="80" x2="170" y2="170"/><line x1="70" y1="170" x2="290" y2="80"/>
<line x1="170" y1="80" x2="290" y2="170"/><line x1="70" y1="170" x2="290" y2="170"/>
<line x1="70" y1="125" x2="290" y2="125"/><line x1="170" y1="80" x2="170" y2="170"/>
</g>
<text x="180" y="190" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">全パッチが相互に参照 → 計算量 O(N²)</text>
<text x="540" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">窓 attention（多数層）</text>
<rect x="400" y="50" width="280" height="150" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<rect x="412" y="62" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<rect x="500" y="62" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<rect x="588" y="62" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<rect x="412" y="130" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<rect x="500" y="130" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<rect x="588" y="130" width="80" height="60" rx="4" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.5"/>
<text x="540" y="218" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">8×8パッチの窓内のみ参照 → 計算量 ほぼ O(N)</text>
<rect x="40" y="248" width="640" height="56" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="360" y="270" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">少数の full attention 層（{7,15,23,31}）が窓をまたぐ大域文脈を回収</text>
<text x="360" y="290" text-anchor="middle" font-size="11" fill="#71717a">→ 大域情報を失わずに、計算量を画像サイズ／パッチ数にほぼ線形へ</text>
</svg><figcaption>多くの層を <b>窓 attention</b>にして計算量を<b>ほぼ線形</b>に抑え、ごく少数の <b>full attention 層</b>で窓をまたぐ大域文脈を回収する。高解像度・長尺入力でも ViT がボトルネックになりにくい。</figcaption></figure>

設計上の細部も LLM と整合させてある。位置符号は **2D-RoPE**（空間の縦横を回転位置で表現）、正規化は `RMSNorm`、活性化は `SwiGLU`。これらは LayerNorm + GELU の旧来 ViT 流儀から離れ、Qwen2.5 LLM 側と同じ部品を使うことで、視覚と言語の橋渡しを素直にしている。なお ViT は事前学習済み CLIP からの流用ではなく **スクラッチ学習**（DataComp 等で CLIP 事前学習 → 整列 → end-to-end fine-tune）である点も Qwen2-VL から踏襲・強化された方針だ。

---

## 絶対時間エンコーディングと長尺動画

動画理解の鍵は「フレーム番号」ではなく「いつ起きたか（秒）」を扱えることだ。Qwen2-VL の M-RoPE は位置を **時間・縦・横**の三成分に分解していたが、その**時間成分はフレーム数に紐づいて**いた。これだと FPS が変われば同じ「秒」でも時間 ID がずれてしまい、イベントの**速さ**や**絶対的なタイミング**を正しく学べない。

Qwen2.5-VL の改良は一点突破で明快だ。**M-RoPE の時間成分を絶対時間（秒）に対応付ける**。時間 ID 同士の**間隔**をそのまま秒の間隔として解釈させることで、FPS が違う動画でも一貫した時間整列を学習でき、しかも**追加の計算ヘッドや計算オーバーヘッドは不要**である（テキストのタイムスタンプを差し込む方式とも、専用ヘッドを足す方式とも異なる）。学習時には **dynamic FPS sampling**（FPS を動的にサンプル）で多様なフレームレートを満遍なく経験させる。

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="フレーム列を絶対秒に対応付ける図。FPSが異なっても同じ秒には同じ時間IDが割り当てられる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="30" y="40" font-size="13" font-weight="700" fill="#4338ca">高FPS動画</text>
<g>
<rect x="120" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<rect x="180" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<rect x="240" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<rect x="300" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<rect x="360" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
<rect x="420" y="22" width="40" height="32" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
</g>
<text x="30" y="120" font-size="13" font-weight="700" fill="#0e7490">低FPS動画</text>
<g>
<rect x="120" y="102" width="40" height="32" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
<rect x="240" y="102" width="40" height="32" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
<rect x="360" y="102" width="40" height="32" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
</g>
<line x1="120" y1="180" x2="660" y2="180" stroke="#71717a" stroke-width="2"/>
<polygon points="660,180 650,175 650,185" fill="#71717a"/>
<g font-size="12" font-weight="700" fill="#18181b">
<line x1="140" y1="174" x2="140" y2="186" stroke="#71717a" stroke-width="2"/><text x="140" y="204" text-anchor="middle">0s</text>
<line x1="260" y1="174" x2="260" y2="186" stroke="#71717a" stroke-width="2"/><text x="260" y="204" text-anchor="middle">1s</text>
<line x1="380" y1="174" x2="380" y2="186" stroke="#71717a" stroke-width="2"/><text x="380" y="204" text-anchor="middle">2s</text>
<line x1="500" y1="174" x2="500" y2="186" stroke="#71717a" stroke-width="2"/><text x="500" y="204" text-anchor="middle">3s</text>
<line x1="620" y1="174" x2="620" y2="186" stroke="#71717a" stroke-width="2"/><text x="620" y="204" text-anchor="middle">4s</text>
</g>
<text x="390" y="232" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">M-RoPE 時間成分 = 絶対秒（時間IDの間隔＝秒の間隔）</text>
<rect x="120" y="250" width="520" height="54" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="380" y="272" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">FPSが違っても「同じ秒」には整合した時間IDが付く</text>
<text x="380" y="292" text-anchor="middle" font-size="11" fill="#71717a">→ 可変FPS・時間オーダーの長尺動画に頑健、秒単位のイベント局在化が可能</text>
</svg><figcaption>フレーム番号ではなく <b>絶対秒</b>へ対応付けるため、FPS が違っても時間整列が崩れない。<b>秒単位の時間ローカライズ／grounding</b> と、数時間規模の<b>長尺動画</b>理解が成立する。</figcaption></figure>

この時間軸の刷新は、空間側の **native dynamic resolution**（座標を正規化せず実寸のまま扱い、視覚トークンが実サイズを反映する）と表裏一体である。空間は実ピクセル、時間は実秒——どちらも「正規化に頼らず物理的なスケールをそのまま学ぶ」という一貫した思想だ。長尺対応のため、第 3 段の事前学習では系列長を 32,768 まで延ばし、長尺動画・長尺文書・長尺エージェント軌跡を投入している。

---

## 文書・レイアウト解析と grounding

Qwen2.5-VL のもう一つの目玉は、文書を**構造ごと**読むことだ。従来はレイアウト解析・テキスト抽出・チャート解釈・図表処理を別々のモデルで担うのが普通だったが、本モデルは単一モデルでこれらをまとめてこなす。出力形式が「**QwenVL HTML**」で、レイアウトボックス・表・チャート・数式・画像キャプション・楽譜・化学式を、座標 `data-bbox="x1 y1 x2 y2"` 付きの HTML タグ構造として吐く。

```html
<html><body>
  <p data-bbox="x1 y1 x2 y2">本文段落</p>
  <table data-bbox="x1 y1 x2 y2" class="table{id}"> 表の中身 </table>
  <div class="chart" data-bbox="x1 y1 x2 y2"><img data-bbox="..."/><table> チャートの中身 </table></div>
  <div class="formula" data-bbox="x1 y1 x2 y2"><img data-bbox="..."/><div> 数式の中身 </div></div>
  <div class="chemical formula" format="smile" data-bbox="x1 y1 x2 y2"> ... </div>
</body></html>
```

ポイントは座標が **絶対座標（実ピクセル）**で出ること。これは grounding（物体検出・指差し・カウント）とも同じ流儀で、bounding box / point を実寸座標、あるいは JSON 形式で返す。学習では 10,000 を超える物体カテゴリへ拡張し、open-vocabulary 検出を強化している。さらに **agent 機能**として、PC・モバイル・Web の GUI スクリーンショットを認識し、UI 要素を grounding して関数呼び出し形式の操作（多段の行動）を実行できる。

| 能力 | 出力の形 | 何が嬉しいか |
|---|---|---|
| 文書 omni-parsing | QwenVL HTML（レイアウト/表/チャート/数式/楽譜/化学式） | 構造を保ったまま一括抽出・変換 |
| object grounding | 実寸の bbox / point / JSON | 正規化に頼らずスケールを正しく扱う |
| 動画 grounding | 秒・hmsf 形式のタイムスタンプ | 秒単位のイベント局在化 |
| GUI agent | 共有アクション空間の関数呼び出し | モバイル/PC のタスク自動実行 |

研究者視点で大事なのは、これらが**すべて「絶対座標・絶対時間」という共通の物理スケール表現**に乗っていることだ。空間 grounding・時間 grounding・文書 grounding が同じ枠組みで扱えるからこそ、後段のタスク設計が素直になる。

---

## まとめと、読解後に答えたい問い

Qwen2.5-VL の核は四点に集約できる。**(1)** ViT に window attention を導入し、少数の full attention 層だけ残して計算量を画像サイズにほぼ線形へ。**(2)** native dynamic resolution を継承し、座標を実寸のまま扱う。**(3)** M-RoPE の時間成分を絶対秒へ対応付け、可変 FPS・長尺動画・秒単位 grounding を追加コストなしで獲得。**(4)** QwenVL HTML による構造化文書解析と絶対座標 grounding、GUI agent。空間も時間も「正規化せず物理スケールをそのまま学ぶ」という一貫した思想が貫かれている。

実装は `transformers>=4.49` の `Qwen2_5_VLForConditionalGeneration` と、可変解像度・動画フレーム前処理を担う `qwen-vl-utils` の組み合わせが標準だ。

```python
# transformers>=4.49
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# messages に画像/動画(可変FPS)を入れ、process_vision_info で視覚入力を抽出
```

読解後に自分で答えられるようにしておきたい問い。

1. window attention で計算量がほぼ線形になるのに、なぜ少数の full attention 層を残す必要があるのか。それを全て窓にしたら何が失われるか。
2. M-RoPE の時間成分を「フレーム番号」から「絶対秒」に変えると、可変 FPS 動画の学習で具体的に何が改善するか。テキストタイムスタンプ注入や専用ヘッド方式と比べた利点は。
3. 文書・物体・動画の grounding がすべて「絶対座標／絶対時間」に統一されていることの実務的メリットは何か。
