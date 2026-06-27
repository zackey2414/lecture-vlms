# SigLIP — Sigmoid Loss for Language Image Pre-training

SigLIP（Zhai+, 2023, arXiv:2303.15343）は、CLIP の対照学習で使われていた **softmax ベースの対照損失** を、より単純な **sigmoid 損失** に置き換えた研究である。モデル構造（画像エンコーダとテキストエンコーダの二塔構成）は CLIP と同じで、変えたのは「損失関数」だけ。それだけで、小バッチでの安定性・メモリ効率・分散学習の単純さが大きく改善する。本章では CLIP（前章）の知識を前提に、なぜ損失を変えるだけでこれほど効くのかを式と図で読み解く。

## 全体像（CLIPとの差分）

CLIP は画像とテキストをそれぞれエンコーダでベクトル化し、対になる画像-テキストを近づけ、無関係なペアを遠ざける「対照学習」で表現を獲得する。SigLIP もこの枠組みをそのまま受け継ぐ。違うのは最終段の損失だけである。

- **共通点**: 画像エンコーダ（ViT）とテキストエンコーダ（Transformer）の二塔構成、$\ell_2$ 正規化した埋め込みの内積で類似度を測る点、学習可能な温度パラメータを持つ点。
- **相違点**: CLIP は類似度行列を**バッチ全体で正規化**する softmax を使う。SigLIP は各画像-テキストペアを**独立した二値分類**として sigmoid で扱い、バッチ全体にまたがる正規化を行わない。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="CLIPとSigLIPのパイプライン比較図。画像エンコーダとテキストエンコーダ、埋め込みは共通で、最後の損失関数だけが softmax と sigmoid に分かれることを示す。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <rect x="30" y="55" width="170" height="62" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="115" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像エンコーダ</text>
  <text x="115" y="103" text-anchor="middle" font-size="12" fill="#71717a">ViT</text>
  <rect x="30" y="243" width="170" height="62" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="115" y="270" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">テキストエンコーダ</text>
  <text x="115" y="291" text-anchor="middle" font-size="12" fill="#71717a">Transformer</text>
  <rect x="265" y="149" width="150" height="62" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="340" y="176" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">埋め込み</text>
  <text x="340" y="197" text-anchor="middle" font-size="12" fill="#71717a">正規化ベクトルの内積</text>
  <line x1="200" y1="88" x2="262" y2="162" stroke="#71717a" stroke-width="2"/>
  <polygon points="262,162 250,158 256,170" fill="#71717a"/>
  <line x1="200" y1="272" x2="262" y2="198" stroke="#71717a" stroke-width="2"/>
  <polygon points="262,198 256,190 250,202" fill="#71717a"/>
  <rect x="490" y="55" width="200" height="62" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="590" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">CLIP: softmax 損失</text>
  <text x="590" y="101" text-anchor="middle" font-size="12" fill="#71717a">バッチ全体で正規化</text>
  <rect x="490" y="243" width="200" height="62" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="590" y="268" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">SigLIP: sigmoid 損失</text>
  <text x="590" y="289" text-anchor="middle" font-size="12" fill="#71717a">ペアごとに独立</text>
  <line x1="415" y1="168" x2="486" y2="92" stroke="#71717a" stroke-width="2"/>
  <polygon points="486,92 474,94 482,104" fill="#71717a"/>
  <line x1="415" y1="192" x2="486" y2="268" stroke="#71717a" stroke-width="2"/>
  <polygon points="486,268 482,256 474,266" fill="#71717a"/>
  <text x="360" y="338" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">エンコーダ・埋め込みは共通、最後の損失だけを差し替える</text>
</svg>
<figcaption>SigLIP は CLIP のアーキテクチャをそのまま使い、<b>損失関数だけ</b>を softmax から sigmoid に置き換える。だからこそ既存の CLIP 学習コードへ低コストで適用でき、効果は学習の安定性・効率に直接現れる。</figcaption>
</figure>

## softmax対照損失 vs sigmoid損失

両者の本質的な違いは「正規化の範囲」にある。記号を揃えておくと、ミニバッチ $\mathcal{B}=\{(I_1,T_1),(I_2,T_2),\dots\}$ について、画像埋め込みを $\mathbf{x}_i=f(I_i)/\lVert f(I_i)\rVert_2$、テキスト埋め込みを $\mathbf{y}_j=g(T_j)/\lVert g(T_j)\rVert_2$ とする。$t$ は学習可能な温度（実装上は $t=\exp(t')$）である。

### CLIP: softmax は「行・列の全体正規化」を要する

CLIP の損失は、各画像について「全テキストの中で正しいテキストを選ぶ」確率と、各テキストについて「全画像の中で正しい画像を選ぶ」確率を、それぞれ softmax で評価する。

$$\mathcal{L}_{\text{softmax}}=-\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\Bigg(\underbrace{\log\frac{e^{t\,\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\,\mathbf{x}_i\cdot\mathbf{y}_j}}}_{\text{画像}\to\text{テキスト}}+\underbrace{\log\frac{e^{t\,\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\,\mathbf{x}_j\cdot\mathbf{y}_i}}}_{\text{テキスト}\to\text{画像}}\Bigg)$$

ここで効いてくるのが分母の $\sum_{j=1}^{|\mathcal{B}|}$ である。第1項の分母は「画像 $i$ に対するバッチ内すべてのテキスト」の和（=類似度行列の**1行**全体）、第2項の分母は「テキスト $i$ に対するバッチ内すべての画像」の和（=**1列**全体）。つまり1ペアのスコアを求めるだけでも、その行・列のすべての類似度が揃っていなければ正規化できない。softmax は非対称なので、画像方向とテキスト方向で**正規化を2回**独立に行う必要もある。さらに数値安定化のために各行の最大値を引く処理が入り、もう1パスを要する。

### SigLIP: sigmoid は「ペアごとの二値分類」

SigLIP は問題設定そのものを置き換える。類似度行列の各セル $(i,j)$ を独立に見て、「このペアは対応する（正例）か否（負例）か」という二値分類として解く。対応するラベルを $z_{ij}$ とおくと、損失は次のように書ける（$\sigma(u)=1/(1+e^{-u})$ はシグモイド関数）。

$$\mathcal{L}=-\frac{1}{|\mathcal{B}|}\sum_{i}\sum_{j}\log\sigma\!\big(z_{ij}\,(t\,\mathbf{x}_i\!\cdot\!\mathbf{y}_j+b)\big),\quad z_{ij}=\begin{cases}+1 & i=j\\-1 & i\ne j\end{cases}$$

対角（$i=j$）が正例で $z_{ij}=+1$、それ以外は負例で $z_{ij}=-1$。各項は他のセルに一切依存せず、$\mathbf{x}_i\cdot\mathbf{y}_j$ と学習可能な温度 $t$・バイアス $b$ だけで完結する。**バッチ全体にまたがる和（正規化）が分母から消えた**のが最大の差分である。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="softmaxとsigmoidの類似度行列の扱いの対比。左は4かける4の行列で行と列の全体を正規化する softmax、右は各セルを独立に正例・負例へ二値分類する sigmoid を示す。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <text x="180" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">softmax（CLIP）</text>
  <text x="540" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#0e7490">sigmoid（SigLIP）</text>
  <g>
    <rect x="80" y="60" width="200" height="200" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
    <line x1="130" y1="60" x2="130" y2="260" stroke="#c7cae8" stroke-width="1"/>
    <line x1="180" y1="60" x2="180" y2="260" stroke="#c7cae8" stroke-width="1"/>
    <line x1="230" y1="60" x2="230" y2="260" stroke="#c7cae8" stroke-width="1"/>
    <line x1="80" y1="110" x2="280" y2="110" stroke="#c7cae8" stroke-width="1"/>
    <line x1="80" y1="160" x2="280" y2="160" stroke="#c7cae8" stroke-width="1"/>
    <line x1="80" y1="210" x2="280" y2="210" stroke="#c7cae8" stroke-width="1"/>
    <rect x="80" y="60" width="50" height="50" fill="#6366f1" opacity="0.85"/>
    <rect x="130" y="110" width="50" height="50" fill="#6366f1" opacity="0.85"/>
    <rect x="180" y="160" width="50" height="50" fill="#6366f1" opacity="0.85"/>
    <rect x="230" y="210" width="50" height="50" fill="#6366f1" opacity="0.85"/>
    <rect x="76" y="160" width="208" height="50" fill="none" stroke="#dc2626" stroke-width="3"/>
    <text x="300" y="190" text-anchor="start" font-size="12" font-weight="700" fill="#dc2626">行で正規化</text>
    <rect x="180" y="56" width="50" height="208" fill="none" stroke="#dc2626" stroke-width="3"/>
    <text x="205" y="285" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">列で正規化</text>
  </g>
  <g>
    <rect x="440" y="60" width="200" height="200" fill="#ffffff" stroke="#0e7490" stroke-width="2"/>
    <rect x="440" y="60" width="50" height="50" fill="#16a34a" opacity="0.85"/>
    <rect x="490" y="110" width="50" height="50" fill="#16a34a" opacity="0.85"/>
    <rect x="540" y="160" width="50" height="50" fill="#16a34a" opacity="0.85"/>
    <rect x="590" y="210" width="50" height="50" fill="#16a34a" opacity="0.85"/>
    <rect x="490" y="60" width="50" height="50" fill="#fee2e2"/>
    <rect x="540" y="60" width="50" height="50" fill="#fee2e2"/>
    <rect x="590" y="60" width="50" height="50" fill="#fee2e2"/>
    <rect x="440" y="110" width="50" height="50" fill="#fee2e2"/>
    <rect x="540" y="110" width="50" height="50" fill="#fee2e2"/>
    <rect x="590" y="110" width="50" height="50" fill="#fee2e2"/>
    <rect x="440" y="160" width="50" height="50" fill="#fee2e2"/>
    <rect x="490" y="160" width="50" height="50" fill="#fee2e2"/>
    <rect x="590" y="160" width="50" height="50" fill="#fee2e2"/>
    <rect x="440" y="210" width="50" height="50" fill="#fee2e2"/>
    <rect x="490" y="210" width="50" height="50" fill="#fee2e2"/>
    <rect x="540" y="210" width="50" height="50" fill="#fee2e2"/>
    <text x="465" y="90" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">+</text>
    <text x="515" y="140" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">+</text>
    <text x="565" y="190" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">+</text>
    <text x="615" y="240" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">+</text>
    <text x="540" y="290" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">各セル独立に正例(+)/負例(-)を判定</text>
  </g>
  <text x="180" y="330" text-anchor="middle" font-size="12" fill="#71717a">行・列がそろわないと1ペアも評価できない</text>
  <text x="540" y="330" text-anchor="middle" font-size="12" fill="#71717a">隣のセルに依存せず1ペアだけで評価できる</text>
</svg>
<figcaption>左の softmax は対角（青）を正例とし、各行・各列の全要素で正規化するため、行列の<b>その行・列がそろうまで損失を計算できない</b>。右の sigmoid は対角を正例（緑）、残りを負例（赤）とみなし、<b>各セルを独立に二値分類</b>する。グローバルな正規化が不要になる点が SigLIP の核心である。</figcaption>
</figure>

### バイアス項 $b$ が要る理由

二値分類に置き換えると、1つの正例に対して負例が $|\mathcal{B}|-1$ 個ある（行ごとに正例1・負例多数）という強い不均衡が生じる。学習初期はこの大量の負例が損失を支配し、すべてを「負」に倒そうとする過補正が起きやすい。SigLIP では温度 $t$ に加えて学習可能なバイアス $b$ を導入し、$t'=\log 10$、$b$ を負の大きな値（$-10$ 付近）で初期化することで、学習開始時の予測を事前分布に近づけ、この過補正を抑える。$t$ と $b$ がともに学習可能な少数のスカラーである点も実装を軽くしている。

## 小バッチでの安定性と計算効率

損失からグローバル正規化が消えたことが、そのまま実装上の利点に直結する。

- **all-gather が不要**: softmax は行・列の正規化に全デバイスの埋め込みを集約する必要があり、分散学習では高コストな all-gather と、$|\mathcal{B}|\times|\mathcal{B}|$ の巨大な類似度行列のメモリ確保を伴う。sigmoid は各項が独立なので、全埋め込みを一度に集める必要がない。
- **チャンク化で省メモリ**: テキスト表現をデバイス間で順に入れ替え（permute）ながら、各デバイスは自分の手元にある小さな $b\times b$ ブロックだけを逐次計算・累積する。瞬間的にメモリへ展開するのは小ブロックだけで済み、メモリコストは $|\mathcal{B}|^2$ から（デバイスあたりの）$b^2$ オーダーへ下がる。これにより**100万規模のバッチサイズ**での学習すら可能になった。
- **小バッチで強い**: バッチサイズが小さい領域（おおむね 16k 未満）では、sigmoid 損失が softmax 損失を明確に上回る。バッチが大きくなるほど両者の差は縮まり、性能はおおむね 32k 程度で飽和する。「大バッチでないと対照学習は効かない」という従来の前提に対し、**現実的なバッチサイズで十分**であることを示した点が実務的に重要である。なお過度に大きなバッチはむしろ性能を損なう傾向も報告されている。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="メモリと分散実装の対比。左は全デバイスから埋め込みを集める all-gather と巨大行列の展開を要する softmax、右は小ブロックをデバイス間で順送りするチャンク化された sigmoid を示す。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <text x="180" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">softmax: all-gather と全行列展開</text>
  <text x="540" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#0e7490">sigmoid: 小ブロックの順送り</text>
  <rect x="60" y="70" width="240" height="200" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="180" y="60" text-anchor="middle" font-size="12" fill="#71717a">全体行列を materialize</text>
  <g stroke="#c7cae8" stroke-width="1">
    <line x1="60" y1="120" x2="300" y2="120"/>
    <line x1="60" y1="170" x2="300" y2="170"/>
    <line x1="60" y1="220" x2="300" y2="220"/>
    <line x1="120" y1="70" x2="120" y2="270"/>
    <line x1="180" y1="70" x2="180" y2="270"/>
    <line x1="240" y1="70" x2="240" y2="270"/>
  </g>
  <rect x="36" y="100" width="20" height="60" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <rect x="36" y="180" width="20" height="60" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="180" y="300" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">全デバイスの埋め込みを集約（高コスト）</text>
  <rect x="420" y="70" width="240" height="200" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="540" y="60" text-anchor="middle" font-size="12" fill="#71717a">手元の小ブロックだけ計算</text>
  <rect x="430" y="80" width="55" height="55" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="490" y="140" width="55" height="55" fill="#06b6d4" opacity="0.85" stroke="#0e7490" stroke-width="2"/>
  <rect x="550" y="200" width="55" height="55" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <line x1="517" y1="135" x2="517" y2="140" stroke="#0e7490" stroke-width="2"/>
  <text x="630" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">順送り</text>
  <line x1="610" y1="120" x2="610" y2="225" stroke="#16a34a" stroke-width="2"/>
  <polygon points="610,225 605,213 615,213" fill="#16a34a"/>
  <text x="540" y="300" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">展開は小ブロックのみ（省メモリ）</text>
  <text x="360" y="340" text-anchor="middle" font-size="12" fill="#71717a">メモリは バッチ二乗 から デバイスあたり 小バッチ二乗 へ</text>
</svg>
<figcaption>softmax は正規化のため全デバイスの埋め込みを <b>all-gather</b> し、巨大な類似度行列を一度に展開する。sigmoid はテキスト表現をデバイス間で順送りしながら、各時点では <b>小さなブロックだけ</b>をメモリに展開して損失を累積できる。これが小チップ数・超大バッチの学習を可能にした。</figcaption>
</figure>

## 現行MLLMの視覚エンコーダとして

SigLIP の意義は、効率的な事前学習レシピにとどまらない。得られた視覚エンコーダは CLIP と同じ「言語と整合した画像表現」を持ちながら、より高い品質と効率で学習できるため、**後続の多くの MLLM（マルチモーダル LLM）の視覚側バックボーン**として広く採用された。LLaVA-OneVision（本リポジトリの後章）や PaliGemma などが代表例で、画像を SigLIP（およびその改良版 SigLIP2）でエンコードし、射影層（プロジェクタ）を介して LLM の入力空間へ橋渡しする構成が定番になっている。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="SigLIP を視覚エンコーダとして使う MLLM の構成図。画像を SigLIP でエンコードし、プロジェクタで変換して LLM に入力し、テキストを生成する流れを示す。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <rect x="30" y="140" width="120" height="80" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="90" y="176" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">入力画像</text>
  <text x="90" y="197" text-anchor="middle" font-size="12" fill="#71717a">ピクセル</text>
  <rect x="190" y="130" width="150" height="100" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="265" y="170" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">SigLIP</text>
  <text x="265" y="191" text-anchor="middle" font-size="12" fill="#71717a">視覚エンコーダ</text>
  <text x="265" y="210" text-anchor="middle" font-size="12" fill="#71717a">(ViT)</text>
  <rect x="380" y="140" width="130" height="80" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="445" y="176" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">プロジェクタ</text>
  <text x="445" y="197" text-anchor="middle" font-size="12" fill="#71717a">MLP</text>
  <rect x="550" y="130" width="140" height="100" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="620" y="170" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">LLM</text>
  <text x="620" y="191" text-anchor="middle" font-size="12" fill="#71717a">言語モデル</text>
  <text x="620" y="210" text-anchor="middle" font-size="12" fill="#71717a">テキスト生成</text>
  <line x1="150" y1="180" x2="186" y2="180" stroke="#71717a" stroke-width="2"/>
  <polygon points="186,180 174,175 174,185" fill="#71717a"/>
  <line x1="340" y1="180" x2="376" y2="180" stroke="#71717a" stroke-width="2"/>
  <polygon points="376,180 364,175 364,185" fill="#71717a"/>
  <line x1="510" y1="180" x2="546" y2="180" stroke="#71717a" stroke-width="2"/>
  <polygon points="546,180 534,175 534,185" fill="#71717a"/>
  <text x="265" y="270" text-anchor="middle" font-size="12" fill="#71717a">画像トークン列</text>
  <text x="445" y="270" text-anchor="middle" font-size="12" fill="#71717a">LLM 入力空間へ整合</text>
  <text x="360" y="320" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">LLaVA-OneVision・PaliGemma などが SigLIP を視覚側に採用</text>
</svg>
<figcaption>多くの MLLM は SigLIP を <b>視覚エンコーダ</b>として用い、プロジェクタで LLM の入力空間へ変換する。SigLIP の効率と表現品質が、こうした下流のマルチモーダル LLM の土台になっている。</figcaption>
</figure>

## まとめと、読解後に答えたい問い

- SigLIP は CLIP のアーキテクチャを変えず、損失だけを softmax から sigmoid に置き換えた。
- softmax は類似度行列を**行・列で全体正規化**するため all-gather と巨大行列の展開を要する。sigmoid は各ペアを**独立した二値分類**として扱い、グローバル正規化を不要にする。
- 損失は $\mathcal{L}=-\frac{1}{|\mathcal{B}|}\sum_i\sum_j\log\sigma(z_{ij}(t\,\mathbf{x}_i\cdot\mathbf{y}_j+b))$。学習可能な温度 $t$ とバイアス $b$ を持ち、$b$ は正例/負例の不均衡による過補正を抑える。
- 小バッチで強く、チャンク化により省メモリ・超大バッチ学習が可能。性能はおおむね 32k 程度で飽和し、巨大バッチは必須でない。
- 得られた視覚エンコーダは LLaVA-OneVision・PaliGemma など多くの MLLM の視覚バックボーンとして採用された。

読解後に、次の問いへ自分の言葉で答えられるか確認しよう。

1. softmax 対照損失が all-gather を必要とするのはなぜか。式のどの部分がその要因か。
2. sigmoid 損失で「正規化が不要になる」とはどういう意味か。1ペアの損失を計算するのに何が必要で、何が不要になったか。
3. バイアス $b$ を負の値で初期化するのはなぜか。導入しないと学習初期に何が起きるか。
4. なぜ SigLIP は小バッチで softmax を上回り、大バッチでは差が縮むのか。負例の数とどう関係するか。
5. SigLIP が後続 MLLM の視覚エンコーダとして好まれる理由を、CLIP との差分の観点から説明せよ。
