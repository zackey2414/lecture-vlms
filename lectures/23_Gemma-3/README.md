# Gemma 3 — Technical Report（視覚対応の軽量オープンモデル）

Gemma 3 は、Google DeepMind が公開した軽量オープンモデル群に「視覚理解」「長コンテキスト」「多言語」を加えた世代である。サイズは $1\text{B}$ から $27\text{B}$ までで、スマートフォンやノート PC、ハイエンド GPU といった一般的なハードウェアで動かすことを念頭に設計されている。これまでの章で扱ってきた LLaVA 系の VLM が「視覚エンコーダ＋言語モデル」を後付けで接続したのに対し、Gemma 3 は最初から「テキストと画像を一つの系列で扱う言語モデル」として整えられている点が特徴だ。

本章では、技術レポート（arXiv:2503.19786）の記述に沿って、(1) どう画像を言語モデルへ流し込むか、(2) 長コンテキストをどうメモリ効率よく扱うか、という二つの軸を中心に読み解いていく。視覚対応は $4\text{B}$ / $12\text{B}$ / $27\text{B}$ が担い、$1\text{B}$ はテキスト専用である点を最初に押さえておこう。

## 全体像（まず一枚で）

Gemma 3 の処理の流れは「画像 → 視覚エンコーダ → 固定数の視覚トークン → テキストトークンと結合 → デコーダ専用 Transformer」という一本道で理解できる。画像は **SigLIP 系の視覚エンコーダ**（およそ $400\text{M}$ クラスの Vision Transformer）で符号化され、**固定長の $256$ 個の視覚トークン**（ソフトトークン）に圧縮される。これがテキストのトークン列と同じ「系列」に並び、Gemma 3 のデコーダがその上で次トークンを予測する。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="画像をSigLIP系エンコーダで256個の固定長視覚トークンへ符号化し、テキストトークンと結合してGemma 3のデコーダ専用Transformerへ入力する全体パイプラインの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="a1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#4338ca"/></marker>
  </defs>
  <text x="360" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">視覚と言語を一つの系列で扱う</text>
  <rect x="24" y="150" width="92" height="64" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="70" y="180" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像</text>
  <text x="70" y="200" text-anchor="middle" font-size="11" fill="#71717a">1B は対象外</text>
  <line x1="118" y1="182" x2="150" y2="182" stroke="#4338ca" stroke-width="2" marker-end="url(#a1)"/>
  <rect x="152" y="128" width="152" height="108" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="228" y="168" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">SigLIP 系</text>
  <text x="228" y="188" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">視覚エンコーダ</text>
  <text x="228" y="212" text-anchor="middle" font-size="12" fill="#4338ca">入力 896 かける 896</text>
  <line x1="304" y1="182" x2="336" y2="182" stroke="#4338ca" stroke-width="2" marker-end="url(#a1)"/>
  <rect x="338" y="150" width="124" height="64" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="400" y="178" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">256 視覚トークン</text>
  <text x="400" y="198" text-anchor="middle" font-size="12" fill="#0e7490">固定長に圧縮</text>
  <rect x="338" y="252" width="124" height="50" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="400" y="282" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">テキストトークン</text>
  <line x1="462" y1="182" x2="498" y2="182" stroke="#4338ca" stroke-width="2" marker-end="url(#a1)"/>
  <line x1="462" y1="276" x2="498" y2="240" stroke="#71717a" stroke-width="2" marker-end="url(#a1)"/>
  <rect x="500" y="128" width="172" height="174" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="586" y="170" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">Gemma 3</text>
  <text x="586" y="192" text-anchor="middle" font-size="12" fill="#4338ca">デコーダ専用 Transformer</text>
  <text x="586" y="214" text-anchor="middle" font-size="12" fill="#4338ca">local と global の注意</text>
  <text x="586" y="236" text-anchor="middle" font-size="12" fill="#4338ca">GQA と RMSNorm</text>
  <line x1="586" y1="302" x2="586" y2="332" stroke="#4338ca" stroke-width="2" marker-end="url(#a1)"/>
  <text x="586" y="350" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">テキスト応答</text>
</svg><figcaption>画像は SigLIP 系エンコーダで <b>固定長 256 トークン</b>へ符号化され、テキストと同じ系列に並んでデコーダへ入る。<b>要点</b>: 視覚は後付けではなく、最初から一つの言語モデルの中で扱われる。</figcaption></figure>

アーキテクチャの土台は素直な**デコーダ専用 Transformer** であり、Gemma 2 から受け継いだ要素が多い。注意機構には **Grouped-Query Attention（GQA）** を採用し、正規化には pre-norm と post-norm の両方で RMSNorm を用いる。Gemma 2 にあった注意ロジットの soft-capping は、Gemma 3 では **QK-norm**（query/key の正規化）に置き換えられている。語彙は $256\text{k}$ 規模で、Gemini 2.0 と同じ SentencePiece トークナイザを共有し、非英語言語にもバランスよく配分されている。

GQA は、複数の query ヘッドで key/value のヘッドを共有する仕組みで、注意の表現力をほぼ保ったまま **KV キャッシュを小さくする**。これは後述する local/global の配置と同じ「メモリを抑える」方向の工夫であり、Gemma 3 では両者が重なって長コンテキスト時のメモリ効率を底上げしている。視覚トークンも結局は系列に並ぶソフトトークンなので、画像を含む長い入力ほど、この省メモリ設計の恩恵が大きい。

Gemma 2 からの主な変更点を並べておくと、この世代の狙いが見えやすい。

- **視覚対応の追加**: $4\text{B}$ / $12\text{B}$ / $27\text{B}$ に SigLIP 系エンコーダを統合（$1\text{B}$ はテキストのみ）。
- **長コンテキスト化**: コンテキストを $128\text{K}$ へ拡張（$1\text{B}$ は $32\text{K}$）。
- **注意層の比**: local/global を $1:1$ から $5:1$ へ、ローカルの窓を $4096$ から $1024$ へ縮小し KV キャッシュを削減。
- **正規化の更新**: 注意ロジットの soft-capping を **QK-norm** に置き換え。
- **多言語強化**: Gemini 2.0 系トークナイザと多言語データの増強。

LLaVA 系の VLM では「外付けの視覚エンコーダ」と「既存 LLM」を投影層でつなぐ構図が中心だった。Gemma 3 でも視覚エンコーダ自体は SigLIP を流用するが、視覚トークンは**言語モデルの系列にそのまま並ぶソフトトークン**として扱われ、デコーダ側は局所・全域の注意配置や長コンテキスト化まで含めて「画像入りの長い系列」を効率よく処理できるよう作り込まれている。つまり接続の工夫よりも、**受け手である言語モデルの設計**に重心がある。

押さえておきたいのは、この世代の主役が「派手な新規モジュール」ではなく、**視覚の取り込み方**と**注意層の配置**という二つの設計判断だという点である。以降で順に見ていく。

> 直感: Gemma 3 を一言でいえば「コストが読めて、実機で回る VLM」。視覚も注意も長コンテキストも、すべて「メモリと計算を予測可能に小さく保つ」方向で噛み合っている。

## 視覚の扱い

### 画像を固定数のトークンへ

視覚エンコーダは **SigLIP**（CLIP 損失の変種で学習した Vision Transformer）を用い、$4\text{B}$ / $12\text{B}$ / $27\text{B}$ で**共有**され、言語モデルの学習中は**凍結（frozen）**される。エンコーダは固定解像度 $896 \times 896$ の正方形画像を入力に取り、視覚アシスタント系のタスクで微調整されている。学習時には画像の埋め込みをあらかじめ計算しておくため、言語モデルの学習に視覚エンコーダ分の追加コストはほぼ載らない。

ここで効いてくるのが、**視覚埋め込みを固定サイズの $256$ 本のベクトルへ凝縮する**設計だ。LLaVA のように画像パッチをそのまま全部トークン化すると、解像度を上げるほどトークン数が膨らみ推論コストが跳ね上がる。Gemma 3 は高解像度エンコーダの出力に **$4\times4$ の平均プーリング**をかけ、どの解像度でも出力を $256$ トークンに揃える。これにより「画像 1 枚 = ちょうど $256$ トークン」という会計が常に成り立ち、コンテキスト予算の見積もりが容易になる。

この $256$ 本のベクトルは「**ソフトトークン**」と呼ばれる。テキスト側が語彙表から引いた離散トークンの埋め込みであるのに対し、視覚側は SigLIP の出力（連続値ベクトル）をそのまま埋め込みとして使う。両者は次元を揃えられたうえで同じ系列に並ぶので、デコーダから見れば「画像も言語も区別なく続く一本のトークン列」に映る。語彙を拡張して画像トークンを足すのではなく、連続表現を直接差し込む点が、この方式の素直さである。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="896かける896の画像をSigLIPのViTでパッチ埋め込みし、4かける4の平均プーリングで圧縮して固定256視覚トークンを得る流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="a2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#0e7490"/></marker>
  </defs>
  <text x="360" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">高解像度でもトークン数は一定</text>
  <rect x="34" y="120" width="120" height="120" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <line x1="74" y1="120" x2="74" y2="240" stroke="#0e7490" stroke-width="1"/>
  <line x1="114" y1="120" x2="114" y2="240" stroke="#0e7490" stroke-width="1"/>
  <line x1="34" y1="160" x2="154" y2="160" stroke="#0e7490" stroke-width="1"/>
  <line x1="34" y1="200" x2="154" y2="200" stroke="#0e7490" stroke-width="1"/>
  <text x="94" y="262" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">画像 896 かける 896</text>
  <line x1="156" y1="180" x2="194" y2="180" stroke="#0e7490" stroke-width="2" marker-end="url(#a2)"/>
  <rect x="196" y="135" width="146" height="90" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="269" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">SigLIP ViT</text>
  <text x="269" y="194" text-anchor="middle" font-size="12" fill="#4338ca">パッチ埋め込み</text>
  <line x1="344" y1="180" x2="382" y2="180" stroke="#0e7490" stroke-width="2" marker-end="url(#a2)"/>
  <rect x="384" y="140" width="138" height="80" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="453" y="174" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">4 かける 4</text>
  <text x="453" y="196" text-anchor="middle" font-size="12" fill="#16a34a">平均プーリング</text>
  <line x1="524" y1="180" x2="562" y2="180" stroke="#0e7490" stroke-width="2" marker-end="url(#a2)"/>
  <rect x="564" y="140" width="128" height="80" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="628" y="174" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">256 トークン</text>
  <text x="628" y="196" text-anchor="middle" font-size="12" fill="#0e7490">固定長</text>
</svg><figcaption>パッチ列にプーリングをかけ、解像度によらず <b>256 本</b>に揃える。<b>要点</b>: 画像 1 枚あたりのコンテキスト消費が一定になり、推論コストが読める。</figcaption></figure>

### 可変アスペクト比への Pan &amp; Scan

固定の $896 \times 896$ には弱点がある。横長・縦長など正方形でない画像や、文字の多い高解像度画像を無理に正方形へ潰すと、文字が読めなくなったり小さな物体が消えたりする「歪み」が生じる。これに対処するのが **Pan &amp; Scan（P&amp;S）** という適応的な窓切り（windowing）アルゴリズムだ。

P&amp;S は、画像全体を覆うように**重なりのない等サイズのクロップ**へ分割し、各クロップを $896 \times 896$ にリサイズしてエンコーダへ渡す。こうすると各部分を**ネイティブに近い縦横比・解像度**のまま符号化でき、文字読み取りや細部の認識が大きく改善する。重要なのは、P&amp;S が**必要なときだけ**適用され、**クロップ枚数の上限を制御**できる**推論時のみの最適化**である点だ。速度を優先したいときは無効化できる。発想は LLaVA の柔軟な解像度処理から着想を得ている。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="横長の高解像度画像をPan and Scanで等サイズの重ならないクロップに分割し、各クロップを896かける896にリサイズしてエンコーダに渡す流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="a3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><polygon points="0 0, 9 3, 0 6" fill="#4338ca"/></marker>
  </defs>
  <text x="360" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">正方形に潰さず、分割して読み取る</text>
  <rect x="24" y="120" width="200" height="120" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <line x1="90" y1="120" x2="90" y2="240" stroke="#0e7490" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="157" y1="120" x2="157" y2="240" stroke="#0e7490" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="124" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">横長 / 高解像度</text>
  <text x="124" y="262" text-anchor="middle" font-size="12" fill="#71717a">ネイティブに近い縦横比</text>
  <line x1="226" y1="180" x2="262" y2="180" stroke="#4338ca" stroke-width="2" marker-end="url(#a3)"/>
  <rect x="264" y="138" width="138" height="84" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="333" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Pan and Scan</text>
  <text x="333" y="194" text-anchor="middle" font-size="12" fill="#4338ca">適応的に窓へ分割</text>
  <line x1="404" y1="180" x2="440" y2="180" stroke="#4338ca" stroke-width="2" marker-end="url(#a3)"/>
  <rect x="442" y="124" width="128" height="112" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="454" y="150" width="30" height="30" rx="3" fill="#ffffff" stroke="#0e7490" stroke-width="2"/>
  <rect x="491" y="150" width="30" height="30" rx="3" fill="#ffffff" stroke="#0e7490" stroke-width="2"/>
  <rect x="528" y="150" width="30" height="30" rx="3" fill="#ffffff" stroke="#0e7490" stroke-width="2"/>
  <text x="506" y="206" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">各クロップを</text>
  <text x="506" y="224" text-anchor="middle" font-size="12" fill="#0e7490">896 かける 896 に</text>
  <line x1="572" y1="180" x2="606" y2="180" stroke="#4338ca" stroke-width="2" marker-end="url(#a3)"/>
  <rect x="608" y="140" width="92" height="80" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="654" y="172" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">エンコーダ</text>
  <text x="654" y="192" text-anchor="middle" font-size="11" fill="#16a34a">クロップ毎に</text>
  <text x="654" y="208" text-anchor="middle" font-size="11" fill="#16a34a">256 トークン</text>
</svg><figcaption>P&amp;S は画像を等サイズの窓に分割し、各窓をエンコーダのネイティブ解像度で符号化する。<b>要点</b>: 文字読み取りや細部に強くなる、必要時だけ働く推論時オプション。</figcaption></figure>

なお、視覚エンコーダの**入力解像度そのものも品質に効く**。論文の検証では、入力を $256$ → $448$ → $896$ と上げるほど文書・画像テキスト系の指標が改善する。だが解像度をそのままトークン数に反映させると系列が肥大化してしまう。そこで Gemma 3 は「**高解像度で符号化し、出力は $4\times4$ プーリングで $256$ に圧縮する**」という二段構えを取る。入力では細部を拾い、出力では系列長を一定に保つ、という役割分担だ。

定性的には、文書 VQA や画像中テキストの読み取りといった「文字・細部が効く」タスクで P&amp;S は明確に効く。一方で、正方形に近い画像や速度重視の場面ではオフにできる柔軟さを持つ。固定 $256$ トークンによる「予測しやすいコスト」と、入力解像度 + P&amp;S による「必要なときの高解像度」を両立させているのがこの世代の視覚設計の妙だ。

## 長コンテキストとアーキ

Gemma 3 のもう一つの主眼が、長コンテキストを**メモリ効率よく**扱うことである。コンテキスト長は $128\text{K}$ トークンに達する（ただし $1\text{B}$ は $32\text{K}$）。長コンテキストで最大の問題は、推論時に**KV キャッシュのメモリが爆発**することだ。注意機構は過去の全トークンの key/value を保持するため、系列が長くなるほどメモリ消費が線形に膨らみ、オンデバイス利用の障害になる。

### local と global の交互配置

直感的には、文章を理解するとき多くの判断は「近くの数語〜数十語」で足りる一方、たまに「ずっと前の文脈」を引く必要がある。だとすれば、全ての層がいつも系列全体を見張る必要はない。近場専用の層を多数置き、遠くを見る層を少数だけ混ぜれば、平均的なコストを大きく下げられる——これがローカル/グローバル分業の発想である。

この問題への答えが、**ローカル（窓）注意とグローバル注意の交互配置**である。Transformer の各層を二種類に分ける。

- **ローカル層**: スライディングウィンドウ注意で、各トークンは直近 $1024$ トークンの**窓の中だけ**を参照する。
- **グローバル層**: 通常の全域注意で、**系列全長**（最大 $128\text{K}$）を参照する。

Gemma 3 は、この二種類を $L\!:\!G \approx 5:1$、すなわち**グローバル 1 層につきローカル 5 層**の比で並べる（モデルの最初の層はローカル層から始まる）。KV キャッシュのメモリは「どれだけ遠くまで参照するか」に支配されるため、全層をグローバルにする従来構成に比べ、**長距離を見る層を $1/6$ 程度に絞る**ことでキャッシュを大幅に縮められる。論文の対比でも、グローバルのみの構成が大きなメモリ過剰を生むのに対し、ローカルを増やし窓を $1024$ に保つ構成では過剰がはるかに小さく抑えられることが示されている。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="ローカル層5層ごとにグローバル層1層を挟む交互配置と、ローカル層が1024窓を、グローバル層が系列全長を参照することでKVキャッシュを抑える仕組みの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="360" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">ローカル 5 層 ＋ グローバル 1 層 を反復</text>
  <text x="120" y="52" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">層の並び（下が入力側）</text>
  <g>
    <rect x="60" y="60" width="120" height="20" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="120" y="75" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">グローバル</text>
    <rect x="60" y="84" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="99" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="108" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="123" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="132" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="147" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="156" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="171" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="180" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="195" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="204" width="120" height="20" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="120" y="219" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">グローバル</text>
    <rect x="60" y="228" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="243" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="252" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="267" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
    <rect x="60" y="276" width="120" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="120" y="291" text-anchor="middle" font-size="11" fill="#18181b">ローカル</text>
  </g>
  <line x1="270" y1="120" x2="270" y2="120" stroke="#71717a" stroke-width="0"/>
  <text x="470" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">参照する範囲</text>
  <rect x="300" y="96" width="360" height="22" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="480" y="111" text-anchor="middle" font-size="11" fill="#18181b">系列（最大 128K トークン）</text>
  <rect x="470" y="150" width="80" height="22" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="510" y="165" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">窓 1024</text>
  <text x="300" y="165" font-size="12" font-weight="700" fill="#0e7490">ローカル層:</text>
  <rect x="300" y="196" width="360" height="22" rx="4" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="480" y="211" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">全長を参照</text>
  <text x="300" y="245" font-size="12" font-weight="700" fill="#4338ca">グローバル層:</text>
  <text x="480" y="275" text-anchor="middle" font-size="12" fill="#18181b">KV キャッシュは主にグローバル層が支配</text>
  <text x="480" y="296" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">→ 長距離を見る層を絞ってメモリ削減</text>
</svg><figcaption>長距離を参照するのはグローバル層だけで、ローカル層は 1024 の窓に限定する。<b>要点</b>: 5 対 1 の配置で性能をほぼ保ったまま、長コンテキストの KV キャッシュ膨張を抑える。</figcaption></figure>

論文のアブレーションでは、$L\!:\!G$ の比を $1:1$ から $7:1$ まで変えても**検証パープレキシティへの影響は小さい**ことが示されている。つまり「グローバル層を間引いてもモデルの素の言語能力はほとんど落ちない」のであり、だからこそメモリ側のメリットを安心して取りに行ける、という設計判断になっている。

さらにローカル層の**窓の大きさ**も削れる。スライディングウィンドウを $4096$ から $1024$ 程度まで縮めてもパープレキシティはほとんど悪化しない一方で、KV キャッシュは目に見えて軽くなる。Gemma 2 が比 $1:1$・窓 $4096$ だったのに対し、Gemma 3 が比 $5:1$・窓 $1024$ を採るのは、この「比」と「窓」の二つのつまみを両方とも省メモリ側へ振り切った結果だと読める。実際、全層グローバルの構成では長コンテキスト時にキャッシュ由来のメモリ過剰が大きく膨らむのに対し、ローカル中心の構成ではその過剰がはるかに小さく抑えられる。

### 位置符号化と 128K への伸長

長コンテキストを実際に効かせるため、位置符号化（RoPE）にも手が入っている。**グローバル層の RoPE 基底周波数を $10\text{k}$ から $1\text{M}$ へ引き上げ**、一方で**ローカル層は $10\text{k}$ のまま**に保つ。長距離を担うグローバル層だけ周期を伸ばすことで、遠い位置どうしの関係を破綻なく表現できるようにする発想だ。

学習も二段構えになっている。事前学習は $32\text{K}$ 系列で行い、**事前学習の最後に RoPE を再スケールして $128\text{K}$ まで伸長**する（スケーリング係数はおおよそ $8$ が実用的とされる）。位置補間（positional interpolation）に近い手続きでグローバル層の参照スパンを広げる格好だ。結果としてモデルは $128\text{K}$ までよく一般化するが、それ以上へさらに伸ばそうとすると急速に劣化する、という限界も率直に報告されている。

この長コンテキストは視覚と相性がよい。画像 1 枚が固定 $256$ トークンに収まるため、$128\text{K}$ という予算の中に**多数の画像と長いテキストを同居**させられる。コスト会計が読めること（$256$ トークン/枚）と、長い系列をメモリ効率よく扱えること（local/global 配置）が組み合わさってはじめて、「複数画像＋長文」という現実的な入力が実用域に乗る。視覚設計と長コンテキスト設計は別々の話ではなく、同じ「省メモリ」という土台の上で支え合っている。

ここまでをまとめると、Gemma 3 の長コンテキスト戦略は「**グローバル層を少数に絞ってメモリを節約**」「**その少数のグローバル層だけ RoPE を伸ばして遠距離を担当**」という二つの判断の組み合わせとして読める。注意層の役割分担が、メモリと長距離性能の両取りを可能にしているのである。

## 軽量・多言語・オンデバイス

Gemma 3 の各サイズ（$1\text{B}$ / $4\text{B}$ / $12\text{B}$ / $27\text{B}$）は、スマートフォン・ノート PC・単一の高性能 GPU といった**身近なハードウェアで動かす**ことを設計目標に置く。これを支えるのが二つの工夫だ。

第一に、上で見た**注意層の配置による KV キャッシュ削減**である。オンデバイス推論では重みだけでなく実行時のキャッシュメモリがボトルネックになりやすく、ローカル中心の構成はここに直接効く。第二に、**Quantization Aware Training（QAT）** による低ビット版の提供だ。Gemma 3 は生（bfloat16）のチェックポイントに加え、int4（per-channel / per-block）や切替 fp8 といった量子化形式の重みを公開する。QAT は量子化後の分布に合わせて短いステップ微調整するため、単純な後段量子化よりも精度劣化を抑えやすい。これにより、メモリの限られた環境でも実用的なフットプリントで動かせる。

多言語性も重要な軸だ。Gemini 2.0 と同じトークナイザは**非英語言語により均等にトークンを配分**し、事前学習データにも単言語・対訳データを増やして言語カバレッジを広げている。学習自体は大きな教師モデルからの**知識蒸留**で行われ、小さなモデルでも質の高い分布を学べるよう工夫されている。

蒸留では、教師モデルが各トークン位置で示す確率分布（の代表的なロジット）を採取し、生徒モデルがその分布をなぞるように学ぶ。正解ラベルだけを当てる通常の学習よりも、「教師が次に何を考えていたか」という豊かな手がかりを受け取れるため、限られたパラメータで効率よく能力を引き出せる。事前学習トークン数もサイズに応じて積み増され、視覚とテキストを混ぜたデータで学習される。こうした学習レシピ全体が、軽量なのに実力のあるモデル群を支えている。

サイズ展開にも住み分けが見える。$1\text{B}$ は視覚を持たずコンテキストも $32\text{K}$ に抑えた、最も軽量なテキスト専用モデルで、極端に制約の強い環境を想定する。$4\text{B}$ / $12\text{B}$ / $27\text{B}$ は視覚エンコーダ（凍結・共有）と $128\text{K}$ コンテキストを備え、用途と利用可能なメモリに応じて選べる。視覚エンコーダを三サイズで共有・凍結する設計は、学習・運用の両面でコストを抑えると同時に、サイズ間で視覚の振る舞いを揃える効果もある。

設計思想として一貫しているのは、「派手な新規性よりも、**実機で回る軽さ**を積み上げる」という姿勢だ。視覚の固定 $256$ トークン、ローカル中心の注意、GQA、QAT という要素が、いずれも「コストを予測可能に・小さく保つ」方向で噛み合っている。

## まとめと、読解後に答えたい問い

Gemma 3 は、軽量オープン言語モデルに視覚・長コンテキスト・多言語を統合した世代である。要点を三つに圧縮するなら次のとおりだ。

- **視覚**: SigLIP 系エンコーダで画像を**固定長 $256$ トークン**へ凝縮し、可変アスペクト比には**推論時オプションの Pan &amp; Scan** で対応する。コストを一定に保ちつつ、必要なときだけ高解像度に踏み込む。
- **アーキ・長コンテキスト**: ローカル（窓 $1024$）とグローバルを $L\!:\!G \approx 5:1$ で交互配置し、グローバル層の RoPE を $10\text{k}\to 1\text{M}$ へ伸ばして $128\text{K}$ を実現する。**KV キャッシュ膨張を抑えること**が中心命題。
- **軽量・多言語・オンデバイス**: QAT による低ビット版と均等な多言語トークナイザ、知識蒸留で、身近なハードウェアでの実用を狙う。

読解の確認として、次の問いに自分の言葉で答えられるか試してほしい。

1. なぜ画像をパッチそのままではなく**固定 $256$ トークン**に凝縮するのか。解像度を上げたいときのコスト面のトレードオフはどう変わるか。
2. Pan &amp; Scan が**推論時のみ・必要時のみ**の最適化である利点は何か。どんなタスクで効き、どんなときに切ってよいか。
3. ローカル層とグローバル層は、それぞれ**何を**参照し、**KV キャッシュ**のどこに効くのか。$5:1$ という比を選べるのは、アブレーションのどんな観察に支えられているか。
4. $128\text{K}$ を実現するために RoPE をグローバル層だけ伸ばすのはなぜか。ローカル層の基底周波数を据え置く理由は何か。
5. $1\text{B}$ だけが視覚非対応かつ $32\text{K}$ である事実は、各サイズの**用途の住み分け**についてどんな示唆を与えるか。
6. GQA・local/global 配置・QAT は、いずれも「メモリを抑える」工夫として並べられる。実機での推論を想定したとき、これらはどの局面（重み・KV キャッシュ・帯域）にそれぞれ効くか。

これらに答えられれば、Gemma 3 を「軽量・視覚対応・長コンテキスト」という三つの軸で説明できるようになっているはずだ。次は実装演習で、固定 $256$ トークン化や local/global の挙動を小さく再現しながら、設計判断が推論コストにどう跳ね返るかを手で確かめていきたい。
