# Qwen-VL — Versatile Vision-Language Model（Position-aware Adapter・3段階学習・grounding/OCR）

Qwen-VL（arXiv:2308.12966, Alibaba Group）は、LLM基盤に Qwen-7B を据えた汎用ビジョン言語モデル（LVLM）です。LLaVA系が「ViT特徴を MLP で言語空間へ**射影**してそのまま流し込む」のに対し、Qwen-VL は**学習可能な256個のクエリによる単層 cross-attention** で視覚特徴を 256 トークンに**圧縮**するコネクタ（Position-aware Vision-Language Adapter）を採用します。さらに、低解像度→高解像度→対話 の**3段階学習**と、bbox を座標テキストトークンで表す **grounding / OCR** を一つのモデルに統合した点が特徴です。本ページでは、これらを LLaVA系と対比しながら整理します。

---

## 全体像（まず一枚で）

Qwen-VL は 3 つの部品からなります。視覚エンコーダ（OpenCLIP の **ViT-bigG/14** 由来の ViT）、コネクタ（**Position-aware Vision-Language Adapter**）、そして言語モデル（**Qwen-7B**）です。画像は固定解像度（最終的に $448 \times 448$）にリサイズされ、stride 14 のパッチに分割されて多数の ViT 特徴になります。Adapter はこの可変長の特徴列を**固定長 256 トークン**へ圧縮し、`<img>` … `</img>` の特殊トークンで囲んでテキストトークン列に差し込み、Qwen-7B が自己回帰でテキスト（説明・回答・座標）を生成します。

<figure class="lec-fig"><svg viewBox="0 0 860 300" role="img" aria-label="画像をViTで符号化し、Position-aware AdapterでViT特徴を256トークンに圧縮してQwen-7B LLMへ入力する全体構成図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="15" y="110" width="90" height="70" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="60" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">画像</text>
<text x="60" y="160" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">448×448</text>
<line x1="107" y1="145" x2="128" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="135,145 125,140 125,150" fill="#71717a"/>
<rect x="137" y="105" width="120" height="80" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="197" y="138" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">ViT (OpenCLIP</text>
<text x="197" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">ViT-bigG/14)</text>
<text x="197" y="173" text-anchor="middle" font-size="11" font-weight="700" fill="#155e75">stride14 パッチ</text>
<line x1="259" y1="145" x2="288" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="295,145 285,140 285,150" fill="#71717a"/>
<rect x="297" y="95" width="160" height="100" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="377" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">Position-aware</text>
<text x="377" y="144" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">VL Adapter</text>
<text x="377" y="162" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">256クエリ cross-attn</text>
<text x="377" y="180" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">+2D絶対位置</text>
<line x1="459" y1="145" x2="494" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="501,145 491,140 491,150" fill="#71717a"/>
<rect x="503" y="110" width="135" height="70" rx="8" fill="#ecfeff" stroke="#06b6d4" stroke-width="2"/>
<text x="570" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">&lt;img&gt; 256トークン</text>
<text x="570" y="160" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">&lt;/img&gt;</text>
<line x1="640" y1="145" x2="671" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="678,145 668,140 668,150" fill="#71717a"/>
<rect x="680" y="95" width="165" height="100" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="762" y="132" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">Qwen-7B (LLM)</text>
<text x="762" y="152" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">自己回帰で</text>
<text x="762" y="170" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">テキスト/座標生成</text>
</svg><figcaption>ViT特徴は <b>Adapter で256トークンに圧縮</b>され、<code>&lt;img&gt;…&lt;/img&gt;</code> で囲んで LLM に渡されます。<b>視覚を「固定数の少数トークン」に畳む</b>のが LLaVA系との最大の違いです。</figcaption></figure>

論文 Table 1 のパラメータ規模はおおむね、ViT ≈ 1.9B、VL Adapter ≈ 0.08B、LLM ≈ 7.7B、合計 ≈ 9.6B です。Adapter が**極めて軽量**（数千万パラメータ）でありながら、視覚側の「長さ」を一定に保つ役割を担う点に注目してください。

---

## コネクタ：Position-aware Vision-Language Adapter

LLaVA系のコネクタ（MLP projector）は、ViT が出す各パッチ特徴を**位置ごとに**言語埋め込み次元へ射影します。トークン数は ViT のパッチ数のまま保たれるため、解像度を上げるほど LLM に入る視覚トークンが増え、系列長と計算コストが膨らみます。

Qwen-VL の Adapter はここを設計から変えます。**ランダム初期化された単層 cross-attention** モジュールを置き、**学習可能な256個のベクトル（クエリ）** を用意します。クエリを query、ViT 特徴を key/value として cross-attention を一度かけることで、可変長の視覚特徴を**常に256トークン**へ圧縮します。これにより、解像度が上がっても LLM 側の視覚トークン数は 256 のまま一定に保たれます。

圧縮には情報損失、とりわけ**空間位置の損失**が付きまといます。Qwen-VL はこれを抑えるため、cross-attention の query–key ペアに **2D 絶対位置エンコーディング**を組み込みます。「どのクエリが画像のどのあたりを見たか」を保つことが、後述の grounding（座標出力）を成立させる前提になっています。

<figure class="lec-fig"><svg viewBox="0 0 780 380" role="img" aria-label="LLaVAのMLP projectorはトークン数を保存し、Qwen-VLのAdapterはcross-attentionで256トークンに圧縮することを対比した図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="20" y="36" font-size="14" font-weight="700" fill="#3730a3">LLaVA系: MLP projector（位置ごと射影・トークン数を保存）</text>
<rect x="30" y="60" width="120" height="55" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="90" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ViT特徴</text>
<text x="90" y="102" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">N 個（多数）</text>
<line x1="152" y1="87" x2="198" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="205,87 195,82 195,92" fill="#71717a"/>
<rect x="207" y="60" width="120" height="55" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="267" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">2層 MLP</text>
<text x="267" y="102" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">（位置ごと射影）</text>
<line x1="329" y1="87" x2="375" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="382,87 372,82 372,92" fill="#71717a"/>
<rect x="384" y="60" width="150" height="55" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="459" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">N トークン</text>
<text x="459" y="102" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">（入力と同数）</text>
<line x1="536" y1="87" x2="572" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="579,87 569,82 569,92" fill="#71717a"/>
<rect x="581" y="60" width="120" height="55" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="641" y="93" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">LLM</text>
<line x1="20" y1="160" x2="760" y2="160" stroke="#e4e4e7" stroke-width="2"/>
<text x="20" y="196" font-size="14" font-weight="700" fill="#155e75">Qwen-VL: Position-aware Adapter（cross-attnで256に圧縮）</text>
<rect x="30" y="220" width="125" height="48" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="92" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ViT特徴</text>
<text x="92" y="259" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">多数（key/value）</text>
<rect x="30" y="292" width="125" height="48" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="92" y="314" text-anchor="middle" font-size="12" font-weight="700" fill="#991b1b">256 学習可能クエリ</text>
<text x="92" y="331" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">（query）</text>
<line x1="157" y1="244" x2="203" y2="270" stroke="#71717a" stroke-width="2"/><polygon points="209,274 198,272 203,263" fill="#71717a"/>
<line x1="157" y1="316" x2="203" y2="290" stroke="#71717a" stroke-width="2"/><polygon points="209,286 203,297 198,288" fill="#71717a"/>
<rect x="211" y="250" width="150" height="60" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="286" y="276" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">単層 cross-attn</text>
<text x="286" y="296" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">+2D絶対位置</text>
<line x1="363" y1="280" x2="399" y2="280" stroke="#71717a" stroke-width="2"/><polygon points="406,280 396,275 396,285" fill="#71717a"/>
<rect x="408" y="252" width="150" height="55" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="483" y="276" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">256 トークン</text>
<text x="483" y="294" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">（常に固定長）</text>
<line x1="560" y1="280" x2="596" y2="280" stroke="#71717a" stroke-width="2"/><polygon points="603,280 593,275 593,285" fill="#71717a"/>
<rect x="605" y="252" width="120" height="55" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="665" y="285" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">LLM</text>
</svg><figcaption>MLP は<b>トークン数を保存</b>（解像度↑で系列長↑）。Adapter は cross-attn で<b>常に256トークンへ圧縮</b>し、<b>2D絶対位置</b>で空間情報を残します。</figcaption></figure>

| 観点 | LLaVA系 MLP projector | Qwen-VL Position-aware Adapter |
| --- | --- | --- |
| 演算 | 位置ごとの線形射影（次元変換） | 単層 cross-attention（要約・圧縮） |
| 視覚トークン数 | ViT のパッチ数に比例（可変） | 常に 256（固定） |
| 空間情報 | パッチの並びがそのまま保持 | 2D 絶対位置エンコーディングで補償 |
| パラメータ | 小（数百万） | 小（≈0.08B、クエリ＋単層attn） |
| 解像度を上げると | LLM 系列長が増えコスト増 | 視覚トークン数は不変 |

> 直感：MLP は「すべてのパッチをそのまま言語に翻訳する」、Adapter は「256個の質問者（クエリ）が画像を読み、256行の要約に畳む」。前者は密で位置に忠実、後者は固定コストで長系列に強い、というトレードオフです。

---

## 3段階学習

Qwen-VL は学習を 3 段階に分け、「何を凍結し、何を解凍し、どの解像度・どのデータで学ぶか」を段階ごとに切り替えます。

- **Stage 1：事前学習（LLM 凍結・低解像度 224）。** Web 由来の弱教師 image-text ペア（クリーニング後 ≈1.4B、英語約77%・中国語約23%）で、**ViT と Adapter のみ**を学習します。LLM（Qwen-7B）は凍結。目的は視覚側を言語空間へ大まかに整列させること。入力は $224 \times 224$。
- **Stage 2：マルチタスク事前学習（全解凍・高解像度 448）。** 解像度を 224 → **$448 \times 448$** に引き上げ、**モデル全体を解凍**して 7 種のタスクを同時学習します：captioning / VQA / grounding / referring grounding / grounded captioning / **OCR・text-reading** / pure-text autoregression。interleaved（画像とテキストが交互に並ぶ）系列も使い、grounding・OCR といった**細粒度能力**をここで獲得します。
- **Stage 3：教師ありファインチューニング（SFT → Qwen-VL-Chat）。** 指示追従・対話能力を付与する段階で、**視覚エンコーダを凍結**し **LLM と Adapter** を最適化します。複数画像対話や grounding を含む対話データ、純テキスト対話を混ぜ、得られるのが対話モデル **Qwen-VL-Chat** です。

ここで効くのが Stage 1 と Stage 3 の**凍結対象の入れ替え**です。Stage 1 は「視覚を言語へ合わせる」ため視覚側（ViT+Adapter）を動かして LLM を守り、Stage 3 は「指示に従って話す」ため言語側（LLM+Adapter）を動かして視覚エンコーダを守ります。Stage 2 だけが全解凍で、能力の中核を一気に作り込みます。

<figure class="lec-fig"><svg viewBox="0 0 780 330" role="img" aria-label="Qwen-VLの3段階学習を示す図。Stage1はLLM凍結で低解像度、Stage2は全解凍で高解像度マルチタスク、Stage3はViT凍結で対話SFT" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="18" y="40" width="232" height="270" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="2"/>
<text x="134" y="64" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">Stage 1: 事前学習</text>
<rect x="43" y="78" width="182" height="40" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/><text x="134" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#52525b">Qwen-7B LLM（凍結）</text>
<rect x="43" y="126" width="182" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="134" y="151" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Adapter（学習）</text>
<rect x="43" y="174" width="182" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="134" y="199" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ViT（学習）</text>
<text x="134" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">解像度 224×224</text>
<text x="134" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">弱教師 image-textペア</text>
<text x="134" y="290" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">（視覚を言語へ整列）</text>
<rect x="274" y="40" width="232" height="270" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="2"/>
<text x="390" y="64" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">Stage 2: マルチタスク</text>
<rect x="299" y="78" width="182" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="390" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">Qwen-7B LLM（学習）</text>
<rect x="299" y="126" width="182" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="390" y="151" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Adapter（学習）</text>
<rect x="299" y="174" width="182" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="390" y="199" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ViT（学習）</text>
<text x="390" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">解像度 448×448（全解凍）</text>
<text x="390" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">VQA/grounding/OCR/</text>
<text x="390" y="290" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">referring/text-reading</text>
<rect x="530" y="40" width="232" height="270" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="2"/>
<text x="646" y="64" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">Stage 3: SFT</text>
<rect x="555" y="78" width="182" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="646" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">Qwen-7B LLM（学習）</text>
<rect x="555" y="126" width="182" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="646" y="151" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Adapter（学習）</text>
<rect x="555" y="174" width="182" height="40" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/><text x="646" y="199" text-anchor="middle" font-size="12" font-weight="700" fill="#52525b">ViT（凍結）</text>
<text x="646" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">→ Qwen-VL-Chat</text>
<text x="646" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">複数画像対話・grounding対話</text>
<text x="646" y="290" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">＋純テキスト対話を混合</text>
</svg><figcaption>凍結対象が段階で入れ替わります。<b>Stage1=視覚側を学習</b>（LLM保護）、<b>Stage2=全解凍で能力構築</b>、<b>Stage3=視覚を凍結し対話化</b>。解像度は 224→448 へ。</figcaption></figure>

| 段階 | 解像度 | 凍結 | 学習 | 主データ／目的 |
| --- | --- | --- | --- | --- |
| Stage 1 事前学習 | $224 \times 224$ | LLM | ViT + Adapter | 弱教師 image-text、視覚↔言語の整列 |
| Stage 2 マルチタスク | $448 \times 448$ | なし（全解凍） | 全体 | captioning/VQA/grounding/OCR/referring |
| Stage 3 SFT | $448 \times 448$ | ViT | LLM + Adapter | 指示・対話、Qwen-VL-Chat 化 |

---

## grounding と OCR

Qwen-VL の「versatile（万能）」を象徴するのが、**位置を言葉で表す**設計です。bbox 用に新しい座標語彙を足すのではなく、座標を**普通のテキストとして**書き下します。具体的には、各座標を $[0, 1000]$ の範囲に正規化し、`(X_左上, Y_左上),(X_右下, Y_右下)` という文字列にして、これを `<box>` … `</box>` で囲みます。さらに、その bbox が**どの語句を指すか**を結びつけるために `<ref>` … `</ref>` を導入し、参照表現（referring expression）を表します。

この設計の利点は、検出を**生成タスクとして**統一できることです。LLM は座標も含めて自己回帰でトークンを吐くだけなので、「説明 → 参照 → 座標」を一つの系列で混在させられ、grounded captioning（語句に bbox を付けた説明）や referring grounding（説明から対象を特定）が自然に表現できます。OCR・text-reading（合成文書 SynthDoG や Web の PDF/HTML 由来データなどで学習）も同じ枠組みに統合され、`<img>…</img>` で囲んだ視覚トークンの並びと相まって、画像中の文字読み取りと文字配置の理解が一つのモデルで完結します。

<figure class="lec-fig"><svg viewBox="0 0 720 280" role="img" aria-label="画像中の対象のbboxを正規化した座標テキストトークンで表現し、refとboxの特殊トークンで参照表現と座標を記述する図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="30" y="50" width="210" height="190" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="135" y="44" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">入力画像（左上原点・[0,1000]正規化）</text>
<rect x="92" y="100" width="95" height="80" rx="4" fill="none" stroke="#dc2626" stroke-width="3"/>
<text x="139" y="145" text-anchor="middle" font-size="12" font-weight="700" fill="#991b1b">対象</text>
<circle cx="92" cy="100" r="4" fill="#dc2626"/><text x="84" y="96" text-anchor="end" font-size="10" font-weight="700" fill="#991b1b">(x1,y1)</text>
<circle cx="187" cy="180" r="4" fill="#dc2626"/><text x="195" y="195" font-size="10" font-weight="700" fill="#991b1b">(x2,y2)</text>
<line x1="242" y1="140" x2="288" y2="140" stroke="#71717a" stroke-width="2"/><polygon points="295,140 285,135 285,145" fill="#71717a"/>
<rect x="300" y="80" width="392" height="120" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="320" y="112" font-size="14" font-weight="700" fill="#3730a3">&lt;ref&gt;その人&lt;/ref&gt;</text>
<text x="320" y="142" font-size="14" font-weight="700" fill="#3730a3">&lt;box&gt;(x1,y1),(x2,y2)&lt;/box&gt;</text>
<text x="320" y="172" font-size="11" font-weight="700" fill="#71717a">座標は[0,1000]に正規化し「テキスト」としてトークン化</text>
<text x="320" y="189" font-size="11" font-weight="700" fill="#71717a">（専用の座標語彙は追加しない）</text>
</svg><figcaption>bbox は <b><code>&lt;box&gt;(x1,y1),(x2,y2)&lt;/box&gt;</code></b> という<b>座標テキスト</b>で、参照対象は <b><code>&lt;ref&gt;…&lt;/ref&gt;</code></b> で表現。検出が<b>生成タスク</b>に溶け込みます。</figcaption></figure>

> Adapter の 2D 絶対位置エンコーディングと、この座標トークン表現は表裏一体です。圧縮しても位置が残るからこそ、LLM が `<box>` の中身として妥当な座標を生成できます。

---

## 実装注記（Hugging Face）

> 以下は HF モデルカード由来の使い方の要点で、論文の主張ではありません。`Qwen/Qwen-VL-Chat` は **custom code（レガシー）** で配布されており、`trust_remote_code=True` が必須です。推論は専用の `model.chat()` API、入力は `tokenizer.from_list_format([...])` で画像とテキストを混在指定します。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True
).eval()

# 画像とテキストを混在させてクエリを構築（<img>…</img> は内部で付与される）
query = tokenizer.from_list_format([
    {"image": "https://example.com/demo.jpg"},
    {"text": "この画像には何が写っていますか？"},
])
response, history = model.chat(tokenizer, query=query, history=None)

# grounding を促すと <ref>…</ref><box>(x1,y1),(x2,y2)</box> 形式で座標が返り得る
# tokenizer.draw_bbox_on_latest_picture(response, history) で可視化できる
```

LLaVA系が標準の `AutoProcessor` + `generate()` に寄せていくのと違い、Qwen-VL（初代）は独自 API・独自トークナイザ拡張に依存します。再現実装やパイプライン組み込みの際は、この「custom code 前提」を最初に押さえておくとつまずきません。

---

## まとめと、読解後に答えたい問い

Qwen-VL の核は 3 点に集約できます。第一に、**Position-aware Vision-Language Adapter** が cross-attention で視覚特徴を 256 トークンに圧縮し、2D 絶対位置で空間情報を残すこと（LLaVA の MLP 射影との根本的な違い）。第二に、**3段階学習**で凍結対象と解像度を切り替え、Stage 2 の全解凍・高解像度マルチタスクで grounding/OCR を作り込むこと。第三に、bbox を `<box>(x1,y1),(x2,y2)</box>`・`<ref>` の**座標テキストトークン**で表し、検出を生成タスクに統合したこと。これらが「versatile（理解・局在化・文字読み取り）」を一つのモデルで両立させています。

読解後に自分の言葉で答えられるか確認したい問い：

1. MLP projector と cross-attention Adapter は、視覚トークン数・解像度スケーリング・空間情報の保持の観点でどう違うか。なぜ Qwen-VL は 2D 絶対位置を Adapter に組み込む必要があったか。
2. 3 段階で「凍結対象」が Stage 1（LLM）→ Stage 2（なし）→ Stage 3（ViT）と入れ替わるのはなぜか。各段階が何を最適化しているか。
3. bbox を専用語彙ではなく $[0, 1000]$ 正規化のテキストとして書く設計の利点と限界は何か。grounded captioning と referring grounding はこの表現でどう実現されるか。
