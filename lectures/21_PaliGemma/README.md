# PaliGemma — A versatile 3B VLM for transfer（SigLIP＋Gemma）

PaliGemma は、すでに学んだ **SigLIP**（02章）の視覚エンコーダと、Gemma-2B 言語モデルを **線形 projector** でつないだ、合計約 3B パラメータの開いた VLM である。系譜としては **PaLI**（20章）の流れを汲みつつ、視覚側を SigLIP、言語側を Gemma に置き換えた構成にあたる。最大の特徴は「チャットの完成形」を目指すのではなく、**多様な下流タスクへ転移（fine-tune）しやすいベースモデル**として設計されている点である。本章では (1) 全体構成、(2) prefix-LM のマスク設計、(3) 転移前提という設計思想、(4) 学習段階、の順に読み解く。

## 全体像（まず一枚で）

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="PaliGemmaの全体構成。画像をSigLIP視覚エンコーダで符号化し線形projectorでGemmaの埋め込み次元へ写像し、テキストトークンと結合してGemma-2Bデコーダに入力する図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="20" y="55" width="84" height="80" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="62" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">画像</text>
  <text x="62" y="110" text-anchor="middle" font-size="11" fill="#18181b">224/448/896</text>

  <line x1="104" y1="95" x2="116" y2="95" stroke="#71717a" stroke-width="2"/>
  <polygon points="116,91 124,95 116,99" fill="#71717a"/>

  <rect x="124" y="60" width="116" height="70" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="182" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">SigLIP-So400m</text>
  <text x="182" y="110" text-anchor="middle" font-size="11" fill="#18181b">視覚エンコーダ</text>

  <line x1="240" y1="95" x2="252" y2="95" stroke="#71717a" stroke-width="2"/>
  <polygon points="252,91 260,95 252,99" fill="#71717a"/>

  <rect x="260" y="60" width="80" height="70" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="300" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">線形</text>
  <text x="300" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">projector</text>

  <polyline points="340,95 368,95 368,72 384,72" fill="none" stroke="#71717a" stroke-width="2"/>
  <polygon points="380,68 388,72 380,76" fill="#71717a"/>

  <rect x="388" y="46" width="58" height="15" rx="2" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
  <text x="417" y="58" text-anchor="middle" font-size="10" fill="#18181b">img</text>
  <rect x="388" y="64" width="58" height="15" rx="2" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
  <text x="417" y="76" text-anchor="middle" font-size="10" fill="#18181b">img</text>
  <rect x="388" y="82" width="58" height="15" rx="2" fill="#cffafe" stroke="#0e7490" stroke-width="1.5"/>
  <text x="417" y="94" text-anchor="middle" font-size="10" fill="#18181b">img</text>
  <rect x="388" y="104" width="58" height="15" rx="2" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/>
  <text x="417" y="116" text-anchor="middle" font-size="10" fill="#18181b">BOS</text>
  <rect x="388" y="124" width="58" height="15" rx="2" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="417" y="136" text-anchor="middle" font-size="10" fill="#18181b">prefix</text>
  <rect x="388" y="142" width="58" height="15" rx="2" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="417" y="154" text-anchor="middle" font-size="10" fill="#18181b">prefix</text>
  <rect x="388" y="164" width="58" height="15" rx="2" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/>
  <text x="417" y="176" text-anchor="middle" font-size="10" fill="#18181b">SEP</text>
  <rect x="388" y="186" width="58" height="15" rx="2" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/>
  <text x="417" y="198" text-anchor="middle" font-size="10" fill="#18181b">suffix</text>
  <rect x="388" y="204" width="58" height="15" rx="2" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/>
  <text x="417" y="216" text-anchor="middle" font-size="10" fill="#18181b">suffix</text>
  <text x="417" y="236" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">結合トークン列</text>

  <line x1="448" y1="150" x2="478" y2="150" stroke="#71717a" stroke-width="2"/>
  <polygon points="478,146 486,150 478,154" fill="#71717a"/>

  <rect x="486" y="70" width="140" height="180" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="556" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Gemma-2B</text>
  <text x="556" y="172" text-anchor="middle" font-size="11" fill="#18181b">Transformer Decoder</text>

  <line x1="626" y1="160" x2="630" y2="160" stroke="#71717a" stroke-width="2"/>
  <polygon points="630,156 638,160 630,164" fill="#71717a"/>

  <rect x="640" y="115" width="70" height="90" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="675" y="155" text-anchor="middle" font-size="11" fill="#18181b">テキスト</text>
  <text x="675" y="173" text-anchor="middle" font-size="11" fill="#18181b">出力（答え）</text>

  <rect x="20" y="255" width="120" height="56" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="80" y="280" text-anchor="middle" font-size="11" fill="#18181b">「Where is</text>
  <text x="80" y="298" text-anchor="middle" font-size="11" fill="#18181b">it resting?」</text>

  <line x1="140" y1="283" x2="160" y2="283" stroke="#71717a" stroke-width="2"/>
  <polygon points="160,279 168,283 160,287" fill="#71717a"/>

  <rect x="170" y="257" width="96" height="52" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="218" y="280" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">Sentence</text>
  <text x="218" y="298" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">Piece</text>

  <polyline points="266,283 366,283 366,150 384,150" fill="none" stroke="#71717a" stroke-width="2"/>
  <polygon points="380,146 388,150 380,154" fill="#71717a"/>
</svg><figcaption><b>要点</b>: 画像はSigLIPで符号化し線形projectorでGemmaの埋め込み次元へ写像、テキストと結合して1本の系列としてGemmaに入力する。</figcaption></figure>

PaliGemma は3つの部品からなる。

- **視覚エンコーダ: SigLIP-So400m**。形状最適化（shape-optimized）された ViT-So400m（約 400M）で、sigmoid 損失で対照学習済み。小型ながら強い。
- **線形 projector**。SigLIP が出す画像トークンを Gemma の語彙埋め込みと同じ次元へ写す **1層の線形変換**（ゼロ初期化）。MLP など複雑な接続を試しても明確な利得がなかったため、最も単純な線形を採用している。
- **言語モデル: Gemma-2B**。デコーダのみ（decoder-only）の自己回帰 LLM（v1.0 の素の事前学習済みチェックポイント）。

入力の流れはこうだ。画像は **224 / 448 / 896 px** の正方形にリサイズされ、SigLIP を通すとそれぞれ **256 / 1024 / 4096 個**の画像トークンになる。テキストは Gemma の SentencePiece トークナイザで $N_{\text{txt}}$ 個のトークンになる。両者を次の順で1本の系列に並べ、Gemma に入力する。

$$ \texttt{[image tokens..., BOS, prefix..., SEP, suffix..., EOS, PAD...]} $$

ここで **prefix（接頭辞）はタスク指示や質問**、**suffix（接尾辞）はモデルが自己回帰で生成する答え**である。`BOS` がテキストの開始を示し、改行文字 `\n` を `SEP` として（接頭辞・接尾辞と結合されないよう単独で）トークン化する。画像トークンを先頭に置くことで、特別な位置マーカーなしでも「画像 → テキスト」という素直な解釈になる。

## prefix-LM のマスク設計

PaliGemma の注意マスクは、純粋な因果（causal）LM ではなく **prefix-LM** である。すなわち、

- **画像トークン＋接頭辞（prefix）には全注意（双方向）** を許す。
- **接尾辞（suffix）は因果（autoregressive）** にマスクする。`PAD` も因果側に含む。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="prefix-LMの注意マスク。行がクエリ、列がキー。画像と接頭辞からなる左上のブロックは全面参照可、接尾辞は下三角の因果マスクになることを示す7行7列の表" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="308" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">画像＋接頭辞: 双方向（全注意）</text>
  <text x="427" y="40" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">接尾辞: 因果</text>

  <text x="257" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">I1</text>
  <text x="291" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">I2</text>
  <text x="325" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">P1</text>
  <text x="359" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">P2</text>
  <text x="393" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">S1</text>
  <text x="427" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">S2</text>
  <text x="461" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">S3</text>

  <text x="232" y="101" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">I1</text>
  <text x="232" y="135" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">I2</text>
  <text x="232" y="169" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">P1</text>
  <text x="232" y="203" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">P2</text>
  <text x="232" y="237" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">S1</text>
  <text x="232" y="271" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">S2</text>
  <text x="232" y="305" text-anchor="end" font-size="12" font-weight="700" fill="#18181b">S3</text>

  <rect x="240" y="80" width="136" height="238" fill="#dcfce7"/>
  <rect x="376" y="216" width="34" height="34" fill="#dcfce7"/>
  <rect x="376" y="250" width="68" height="34" fill="#dcfce7"/>
  <rect x="376" y="284" width="102" height="34" fill="#dcfce7"/>
  <rect x="376" y="80" width="102" height="136" fill="#fee2e2"/>
  <rect x="410" y="216" width="68" height="34" fill="#fee2e2"/>
  <rect x="444" y="250" width="34" height="34" fill="#fee2e2"/>

  <line x1="240" y1="80" x2="478" y2="80" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="114" x2="478" y2="114" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="148" x2="478" y2="148" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="182" x2="478" y2="182" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="250" x2="478" y2="250" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="284" x2="478" y2="284" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="318" x2="478" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="240" y1="80" x2="240" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="274" y1="80" x2="274" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="308" y1="80" x2="308" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="342" y1="80" x2="342" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="410" y1="80" x2="410" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="444" y1="80" x2="444" y2="318" stroke="#e4e4e7" stroke-width="1"/>
  <line x1="478" y1="80" x2="478" y2="318" stroke="#e4e4e7" stroke-width="1"/>

  <line x1="376" y1="80" x2="376" y2="318" stroke="#4338ca" stroke-width="2" stroke-dasharray="5 3"/>
  <line x1="240" y1="216" x2="478" y2="216" stroke="#4338ca" stroke-width="2" stroke-dasharray="5 3"/>

  <rect x="520" y="118" width="18" height="18" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="546" y="132" font-size="12" fill="#18181b">参照できる</text>
  <rect x="520" y="148" width="18" height="18" rx="3" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="546" y="162" font-size="12" fill="#18181b">マスク（参照不可）</text>
  <text x="520" y="198" font-size="12" fill="#71717a">行=クエリ、列=キー</text>
  <text x="520" y="220" font-size="12" fill="#71717a">左上ブロックは双方向</text>
  <text x="520" y="240" font-size="12" fill="#71717a">右下は因果（下三角）</text>
</svg><figcaption><b>要点</b>: 画像＋接頭辞は双方向（左上ブロックが全面参照可）、接尾辞は因果（下三角）。行がクエリ、列がキー。</figcaption></figure>

直感的には、画像トークンが「先回りして（lookahead）」接頭辞＝クエリを参照できるため、**画像表現を質問の文脈に合わせて更新**できる。小型モデルでも入力全体を双方向に使い切ることで容量を最大化する狙いがある（論文の ablation でも、画像・接頭辞まで因果にする純 AR より prefix-LM の方が明確に良い）。

式で書くと、系列長を $n$、画像＋接頭辞の末尾位置を $p$ とすると、クエリ位置 $i$ がキー位置 $j$ を参照できる条件は

$$ m_{ij} = \mathbb{1}\!\left[\, j \le \max(p,\, i) \,\right] $$

と一行で表せる。$i \le p$（画像・接頭辞側）なら $j \le p$ の全体を、$i > p$（接尾辞側）なら $j \le i$ という因果範囲を参照する。なお系列長 $N_{\text{txt}}$ と言うときは通常 **接頭辞と接尾辞の合計**を指し、画像トークンは数えない。

## 転移のためのベースモデルという設計思想

近年の多くの VLM は instruction tuning / chat tuning を施して「そのまま使える」最終形を目指す。PaliGemma の狙いはそこではない。**「0-shot で便利」よりも「fine-tune でよく伸びる」**ことを優先した、転移用のベースモデルである。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="PaliGemmaベースモデルがfine-tuneを通じてキャプション生成、VQA、物体検出、セグメンテーション、OCR、リモートセンシングなど多様な下流タスクへ転移することを示す扇形の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="30" y="130" width="170" height="100" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="115" y="168" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">PaliGemma</text>
  <text x="115" y="190" text-anchor="middle" font-size="13" fill="#18181b">ベースモデル</text>
  <text x="115" y="210" text-anchor="middle" font-size="11" fill="#71717a">（事前学習済み）</text>

  <line x1="200" y1="180" x2="262" y2="180" stroke="#71717a" stroke-width="2"/>
  <polygon points="262,176 270,180 262,184" fill="#71717a"/>

  <rect x="270" y="160" width="150" height="40" rx="20" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="345" y="185" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">fine-tune（Stage3）</text>

  <line x1="420" y1="180" x2="462" y2="42" stroke="#71717a" stroke-width="2"/>
  <polygon points="455,40 466,42 459,51" fill="#71717a"/>
  <line x1="420" y1="180" x2="462" y2="92" stroke="#71717a" stroke-width="2"/>
  <polygon points="455,89 466,92 458,100" fill="#71717a"/>
  <line x1="420" y1="180" x2="462" y2="142" stroke="#71717a" stroke-width="2"/>
  <polygon points="455,140 466,142 457,150" fill="#71717a"/>
  <line x1="420" y1="180" x2="462" y2="192" stroke="#71717a" stroke-width="2"/>
  <polygon points="455,188 466,192 456,196" fill="#71717a"/>
  <line x1="420" y1="180" x2="462" y2="242" stroke="#71717a" stroke-width="2"/>
  <polygon points="454,236 466,242 456,246" fill="#71717a"/>
  <line x1="420" y1="180" x2="462" y2="292" stroke="#71717a" stroke-width="2"/>
  <polygon points="454,286 466,292 455,294" fill="#71717a"/>

  <rect x="470" y="22" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="47" text-anchor="middle" font-size="13" fill="#18181b">画像キャプション生成</text>
  <rect x="470" y="72" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="97" text-anchor="middle" font-size="13" fill="#18181b">視覚質問応答（VQA）</text>
  <rect x="470" y="122" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="147" text-anchor="middle" font-size="13" fill="#18181b">物体検出（detect）</text>
  <rect x="470" y="172" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="197" text-anchor="middle" font-size="13" fill="#18181b">参照表現セグメンテーション</text>
  <rect x="470" y="222" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="247" text-anchor="middle" font-size="13" fill="#18181b">OCR・文書・図表理解</text>
  <rect x="470" y="272" width="232" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="586" y="297" text-anchor="middle" font-size="13" fill="#18181b">リモートセンシング・動画</text>
</svg><figcaption><b>要点</b>: チャット完成形ではなく、fine-tuneで多様な下流タスクへ転移するベースモデルとして設計される。</figcaption></figure>

この思想は具体的な選択に表れている。

- **事前学習タスクは「スキルの寄せ集め」として設計**。captioning, OCR, VQA, detection, segmentation など多様なタスクを混ぜ、各タスクに固有の接頭辞を付ける。転移時（Stage3）には「どのスキルを使い、どの出力構文に従うか」をモデルが選び直すだけで済む。
- **大型商用 VLM の出力に依存しない**。LLaVA 系が GPT-4 生成の instruction データを使うのと異なり、PaliGemma の事前学習タスクはどれも大型商用 VLM の出力ではない。
- **転移は単純なレシピ**。Stage3 では全パラメータを fine-tune し、調整するハイパラは解像度・エポック数・学習率など少数に絞れる。chat ではないが、複数タスクを同時に転移する「mix」チェックポイントも提供しており、instruction tuning への一歩でもある。

構造を単純な「画像＋テキスト入力 → テキスト出力」API に保つことで、検出やセグメンテーションのような構造的出力も、座標やマスクを**テキスト（特殊トークン）に変換**して同じ枠組みで扱える。実際、語彙には正規化座標用の **1024 個の位置トークン**（`<loc0000>`〜`<loc1023>`）と、参照表現セグメンテーション用の **128 個のマスクトークン**（`<seg000>`〜`<seg127>`）が追加されている。

## 学習段階: Stage0 〜 Stage3

学習は PaLI 系を踏襲した多段階構成で進む。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="PaliGemmaの学習段階。Stage0で既製部品を用意、Stage1でマルチモーダル事前学習、Stage2で解像度引き上げ、Stage3で転移を行う4段階を左から右へ並べた図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="16" y="70" width="158" height="190" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="95" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage0</text>
  <text x="95" y="122" text-anchor="middle" font-size="12" fill="#18181b">単一モダリティ事前学習</text>
  <text x="95" y="156" text-anchor="middle" font-size="11" fill="#18181b">既製のSigLIP・</text>
  <text x="95" y="176" text-anchor="middle" font-size="11" fill="#18181b">Gemmaを利用</text>
  <text x="95" y="196" text-anchor="middle" font-size="11" fill="#18181b">（独自学習なし）</text>

  <line x1="174" y1="165" x2="180" y2="165" stroke="#71717a" stroke-width="2"/>
  <polygon points="180,161 188,165 180,169" fill="#71717a"/>

  <rect x="188" y="70" width="158" height="190" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="267" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage1</text>
  <text x="267" y="122" text-anchor="middle" font-size="12" fill="#18181b">マルチモーダル事前学習</text>
  <text x="267" y="156" text-anchor="middle" font-size="11" fill="#18181b">全体を学習</text>
  <text x="267" y="176" text-anchor="middle" font-size="11" fill="#18181b">凍結なし</text>
  <text x="267" y="196" text-anchor="middle" font-size="11" fill="#18181b">224px・約10億例</text>

  <line x1="346" y1="165" x2="352" y2="165" stroke="#71717a" stroke-width="2"/>
  <polygon points="352,161 360,165 352,169" fill="#71717a"/>

  <rect x="360" y="70" width="158" height="190" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="439" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage2</text>
  <text x="439" y="122" text-anchor="middle" font-size="12" fill="#18181b">解像度引き上げ</text>
  <text x="439" y="156" text-anchor="middle" font-size="11" fill="#18181b">448→896px</text>
  <text x="439" y="176" text-anchor="middle" font-size="11" fill="#18181b">高解像度タスク</text>
  <text x="439" y="196" text-anchor="middle" font-size="11" fill="#18181b">の比重を上げる</text>

  <line x1="518" y1="165" x2="524" y2="165" stroke="#71717a" stroke-width="2"/>
  <polygon points="524,161 532,165 524,169" fill="#71717a"/>

  <rect x="532" y="70" width="158" height="190" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="611" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage3</text>
  <text x="611" y="122" text-anchor="middle" font-size="12" fill="#18181b">転移</text>
  <text x="611" y="156" text-anchor="middle" font-size="11" fill="#18181b">タスク特化へ</text>
  <text x="611" y="176" text-anchor="middle" font-size="11" fill="#18181b">fine-tune</text>
  <text x="611" y="196" text-anchor="middle" font-size="11" fill="#18181b">3解像度のCKPT</text>

  <line x1="16" y1="300" x2="682" y2="300" stroke="#6366f1" stroke-width="2"/>
  <polygon points="682,296 690,300 682,304" fill="#6366f1"/>
  <text x="353" y="324" text-anchor="middle" font-size="12" fill="#71717a">連続した学習率スケジュール（段階間で減衰させず、転移が冷却として働く）</text>
</svg><figcaption><b>要点</b>: 既製部品→マルチモーダル事前学習（凍結なし）→解像度引き上げ→タスク転移、の4段階。学習率は段階間で減衰させない。</figcaption></figure>

- **Stage0: 単一モダリティ事前学習**。視覚・言語の各部品を個別に事前学習する段階だが、PaliGemma では独自学習は行わず、**既製の公開チェックポイント**（SigLIP と Gemma）をそのまま使う。
- **Stage1: マルチモーダル事前学習**。部品を結合し、広範なタスク混合で**モデル全体を学習**する。**何も凍結しない**点が特徴で、視覚エンコーダも凍結しない（CapPa / LocCa の知見にならい、captioning 系の学習が SigLIP に欠けがちな空間・関係理解を与える）。解像度 224px（$N_{\text{img}}=256$, $N_{\text{txt}}=128$）で約 **10 億例**。ただし凍結を外す代償として、初期の不整合な勾配が SigLIP を壊さないよう、視覚エンコーダの学習率は**ゆるやかに線形ウォームアップ**する。
- **Stage2: 解像度引き上げ**。Stage1 が広い知識を与えた前提で、**448×448（追加 5000 万例）→ 896×896（追加 1000 万例）**へと短く継続学習し、高解像度を要するタスク（OCR・図表・小さな物体の検出/分割）の比重を上げる。接尾辞も長くでき、例えば OCR では $N_{\text{txt}}=512$ まで伸ばして画像内の全テキストをラスタ順に読ませる。
- **Stage3: 転移**。Stage1+2 の成果は **224 / 448 / 896 px の3つのベースチェックポイント**で、これを各タスク特化のスペシャリストへ fine-tune する段階。

全体を貫くのは、段階間で学習率を減衰させない**「無限（infinite）学習率スケジュール」**で、転移がそのまま冷却（cooldown）として働く。

なお ablation からの含意として、(a) **どこも凍結しない**ほうが転移後の性能・予測可能性ともに良い、(b) 接続は **MLP より線形** が好ましい、(c) SigLIP エンコーダを外して生パッチを直接入れる Fuyu 風構成は学習効率で明確に劣る、という結果も報告されている（小型・転移重視という設計判断の裏付け）。

## まとめと、読解後に答えたい問い

要点。

- PaliGemma = **SigLIP-So400m ＋ 線形 projector ＋ Gemma-2B**（約 3B）。画像トークンを接頭辞として並べ、接尾辞を自己回帰生成する単純な「画像＋テキスト → テキスト」API。
- **prefix-LM**: 画像＋接頭辞は双方向、接尾辞は因果。マスクは $m_{ij}=\mathbb{1}[\,j \le \max(p,i)\,]$。
- **転移前提のベースモデル**。chat 完成形ではなく fine-tune で伸びることを優先。座標・マスクもテキスト化して統一 API に載せる。
- 学習は **Stage0（既製部品）→ Stage1（凍結なし・224px・約10億例）→ Stage2（448→896px）→ Stage3（転移）**。

読解後に自分で答えたい問い。

1. なぜ接尾辞だけを因果にし、画像＋接頭辞は双方向にするのか。純 AR にすると何が失われるか、$m_{ij}$ の式で説明できるか。
2. 解像度を上げると画像トークン数が $256 \to 1024 \to 4096$ と増える。計算量と性能のトレードオフを、どのタスクが解像度に敏感かと結びつけて説明できるか。
3. 「chat ではなく転移用ベース」という設計は、LLaVA 系の instruction tuning とどんな場面で優劣が分かれるか。
4. 視覚エンコーダを凍結しない判断は、SigLIP の対照学習が苦手とする何を補うためか。02章の SigLIP の性質と結びつけて言えるか。
