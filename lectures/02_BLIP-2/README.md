# BLIP-2 — Q-Former で凍結エンコーダと凍結LLMを橋渡し

BLIP-2 は、巨大化を続ける画像エンコーダと大規模言語モデル（LLM）を **どちらも凍結したまま** 流用し、その間に **Q-Former** という軽量モジュールだけを学習させて、視覚と言語のモダリティギャップを埋める手法です。Flamingo が「凍結 LLM の内部に cross-attention 層を差し込んで視覚情報を注入する」のに対し、BLIP-2 は **凍結エンコーダ側に小さなクエリ機構を置き、画像を「言語に効く固定少数トークン」へ要約してから LLM の入力に前置する** という、コネクタの設計思想が異なります。腹落ちさせたいのは「Q-Former が何を抽出しているのか」と「なぜ 2 段階で学習するのか」の 2 点です。

## 全体像（まず一枚で）

構成は **凍結画像エンコーダ（ViT）→ Q-Former →（線形射影）→ 凍結 LLM** の一直線です。学習対象は Q-Former と小さな射影層だけで、論文では学習可能パラメータ約 188M。それでもゼロショット VQAv2 で Flamingo80B を 8.7 ポイント上回り、しかも学習パラメータは 54 分の 1、という極端な効率を示しました。凍結バックボーンを使うこと自体は Flamingo と同じ発想ですが、BLIP-2 は「凍結する両端の良さを最大限活かすために、間のコネクタをどう設計し、どう学習させるか」に振り切っています。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="入力画像が凍結画像エンコーダを通り、Q-Formerで固定数の視覚特徴へ圧縮され、線形射影を経て凍結LLMへソフトな視覚プロンプトとして前置される全体像の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="20" y="20" width="16" height="16" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="44" y="33" text-anchor="start" font-size="12" font-weight="700" fill="#155e75">凍結（更新しない）</text>
  <rect x="210" y="20" width="16" height="16" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="234" y="33" text-anchor="start" font-size="12" font-weight="700" fill="#3730a3">学習対象</text>
  <rect x="18" y="120" width="84" height="64" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="60" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力画像</text>
  <line x1="102" y1="152" x2="118" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="126,152 116,147 116,157" fill="#71717a"/>
  <rect x="128" y="110" width="116" height="84" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="186" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">画像エンコーダ</text>
  <text x="186" y="165" text-anchor="middle" font-size="11" fill="#155e75">ViT (EVA-CLIP 等)</text>
  <rect x="156" y="172" width="60" height="16" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="1.5"/>
  <text x="186" y="184" text-anchor="middle" font-size="10" font-weight="700" fill="#0e7490">凍結</text>
  <line x1="244" y1="152" x2="260" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="268,152 258,147 258,157" fill="#71717a"/>
  <rect x="270" y="98" width="150" height="108" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="345" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3">Q-Former</text>
  <text x="345" y="146" text-anchor="middle" font-size="11" fill="#3730a3">32 学習可能クエリ</text>
  <text x="345" y="163" text-anchor="middle" font-size="10" fill="#3730a3">cross-attn で</text>
  <text x="345" y="176" text-anchor="middle" font-size="10" fill="#3730a3">視覚特徴を抽出</text>
  <rect x="312" y="184" width="66" height="16" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="345" y="196" text-anchor="middle" font-size="10" font-weight="700" fill="#4338ca">学習対象</text>
  <line x1="420" y1="152" x2="438" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="446,152 436,147 436,157" fill="#71717a"/>
  <text x="438" y="138" text-anchor="middle" font-size="10" fill="#71717a">Z(32×768)</text>
  <rect x="456" y="120" width="64" height="64" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="488" y="148" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">線形</text>
  <text x="488" y="164" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">射影</text>
  <line x1="520" y1="152" x2="536" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="544,152 534,147 534,157" fill="#71717a"/>
  <rect x="552" y="110" width="150" height="84" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="627" y="144" text-anchor="middle" font-size="14" font-weight="700" fill="#155e75">凍結 LLM</text>
  <text x="627" y="163" text-anchor="middle" font-size="11" fill="#155e75">OPT / FlanT5</text>
  <rect x="597" y="170" width="60" height="16" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="1.5"/>
  <text x="627" y="182" text-anchor="middle" font-size="10" font-weight="700" fill="#0e7490">凍結</text>
  <line x1="627" y1="194" x2="627" y2="222" stroke="#71717a" stroke-width="2"/>
  <polygon points="627,230 622,220 632,220" fill="#71717a"/>
  <rect x="520" y="232" width="190" height="48" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="615" y="261" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">出力テキスト（説明・回答）</text>
</svg><figcaption><b>両端は凍結</b>、間の <b>Q-Former と線形射影だけを学習</b>します。Q-Former の出力 Z（32×768）は射影後、<b>ソフトな視覚プロンプト</b>として凍結 LLM の入力に前置されます。</figcaption></figure>

## Q-Former（Querying Transformer）

Q-Former は BERT_base で初期化された軽量 Transformer（約 188M）で、内部に 2 つのサブモジュール—**画像 Transformer**（凍結画像特徴と相互作用）と **テキスト Transformer**（エンコーダにもデコーダにもなる）—を持ち、両者は **同じ self-attention 層を共有** します。中核は **学習可能なクエリ埋め込み（論文では 32 個、各次元 768）** です。これらのクエリは self-attention で互いに作用し、**cross-attention で凍結画像特徴を参照** して、画像から固定数の出力特徴を抜き出します。cross-attention 層は 1 ブロックおきに挿入され、ここだけはランダム初期化です。

ポイントは **出力 Z のサイズが入力解像度に依存しない** ことです。ViT-L/14 の凍結特徴は例えば 257×1024 と長大ですが、Q-Former はこれを常に 32×768 へ畳み込みます。つまり Q-Former は **情報ボトルネック** として働き、「LLM にとって有益な視覚情報だけ」を少数トークンへ絞り込む役割を担います。学習されるのはこのクエリと Q-Former 本体（と Stage2 の射影）だけで、画像エンコーダ自体は一切更新されません。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="32個の学習可能クエリが可変長の凍結画像特徴をcross-attentionで参照し、固定長32個の出力特徴へ圧縮する情報ボトルネックの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="340" y="26" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">32 学習可能クエリ（入力）</text>
  <rect x="262" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="280" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="298" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="316" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="334" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="352" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <rect x="370" y="36" width="14" height="14" rx="3" fill="#c7d2fe" stroke="#4338ca" stroke-width="1.5"/>
  <text x="402" y="48" text-anchor="start" font-size="11" font-weight="700" fill="#3730a3">…×32</text>
  <line x1="340" y1="52" x2="340" y2="86" stroke="#71717a" stroke-width="2"/>
  <polygon points="340,92 335,82 345,82" fill="#71717a"/>
  <rect x="30" y="120" width="120" height="120" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="90" y="142" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">凍結画像特徴</text>
  <text x="90" y="158" text-anchor="middle" font-size="10" fill="#155e75">可変長・多数</text>
  <text x="90" y="171" text-anchor="middle" font-size="10" fill="#155e75">例 257×1024</text>
  <rect x="50" y="184" width="80" height="10" rx="2" fill="#ecfeff" stroke="#0e7490" stroke-width="1"/>
  <rect x="50" y="198" width="80" height="10" rx="2" fill="#ecfeff" stroke="#0e7490" stroke-width="1"/>
  <rect x="50" y="212" width="80" height="10" rx="2" fill="#ecfeff" stroke="#0e7490" stroke-width="1"/>
  <rect x="50" y="226" width="80" height="10" rx="2" fill="#ecfeff" stroke="#0e7490" stroke-width="1"/>
  <line x1="150" y1="150" x2="256" y2="146" stroke="#71717a" stroke-width="2"/>
  <polygon points="264,145 254,141 255,151" fill="#71717a"/>
  <text x="200" y="138" text-anchor="middle" font-size="9" fill="#71717a">視覚特徴を参照</text>
  <rect x="240" y="94" width="230" height="148" rx="10" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="355" y="112" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">Q-Former</text>
  <rect x="260" y="120" width="190" height="46" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="355" y="139" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Cross-Attention</text>
  <text x="355" y="156" text-anchor="middle" font-size="9" fill="#3730a3">凍結画像特徴を参照（1ブロックおき）</text>
  <rect x="260" y="178" width="190" height="46" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="355" y="197" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Self-Attention</text>
  <text x="355" y="214" text-anchor="middle" font-size="9" fill="#3730a3">クエリ同士が相互作用</text>
  <line x1="355" y1="242" x2="355" y2="268" stroke="#71717a" stroke-width="2"/>
  <polygon points="355,276 350,266 360,266" fill="#71717a"/>
  <text x="340" y="304" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">32 出力特徴 Z（固定長 32×768）</text>
  <rect x="262" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="280" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="298" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="316" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="334" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="352" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <rect x="370" y="282" width="14" height="14" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <text x="402" y="294" text-anchor="start" font-size="11" font-weight="700" fill="#16a34a">…×32</text>
  <rect x="500" y="110" width="200" height="124" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="600" y="134" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">情報ボトルネック</text>
  <text x="600" y="156" text-anchor="middle" font-size="10" fill="#3f3f46">可変長の画像特徴を</text>
  <text x="600" y="172" text-anchor="middle" font-size="10" fill="#3f3f46">解像度に依存しない</text>
  <text x="600" y="188" text-anchor="middle" font-size="10" fill="#3f3f46">固定長 32 トークンへ圧縮</text>
  <text x="600" y="210" text-anchor="middle" font-size="10" fill="#3f3f46">→ LLM に有益な情報だけ</text>
  <text x="600" y="224" text-anchor="middle" font-size="10" fill="#3f3f46">を残す</text>
</svg><figcaption><b>32 個の学習可能クエリ</b>が cross-attention で凍結画像特徴を参照し、入力解像度に依らず <b>常に 32×768 の固定長 Z</b> を出力します。Q-Former は画像情報を絞る <b>ボトルネック</b>として働きます。</figcaption></figure>

## 2段階学習

BLIP-2 の核心は、Q-Former を **2 段階** で事前学習することです。順番に「画像と言語をつなぐ表現」を作ってから「LLM に通じる言葉」へ翻訳する、という分業になっています。

- **Stage 1（表現学習）**：Q-Former を **凍結画像エンコーダ** に接続し、画像-テキストのペアで学習します。後述する **ITC / ITG / ITM の 3 目的を同時に** 最適化し、クエリに「テキストにとって最も有益な視覚特徴」を学ばせます。ここで LLM はまだ登場しません。
- **Stage 2（生成学習）**：Stage 1 で学んだ Q-Former（と凍結画像エンコーダ）を **凍結 LLM** に接続します。Q-Former の出力 Z を **線形射影（FC）** で LLM の埋め込み次元に合わせ、テキスト埋め込みの前に **ソフトな視覚プロンプト** として前置します。デコーダ型 LLM（OPT）なら言語モデリング損失、エンコーダ・デコーダ型（FlanT5）ならプレフィックス言語モデリング損失で、**Q-Former と射影だけ** を更新します。

なぜ 2 段階に分けるのか。Stage 1 を飛ばして Stage 2 だけで学習すると、Q-Former は「モダリティギャップを埋める」と「言葉へ翻訳する」を同時にこなさねばならず、論文の分析（Figure 5）では性能が大きく落ち、とりわけ OPT では学習が進むほど性能が崩れる **破滅的忘却** が起きました。Stage 1 で先に Z を「言語的に意味のある表現」へ仕立てておくことで、Stage 2 では凍結 LLM の負担が減り、忘却も緩和されます。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="Stage1は凍結画像エンコーダとQ-Formerで3目的を学ぶ表現学習、Stage2はQ-Former出力を射影し凍結LLMへ前置する生成学習で、Q-Formerの重みを継承する2段階学習の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="15" y="44" width="74" height="44" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="52" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">Stage 1</text>
  <text x="52" y="80" text-anchor="middle" font-size="10" fill="#3730a3">表現学習</text>
  <rect x="110" y="40" width="100" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="160" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">画像Enc</text>
  <text x="160" y="82" text-anchor="middle" font-size="10" font-weight="700" fill="#0e7490">凍結</text>
  <line x1="210" y1="68" x2="234" y2="68" stroke="#71717a" stroke-width="2"/>
  <polygon points="242,68 232,63 232,73" fill="#71717a"/>
  <rect x="244" y="36" width="120" height="66" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="304" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">Q-Former</text>
  <text x="304" y="82" text-anchor="middle" font-size="10" font-weight="700" fill="#4338ca">学習</text>
  <line x1="364" y1="68" x2="392" y2="68" stroke="#71717a" stroke-width="2"/>
  <polygon points="400,68 390,63 390,73" fill="#71717a"/>
  <rect x="402" y="34" width="128" height="106" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/>
  <text x="466" y="54" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">3つの目的</text>
  <rect x="414" y="62" width="104" height="20" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
  <text x="466" y="76" text-anchor="middle" font-size="10" font-weight="700" fill="#166534">ITC（対照）</text>
  <rect x="414" y="86" width="104" height="20" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="466" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#3730a3">ITG（生成）</text>
  <rect x="414" y="110" width="104" height="20" rx="5" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="466" y="124" text-anchor="middle" font-size="10" font-weight="700" fill="#991b1b">ITM（マッチ）</text>
  <text x="624" y="62" text-anchor="middle" font-size="10" fill="#3f3f46">クエリ⇄テキストの</text>
  <text x="624" y="78" text-anchor="middle" font-size="10" fill="#3f3f46">attention mask を</text>
  <text x="624" y="94" text-anchor="middle" font-size="10" fill="#3f3f46">目的ごとに切替えて</text>
  <text x="624" y="110" text-anchor="middle" font-size="10" fill="#3f3f46">同時に学習</text>
  <line x1="304" y1="140" x2="304" y2="246" stroke="#4338ca" stroke-width="2" stroke-dasharray="6 4"/>
  <polygon points="304,254 299,244 309,244" fill="#4338ca"/>
  <text x="320" y="200" text-anchor="start" font-size="11" font-weight="700" fill="#4338ca">Q-Former 重みを継承</text>
  <rect x="15" y="252" width="74" height="44" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="52" y="272" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">Stage 2</text>
  <text x="52" y="288" text-anchor="middle" font-size="10" fill="#155e75">生成学習</text>
  <rect x="108" y="256" width="92" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="154" y="280" text-anchor="middle" font-size="11" font-weight="700" fill="#155e75">画像Enc</text>
  <text x="154" y="298" text-anchor="middle" font-size="10" font-weight="700" fill="#0e7490">凍結</text>
  <line x1="200" y1="284" x2="216" y2="284" stroke="#71717a" stroke-width="2"/>
  <polygon points="224,284 214,279 214,289" fill="#71717a"/>
  <rect x="226" y="254" width="86" height="60" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="269" y="280" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">Q-Former</text>
  <text x="269" y="298" text-anchor="middle" font-size="10" font-weight="700" fill="#4338ca">学習</text>
  <line x1="312" y1="284" x2="326" y2="284" stroke="#71717a" stroke-width="2"/>
  <polygon points="334,284 324,279 324,289" fill="#71717a"/>
  <rect x="336" y="258" width="52" height="52" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="362" y="282" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">FC</text>
  <text x="362" y="298" text-anchor="middle" font-size="9" font-weight="700" fill="#4338ca">学習</text>
  <line x1="388" y1="284" x2="402" y2="284" stroke="#71717a" stroke-width="2"/>
  <polygon points="410,284 400,279 400,289" fill="#71717a"/>
  <rect x="412" y="252" width="128" height="64" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="476" y="278" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">凍結 LLM</text>
  <text x="476" y="296" text-anchor="middle" font-size="9" fill="#155e75">OPT / FlanT5</text>
  <line x1="540" y1="284" x2="556" y2="284" stroke="#71717a" stroke-width="2"/>
  <polygon points="564,284 554,279 554,289" fill="#71717a"/>
  <rect x="566" y="258" width="130" height="52" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="631" y="288" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">出力テキスト</text>
</svg><figcaption><b>Stage 1</b> で凍結画像エンコーダと Q-Former を 3 目的で学習し、<b>その重みを Stage 2 が継承</b>。Stage 2 では射影した Z を <b>凍結 LLM に前置</b>して生成を学びます。先に表現を作るので LLM の負担が減り、破滅的忘却を抑えられます。</figcaption></figure>

Stage 2 の前置を擬似コードで示すと、勾配が流れるのは Q-Former と FC だけ、という点が一目で分かります。

```python
# Stage 2 の概念（LLM は凍結）
img_feat   = frozen_vit(image)              # 可変長の凍結画像特徴
Z          = q_former(img_feat)             # (32, 768) クエリ非依存の視覚要約
soft_prompt = linear_proj(Z)                # (32, d_llm) LLM 埋め込み次元へ射影
inputs     = concat([soft_prompt, text_embeds])   # 視覚プロンプトを前置
loss       = frozen_llm(inputs)             # 凍結。勾配は q_former と linear_proj のみ
```

## 3つの目的関数（Stage1）

Stage 1 では 3 つの目的を **同じ入力・同じパラメータで共有** しつつ、**目的ごとに self-attention マスクを切り替える** ことで、クエリとテキストの相互作用を制御します。これが BLIP-2 の地味だが効く工夫です。

- **ITC（Image-Text Contrastive / 画像-文 対照）**：画像表現 Z とテキスト表現（テキスト Transformer の [CLS] 出力）の相互情報量を最大化します。32 個の各クエリ出力とテキストの類似度を計算し、その **最大値** を画像-文類似度として、正例ペアを負例に対して引き離します。マスクは **uni-modal**（クエリとテキストが互いに見えない）で、情報リークを防ぎます。凍結エンコーダのおかげで GPU あたりの試料数を稼げるため、モメンタムキューではなく **バッチ内負例** を使います。
- **ITG（Image-grounded Text Generation / 画像を条件とした文生成）**：画像を条件にテキストを生成します。Q-Former では凍結画像エンコーダとテキストトークンが直接やり取りできないため、生成に必要な情報は **いったんクエリへ集約され、self-attention 経由でテキストへ渡される** 必要があります。これにより、クエリはテキストに関する情報をすべて捉えるよう強制されます。マスクは **multimodal causal**（クエリ同士は相互参照可、テキストはクエリ全体と過去のテキストのみ参照可）で、先頭トークンは [CLS] を [DEC] に置き換えます。
- **ITM（Image-Text Matching / 画像-文 マッチング）**：画像と文が一致するかを 2 値分類します。マスクは **bi-directional**（全クエリと全テキストが相互参照可）で、各クエリ出力を 2 クラス分類器に通し、ロジットを全クエリで平均してマッチスコアにします。難しい負例を集める **hard negative mining** も併用します。

| 目的 | 学ぶこと | attention mask | クエリとテキストの関係 |
|---|---|---|---|
| **ITC** | 画像表現と文表現の整合（対照学習） | uni-modal | 相互不可視（情報リーク防止） |
| **ITG** | 画像を条件としたテキスト生成 | multimodal causal | テキストはクエリ＋過去テキストを参照、クエリはテキスト不可視 |
| **ITM** | 画像-文の一致/不一致の 2 値判定 | bi-directional | 全要素が相互参照 |

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="ITC・ITG・ITMの3目的でクエリとテキストのattention maskが異なることを示す2かける2のマスク図。ITCは相互不可視、ITGは因果的、ITMは全可視" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="120" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">ITC：対照</text>
  <text x="120" y="56" text-anchor="middle" font-size="10" fill="#3f3f46">uni-modal（互いに不可視）</text>
  <text x="95" y="84" text-anchor="middle" font-size="10" fill="#71717a">参照先 →</text>
  <text x="95" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="145" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <text x="48" y="148" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="48" y="198" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <rect x="70" y="120" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="95" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="120" y="120" width="50" height="50" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <text x="145" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">不可</text>
  <rect x="70" y="170" width="50" height="50" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <text x="95" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">不可</text>
  <rect x="120" y="170" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="145" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <text x="370" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">ITG：生成</text>
  <text x="370" y="56" text-anchor="middle" font-size="10" fill="#3f3f46">multimodal causal（因果的）</text>
  <text x="345" y="84" text-anchor="middle" font-size="10" fill="#71717a">参照先 →</text>
  <text x="345" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="395" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <text x="298" y="148" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="298" y="198" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <rect x="320" y="120" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="345" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="370" y="120" width="50" height="50" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <text x="395" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">不可</text>
  <rect x="320" y="170" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="345" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="370" y="170" width="50" height="50" fill="#e0e7ff" stroke="#71717a" stroke-width="1"/>
  <text x="395" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">因果</text>
  <text x="610" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="#991b1b">ITM：マッチ</text>
  <text x="610" y="56" text-anchor="middle" font-size="10" fill="#3f3f46">bi-directional（全可視）</text>
  <text x="585" y="84" text-anchor="middle" font-size="10" fill="#71717a">参照先 →</text>
  <text x="585" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="635" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <text x="538" y="148" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">Q</text>
  <text x="538" y="198" text-anchor="middle" font-size="11" font-weight="700" fill="#3f3f46">T</text>
  <rect x="560" y="120" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="585" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="610" y="120" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="635" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="560" y="170" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="585" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="610" y="170" width="50" height="50" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="635" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">可</text>
  <rect x="180" y="262" width="18" height="18" fill="#dcfce7" stroke="#71717a" stroke-width="1"/>
  <text x="206" y="276" text-anchor="start" font-size="11" fill="#3f3f46">可（参照できる）</text>
  <rect x="330" y="262" width="18" height="18" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <text x="356" y="276" text-anchor="start" font-size="11" fill="#3f3f46">不可（マスク）</text>
  <rect x="470" y="262" width="18" height="18" fill="#e0e7ff" stroke="#71717a" stroke-width="1"/>
  <text x="496" y="276" text-anchor="start" font-size="11" fill="#3f3f46">因果（過去のみ）</text>
</svg><figcaption>行＝参照元、列＝参照先（Q＝クエリ、T＝テキスト）。<b>ITC は相互不可視</b>、<b>ITG はテキストがクエリ＋過去テキストのみ参照（因果的）</b>、<b>ITM は全可視</b>。同じネットワークでマスクだけを差し替えて 3 目的を同時に学びます。</figcaption></figure>

## ACC 研究との関連

発展研究の **Adaptive Cluster-CLIP（ACC）** は、物体中心の Open-Vocabulary な動画フレーム検索を狙う手法です。Dense-CLIP 系の **局所特徴をクラスタリングして少数の集約ベクトルへ圧縮** し、その **クエリ非依存インデックス** を多数の検索クエリで再利用します。背景には「生成型 MLLM はフレームごとに全体を再処理するため重い」という課題意識があります。

BLIP-2 の Q-Former が「画像 → 固定少数トークン」へ畳み込む発想は、ACC の「局所特徴 → 少数集約ベクトル」と **圧縮の形がよく似ています**。どちらも **凍結 CLIP 系の特徴の上に軽量なコネクタを載せ**、長大な視覚特徴を扱いやすい少数表現へ要約する、という設計を共有します。一方で用途と性質は対照的です。

| 観点 | BLIP-2 の Q-Former | ACC の集約ベクトル |
|---|---|---|
| 目的 | 生成（LLM へ渡す視覚プロンプト） | 検索（フレームのインデックス） |
| クエリ依存性 | 文に依存しない視覚要約（出力 Z は入力文に依らない） | クエリ非依存インデックスを多数クエリで再利用 |
| 圧縮の中身 | 学習可能クエリ × cross-attention で抽出 | 局所特徴のクラスタリングで集約 |
| 凍結バックボーン | 凍結 ViT＋凍結 LLM | 凍結 CLIP（Dense-CLIP）系 |

この対比は「凍結特徴をどう少数表現へ畳むか」という共通課題に対し、**生成に最適化するか（BLIP-2）、検索インデックスに最適化するか（ACC）** で設計が分岐する好例です。ACC を読む際は、Q-Former の「文非依存・固定長の視覚要約」を比較対象として持っておくと、ACC の「クエリ非依存インデックスの再利用」という効率化の主張が立体的に見えてきます。

## まとめと、読解後に答えたい問い

- BLIP-2 は **凍結画像エンコーダ＋凍結 LLM** を、学習する **Q-Former と射影だけ** で橋渡しする計算効率の高い手法。
- **Q-Former** は少数の学習可能クエリ（例 32 個）を持ち、cross-attention で凍結画像特徴を **固定長（32×768）へ圧縮** する情報ボトルネック。
- **2 段階学習** が肝。Stage 1 で ITC/ITG/ITM の 3 目的を **マスクを切り替えて同時に** 学び、言語的に意味のある視覚表現を作る。Stage 2 でそれを射影し、凍結 LLM に前置して生成を学ぶ。Stage 1 を省くと性能が落ち、OPT では破滅的忘却が起きる。
- 動画は対象外（画像が前提）。生成は凍結 LLM の知識・推論の弱点をそのまま継承する。

読解後に自分の言葉で答えたい問い:

1. Q-Former の出力 Z が「入力画像の解像度に依存しない固定長」であることは、計算量と LLM 側の設計にどんな利点をもたらすか。
2. ITG の multimodal causal マスクは「クエリにテキスト情報を集約させる」よう働くと説明される。なぜマスク設計だけでそれが強制されるのか、情報の通り道を辿って説明できるか。
3. なぜ Stage 1 を省くと OPT で破滅的忘却が顕著になり、FlanT5 でも性能が落ちるのか。表現学習が「LLM の負担」をどう減らしているのか。
4. Q-Former は **文に依存しない** 視覚要約を作る。これは VQA のように質問に応じて見るべき領域が変わるタスクでは弱点になり得るか。ACC の「クエリ非依存インデックス」と同じトレードオフをどう捉えるか。
