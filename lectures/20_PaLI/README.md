# PaLI — A Jointly-Scaled Multilingual Language-Image Model

PaLI（Pathways Language and Image model）は、画像と言語を「一つのテキスト生成問題」として統一的に解くモデルです。鍵となる主張はシンプルで、**視覚（ViT）と言語（mT5）の両方を大きくスケールさせること、とりわけ視覚側を従来より大きくすることが効く**、というものです。これまでの視覚・言語モデルは言語側ばかりが巨大で、画像エンコーダは相対的に小さいことが多くありました。PaLI はその偏りを正し、約 4B パラメータの大規模 ViT（ViT-e）と多言語エンコーダ・デコーダ言語モデルを組み合わせ、captioning・VQA・OCR・物体検出といった多様なタスクを、すべて「画像＋テキスト → テキスト」という同じインターフェースで扱います。学習には 100 言語超・約 10B 画像規模の WebLI を用い、英語タスクの最高性能を保ちながら多言語性能も獲得しました。

この教材では、(1) 全体の骨格、(2) 共同スケールという思想、(3) 全タスクをテキスト生成に落とし込む設計、(4) 多言語データ WebLI、の順に読み解いていきます。


## 全体像（まず一枚で）

PaLI の中核は **テキストのエンコーダ・デコーダ Transformer** です。画像はまず大規模 ViT に通され、その出力（パッチごとの特徴量）が**プーリングされないまま「視覚トークン列」としてエンコーダに渡されます**。エンコーダはこの視覚トークンと、プロンプトや質問などのテキストトークンを一緒に受け取り、デコーダがそれを参照しながら出力テキストを自己回帰的に生成します。物体検出も captioning も VQA も、出力はすべて「テキスト」です。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="PaLIの全体構成。画像を大規模ViTで視覚トークンに変換し、テキストプロンプトとともにエンコーダ・デコーダ言語モデルへ入力し、答えのテキストを生成する流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>

  <!-- テキスト入力 -->
  <rect x="20" y="36" width="230" height="58" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="135" y="60" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">テキスト入力</text>
  <text x="135" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">プロンプト＋質問</text>

  <!-- 画像 -->
  <rect x="20" y="160" width="80" height="80" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="60" y="205" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">画像</text>

  <line x1="100" y1="200" x2="142" y2="200" stroke="#71717a" stroke-width="2"/>
  <polygon points="150,200 142,195 142,205" fill="#71717a"/>

  <!-- ViT-e -->
  <rect x="150" y="150" width="130" height="100" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="215" y="193" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">大規模ViT</text>
  <text x="215" y="214" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">ViT-e（約4B）</text>

  <!-- 視覚トークン矢印 -->
  <line x1="280" y1="195" x2="372" y2="150" stroke="#4338ca" stroke-width="2"/>
  <polygon points="380,146 369,147 374,157" fill="#4338ca"/>
  <text x="318" y="158" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">視覚トークン</text>

  <!-- テキストトークン矢印 -->
  <line x1="250" y1="66" x2="372" y2="118" stroke="#0e7490" stroke-width="2"/>
  <polygon points="380,121 369,110 366,121" fill="#0e7490"/>
  <text x="320" y="84" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">テキストトークン</text>

  <!-- エンコーダ -->
  <rect x="384" y="96" width="120" height="120" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="444" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">エンコーダ</text>
  <text x="444" y="170" text-anchor="middle" font-size="12" font-weight="700" fill="#6366f1">視覚＋言語を統合</text>

  <line x1="504" y1="156" x2="546" y2="156" stroke="#6366f1" stroke-width="2"/>
  <polygon points="554,156 546,151 546,161" fill="#6366f1"/>

  <!-- デコーダ -->
  <rect x="556" y="96" width="120" height="120" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="616" y="150" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">デコーダ</text>
  <text x="616" y="170" text-anchor="middle" font-size="12" font-weight="700" fill="#6366f1">自己回帰生成</text>

  <line x1="616" y1="216" x2="616" y2="266" stroke="#16a34a" stroke-width="2"/>
  <polygon points="616,274 611,266 621,266" fill="#16a34a"/>

  <!-- 出力 -->
  <rect x="486" y="280" width="200" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="586" y="304" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">生成テキスト</text>
  <text x="586" y="324" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">答え／説明／検出結果</text>

  <!-- 既存重みの再利用ラベル -->
  <text x="215" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">事前学習ViTを再利用</text>
  <text x="500" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">事前学習mT5を再利用</text>
</svg>
<figcaption>画像は大規模ViTで視覚トークン列に変換され、テキストと一緒にエンコーダへ。デコーダが答えをテキスト生成する。<b>要点</b>は、視覚と言語の双方で「すでに学習済みの重み（ViTとmT5）を再利用」し、出力を常にテキストに統一していること。</figcaption>
</figure>

設計上の重要点を 3 つ挙げます。

- **既存の単一モダリティ重みを再利用する**：視覚側は大規模 vanilla ViT、言語側は多言語の mT5 を、いずれも事前学習済みチェックポイントから初期化します。ゼロから学ぶより安く、すでに獲得した能力（画像認識、言語の理解と生成）を引き継げます。
- **タスク固有の「ヘッド」を持たない**：分類用のソフトマックス層などを付けず、どのタスクでもテキストを生成します。タスクの区別はテキストのプロンプトで指示します。
- **モジュール性とスケーラビリティ**：視覚と言語が分離しているため、片方ずつ独立に大きくできます。これが次節の「共同スケール」の前提になります。

PaLI には大きく 3 つのサイズがあります。言語側 mT5-Large（約 1B）＋視覚側 ViT-G（約 1.8B）の **PaLI-3B**、言語側 mT5-XXL（約 13B）＋ ViT-G の **PaLI-15B**、そして言語側 mT5-XXL ＋新規に学習した ViT-e（約 4B）の **PaLI-17B** です。最大構成でも視覚側が全体の約 1/4 を占め、従来モデルに比べて視覚と言語の容量配分がかなり均衡しているのが特徴です。


## 視覚と言語の「共同スケール」

PaLI の中心的な問いは、「視覚・言語モデルを強くするには、どこを大きくすべきか」です。言語モデルではスケール則（大きいほど良い）がよく知られていますが、視覚・言語の結合モデルでは視覚側を大きくする効果が十分に検証されていませんでした。多くの先行研究は言語バックボーンを巨大化する一方、画像エンコーダは小さいまま据え置いていたためです。

PaLI は視覚と言語を独立に動かせるモジュール構成を活かし、両方を段階的に大きくして比較しました。結論は明快です。**言語側を大きくしても視覚側を大きくしても性能は伸び、最大の PaLI-17B でも飽和の兆候は見えない。そして視覚側のスケールは「パラメータあたりの投資対効果」が高い**、というものです。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="言語スケールと視覚スケールの二軸上に三つのPaLIを配置した図。言語の拡大と視覚の拡大の両方が性能を伸ばし、視覚の拡大は少ない追加パラメータで大きく効くことを示す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>

  <!-- 軸 -->
  <line x1="120" y1="300" x2="640" y2="300" stroke="#71717a" stroke-width="2"/>
  <polygon points="648,300 640,295 640,305" fill="#71717a"/>
  <line x1="120" y1="300" x2="120" y2="60" stroke="#71717a" stroke-width="2"/>
  <polygon points="120,52 115,60 125,60" fill="#71717a"/>
  <text x="384" y="334" text-anchor="middle" font-size="13" font-weight="700" fill="#71717a">言語スケール（mT5-Large 約1B → mT5-XXL 約13B）</text>
  <text x="60" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#71717a" transform="rotate(-90 60 180)">視覚スケール（ViT-G 約1.8B → ViT-e 約4B）</text>

  <!-- PaLI-3B -->
  <rect x="150" y="244" width="120" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="210" y="268" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">PaLI-3B</text>
  <text x="210" y="287" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">mT5-L ＋ ViT-G</text>

  <!-- PaLI-15B -->
  <rect x="470" y="244" width="120" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="530" y="268" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">PaLI-15B</text>
  <text x="530" y="287" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">mT5-XXL ＋ ViT-G</text>

  <!-- PaLI-17B -->
  <rect x="470" y="92" width="120" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="530" y="116" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">PaLI-17B</text>
  <text x="530" y="135" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">mT5-XXL ＋ ViT-e</text>

  <!-- 3B→15B 言語拡大 -->
  <line x1="270" y1="272" x2="462" y2="272" stroke="#6366f1" stroke-width="2.5"/>
  <polygon points="470,272 462,267 462,277" fill="#6366f1"/>
  <text x="366" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">言語を拡大：約＋12B</text>
  <text x="366" y="250" text-anchor="middle" font-size="11" font-weight="700" fill="#6366f1">パラメータ増は大きい</text>

  <!-- 15B→17B 視覚拡大 -->
  <line x1="530" y1="244" x2="530" y2="156" stroke="#16a34a" stroke-width="2.5"/>
  <polygon points="530,148 525,156 535,156" fill="#16a34a"/>
  <text x="636" y="196" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">視覚を拡大</text>
  <text x="636" y="214" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">約＋2Bで大きく改善</text>
</svg>
<figcaption>同じ最大言語サイズ（mT5-XXL）でも、視覚を ViT-G から ViT-e へ拡大すると、わずかな追加パラメータで平均性能が大きく伸びる。<b>要点</b>は、視覚側のスケールが「投資対効果」の面で言語側の拡大に勝ちうること。</figcaption>
</figure>

具体的には、視覚側を ViT-G（約 1.8B）から ViT-e（約 4B）へ替えると、モデル全体のパラメータは十数パーセントしか増えないにもかかわらず、複数ベンチマークの平均改善幅は、より多くのパラメータを必要とする言語側の拡大（mT5-Large から mT5-XXL、約 12B 増）に匹敵、あるいは上回りました。視覚側はパラメータあたりの効きが良い、という観察です。これを定量化するため、著者らは当時としては最大級の vanilla ViT である **ViT-e（約 4B）** を新たに学習しました。

ここで興味深いのは、**ViT-e は画像分類だけを見ると ViT-G からほぼ伸びない（飽和気味）のに、視覚・言語タスクでは明確に伸びる**点です。つまり「分類精度では頭打ちに見える視覚バックボーンにも、マルチモーダルなタスクで引き出せる伸びしろが残っている」ことを示唆しています。この知見が、視覚側をもっと大きくしてよいという PaLI のメッセージの核です。


## すべてをテキスト生成として解く

PaLI のもう一つの柱は、**多様なタスクを一つの「画像＋テキスト → テキスト」インターフェースに統一する**ことです。画像分類や多くの VQA は「決められた選択肢から選ぶ」問題として解かれがちですが、PaLI はそれらも含めて、答えを語彙から逐次生成する **オープン語彙のテキスト生成** として扱います。タスクごとに専用の出力層を作らず、入力プロンプトで「いま何をするか」を伝えます。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="captioning・VQA・OCR・物体検出という異なるタスクが、いずれも画像とテキストプロンプトを入力し、テキストを出力する共通インターフェースに統一される様子を示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>

  <!-- 中央 PaLI -->
  <rect x="300" y="120" width="120" height="120" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="172" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">PaLI</text>
  <text x="360" y="192" text-anchor="middle" font-size="11" font-weight="700" fill="#6366f1">共通の</text>
  <text x="360" y="208" text-anchor="middle" font-size="11" font-weight="700" fill="#6366f1">テキスト生成</text>

  <!-- 入力プロンプト 4種 -->
  <rect x="20" y="40" width="220" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="130" y="65" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像＋「説明文を生成」</text>

  <rect x="20" y="120" width="220" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="130" y="145" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像＋「何の花？」</text>

  <rect x="20" y="200" width="220" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="130" y="225" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像＋「文字を読み取れ」</text>

  <rect x="20" y="280" width="220" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="130" y="305" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像＋「物体を検出せよ」</text>

  <!-- 入力→PaLI 矢印 -->
  <line x1="240" y1="60" x2="296" y2="140" stroke="#0e7490" stroke-width="2"/>
  <polygon points="300,144 290,138 294,148" fill="#0e7490"/>
  <line x1="240" y1="140" x2="296" y2="160" stroke="#0e7490" stroke-width="2"/>
  <polygon points="300,162 291,156 291,166" fill="#0e7490"/>
  <line x1="240" y1="220" x2="296" y2="200" stroke="#0e7490" stroke-width="2"/>
  <polygon points="300,198 291,194 291,204" fill="#0e7490"/>
  <line x1="240" y1="300" x2="296" y2="220" stroke="#0e7490" stroke-width="2"/>
  <polygon points="300,216 290,222 294,226" fill="#0e7490"/>

  <!-- PaLI→出力 矢印 -->
  <line x1="420" y1="140" x2="476" y2="60" stroke="#16a34a" stroke-width="2"/>
  <polygon points="480,56 470,62 474,66" fill="#16a34a"/>
  <line x1="420" y1="160" x2="476" y2="140" stroke="#16a34a" stroke-width="2"/>
  <polygon points="480,138 471,134 471,144" fill="#16a34a"/>
  <line x1="420" y1="200" x2="476" y2="220" stroke="#16a34a" stroke-width="2"/>
  <polygon points="480,222 471,216 471,226" fill="#16a34a"/>
  <line x1="420" y1="220" x2="476" y2="300" stroke="#16a34a" stroke-width="2"/>
  <polygon points="480,304 470,298 474,294" fill="#16a34a"/>

  <!-- 出力 4種 -->
  <rect x="480" y="40" width="220" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="65" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">「ひまわりの写真」</text>

  <rect x="480" y="120" width="220" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="145" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">「ひまわり」</text>

  <rect x="480" y="200" width="220" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="225" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">「OPEN 24H」</text>

  <rect x="480" y="280" width="220" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="305" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">「cat 12 30 88 60 ; ...」</text>
</svg>
<figcaption>captioning・VQA・OCR・物体検出はどれも「画像＋プロンプト」を入れて「テキスト」を出すだけ。<b>要点</b>は、専用ヘッドを持たず、座標すらテキストとして出すことで、一つのモデルで多タスクを横断学習できること。</figcaption>
</figure>

この統一は事前学習でも転移学習（fine-tuning）でも同じ枠組みで行われ、知識をタスク間で共有しやすくします。代表的な落とし込み方は次の通りです。

- **captioning（説明文生成）**：画像を入力に、説明文を生成。WebLI の alt-text を 2 つに分割して入出力にする「split-captioning」や、翻訳付き Conceptual Captions を使った多言語 captioning が事前学習に含まれます。
- **VQA（視覚的質問応答）**：質問をプロンプトとして与え、答えを生成。選択肢から選ぶ分類ではなく**オープン語彙生成**で解くため、定義済み集合に縛られず、多言語の答えも扱えます。
- **OCR・シーンテキスト理解**：画像中の文字を読み取り、テキストとして出力。WebLI には自動 OCR で得た大量の画像-テキスト対が含まれ、TextVQA や ST-VQA のように画像内の文字を読む必要があるタスクに効きます。
- **物体検出**：検出を「生成的」に扱い、物体クラス名と座標をテキスト列として出力します。バウンディングボックスすらトークン列として書き出してしまう、というのが text-to-text 統一の象徴的な例です。

PaLI の事前学習はこうした **8 種類のタスクの混合（mixture）** で行われます。著者らはこの混合設計が単一目的より性能を底上げすることを ablation で確認しており、たとえば split-captioning が全体に効き、OCR タスクが文字読み取り系を助け、物体関連タスクが複数ベンチを押し上げる、といった役割分担が観察されています。

学習手順にも工夫があります。本体の事前学習は 224×224 解像度で行い、**この段階では視覚側（ViT）を凍結し、言語側のパラメータだけを更新**します。最大の PaLI-17B では、その後に高解像度（588×588）の追加学習フェーズを短く挟み、ここでは全パラメータを更新します。低解像度で広く学び、最後に高解像度で仕上げる、という二段構えです。

成果として、PaLI は captioning や VQA で当時の強力なモデルを上回りました。たとえば COCO captioning（Karpathy split）で CIDEr 149.1、VQAv2 でオープン語彙生成のまま 84.3 という結果が報告されています。注目すべきは、**分類設定（固定語彙）を使うモデルすら、生成設定の PaLI が上回った**点で、テキスト生成という柔軟なインターフェースが性能面でも不利にならないことを示しています。


## 多言語データと WebLI

PaLI の「多言語性」を支えるのが、新たに構築された大規模データセット **WebLI（Web Language-Image）** です。従来の主要な画像-テキストデータは英語中心で、多言語の視覚・言語能力を引き出すデータが不足していました。WebLI は公開ウェブ上の画像とテキストから収集され、**約 100 言語超（109 言語）にわたって約 10B（100 億）枚規模の画像と約 12B の alt-text** を含みます。さらに、画像に対して自動 OCR を適用し、約 29B もの画像-OCR テキスト対も付与されています。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="ウェブから収集した約10Bの多言語画像テキストとOCRを、品質スコア上位約10パーセントに絞り込み、八種類のタスク混合でPaLIを事前学習する流れを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>

  <!-- ウェブ -->
  <rect x="20" y="130" width="150" height="100" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="95" y="172" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">公開ウェブ</text>
  <text x="95" y="194" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">画像＋テキスト</text>

  <line x1="170" y1="180" x2="212" y2="180" stroke="#71717a" stroke-width="2"/>
  <polygon points="220,180 212,175 212,185" fill="#71717a"/>

  <!-- WebLI 生データ -->
  <rect x="220" y="96" width="200" height="168" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="320" y="126" text-anchor="middle" font-size="15" font-weight="700" fill="#0e7490">WebLI（生データ）</text>
  <text x="320" y="158" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">約10B 画像</text>
  <text x="320" y="180" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">約12B alt-text</text>
  <text x="320" y="202" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">100超の言語</text>
  <text x="320" y="224" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">約29B 画像-OCR対</text>

  <line x1="420" y1="180" x2="462" y2="180" stroke="#0e7490" stroke-width="2"/>
  <polygon points="470,180 462,175 462,185" fill="#0e7490"/>
  <text x="446" y="166" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">品質で選別</text>

  <!-- 上位10% -->
  <rect x="470" y="116" width="110" height="128" rx="10" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="525" y="160" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">上位 約10%</text>
  <text x="525" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">高品質サブ</text>
  <text x="525" y="200" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">セット</text>
  <text x="525" y="222" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">約1B 例</text>

  <line x1="580" y1="180" x2="612" y2="180" stroke="#16a34a" stroke-width="2"/>
  <polygon points="620,180 612,175 612,185" fill="#16a34a"/>

  <!-- 学習 -->
  <rect x="612" y="120" width="92" height="120" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="658" y="166" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">8タスク</text>
  <text x="658" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">混合で</text>
  <text x="658" y="206" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">事前学習</text>

  <text x="360" y="312" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">大量に集め、品質で絞り、混合タスクで一気に学ぶ</text>
</svg>
<figcaption>約10B規模・100超言語の生データを、スコア上位約10%（約1B例）に絞ってから事前学習に使う。<b>要点</b>は、規模だけでなく品質フィルタと多言語の網羅性が、多言語マルチタスク能力の源泉になっていること。</figcaption>
</figure>

ただし、巨大であればそのまま使うわけではありません。著者らは品質スコアの**上位約 10%（約 1B 例）** に絞り込んで PaLI の学習に用います。混合全体の学習例は約 1.6B で、100 超の言語にわたる裾の長い分布になっています。データカード（datasheet）を公開し、収集・構成を文書化している点も、大規模データの再利用性と透明性の観点で重要です。

多言語化の効果は、英語性能を犠牲にしていない点に表れます。mT5 系の言語理解・生成能力を引き継ぎつつ、純粋なテキストタスク（span corruption など）も混合に含めることで、**言語モデルとしての破滅的忘却を防ぎ**、英語のベンチマークでも、複数言語にわたる転移評価でも、強い言語理解を保ちました。視覚・言語タスクでも、Crossmodal-3600（多言語 captioning）や xGQA・MaXM（多言語 VQA）といった多言語ベンチで先行手法を大きく上回っています。さらに、適切にスケールさせれば「100 を超える言語を一つのモデルで扱いながら、英語専用タスクでも最高性能を出せる」ことを示しました。これが PaLI の "Multilingual" の意味するところです。


## まとめと、読解後に答えたい問い

PaLI のメッセージは 3 つに集約できます。

1. **共同スケール**：視覚（ViT）と言語（mT5）を独立に大きくでき、両方を伸ばすと性能が伸びる。特に視覚側はパラメータあたりの効きが良く、従来モデルが軽視してきた画像エンコーダにまだ伸びしろがある。
2. **テキスト生成への統一**：captioning・VQA・OCR・物体検出までを「画像＋テキスト → テキスト」に落とし込み、タスク固有ヘッドを排して横断学習する。座標すらトークン列として出力する。
3. **多言語・大規模データ**：100 言語超・約 10B 規模の WebLI を品質フィルタしつつ用い、英語性能を保ったまま多言語の視覚・言語能力を獲得する。

最後に、理解を確かめるための問いを置いておきます。

- なぜ視覚側のスケールは「パラメータあたり」では言語側より効くと考えられるのか。画像分類では飽和するのに視覚・言語タスクで伸びるのはなぜか、仮説を立ててみよう。
- 視覚トークンを「プーリングせず」エンコーダに渡す設計は、CLIP のように画像を 1 ベクトルへ要約する方式と比べて何を得て何を失うか。
- 物体検出をテキスト生成として解くと、座標の精度や評価はどう変わるか。分類ヘッドを持つ検出器と比べた利点・弱点は何か。
- 本体学習で ViT を凍結し、最後だけ高解像度で全体を更新するのはなぜ合理的か。計算コストと精度のトレードオフの観点で説明してみよう。
- WebLI を上位約 10% に絞るのはなぜか。「量」と「質」のどちらが多言語マルチタスク性能に効くのか、ablation の発想で考えてみよう。
- 多言語化しても英語性能が落ちない（破滅的忘却が起きない）のは、混合タスクのどの要素のおかげか。テキスト専用タスクを混ぜる意味を言語化してみよう。
