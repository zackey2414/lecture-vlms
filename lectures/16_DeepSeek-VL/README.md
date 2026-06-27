# DeepSeek-VL — Towards Real-World Vision-Language Understanding

DeepSeek-VL（arXiv:2403.05525）は、**実世界（real-world）のアプリケーションで実際に使える**ことを最優先に設計されたオープンソースの視覚言語モデル（VLM）である。ベンチマークの数字だけを追うのではなく、Web スクリーンショット・PDF・OCR・チャート・専門文書といった「現場で人が投げてくる入力」を相手にできることを狙う。そのために本論文は 3 本の柱を立てる。すなわち、(1) 実世界を広くカバーする**データ設計**、(2) 高解像度を固定トークン予算で扱う**ハイブリッド視覚エンコーダ**、(3) 視覚を獲得しつつ**言語能力を壊さない学習戦略**である。土台となる言語モデルは DeepSeek-LLM で、モデルサイズは 1.3B と 7B の 2 種類が公開された。

このページでは、まず全体像を一枚で掴み、次に視覚エンコーダの作り方、データと言語保持、3 段階の学習手順という順で読み解いていく。

## 全体像（まず一枚で）

DeepSeek-VL の推論経路は「**ハイブリッド視覚エンコーダ → MLP コネクタ（VL アダプタ）→ 言語モデル**」という、LLaVA 系で馴染みのある 3 段構成である。ただし視覚エンコーダが 1 本ではなく、**意味をとらえる SigLIP-L** と **細部をとらえる SAM-B** の 2 本を併用し、両者の特徴を結合してから 1 本の MLP で LLM の入力空間へ橋渡しする点が大きな特徴である。最終的に $1024 \times 1024$ の画像は **576 個の視覚トークン**へと凝縮され、テキストトークンと一緒に DeepSeek-LLM に入力される。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="DeepSeek-VLの全体像。SigLIP-LとSAM-Bからなるハイブリッド視覚エンコーダの特徴を結合し、MLPコネクタを介してDeepSeek LLMへ入力する流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <rect x="20" y="160" width="80" height="60" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="60" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力画像</text>
  <text x="60" y="204" text-anchor="middle" font-size="12" fill="#18181b">1024角</text>
  <rect x="140" y="70" width="165" height="60" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="222" y="95" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">SigLIP-L</text>
  <text x="222" y="114" text-anchor="middle" font-size="12" fill="#18181b">意味・低〜中解像度 384角</text>
  <rect x="140" y="240" width="165" height="60" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="222" y="265" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">SAM-B</text>
  <text x="222" y="284" text-anchor="middle" font-size="12" fill="#18181b">細部・高解像度 1024角</text>
  <line x1="100" y1="190" x2="120" y2="190" stroke="#71717a" stroke-width="2"/>
  <line x1="120" y1="190" x2="120" y2="100" stroke="#71717a" stroke-width="2"/>
  <line x1="120" y1="100" x2="140" y2="100" stroke="#71717a" stroke-width="2"/>
  <polygon points="140,100 132,96 132,104" fill="#71717a"/>
  <line x1="120" y1="190" x2="120" y2="270" stroke="#71717a" stroke-width="2"/>
  <line x1="120" y1="270" x2="140" y2="270" stroke="#71717a" stroke-width="2"/>
  <polygon points="140,270 132,266 132,274" fill="#71717a"/>
  <rect x="345" y="150" width="90" height="80" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="390" y="183" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">特徴を結合</text>
  <text x="390" y="201" text-anchor="middle" font-size="12" fill="#18181b">576 トークン</text>
  <line x1="305" y1="100" x2="325" y2="100" stroke="#71717a" stroke-width="2"/>
  <line x1="325" y1="100" x2="325" y2="175" stroke="#71717a" stroke-width="2"/>
  <line x1="325" y1="175" x2="345" y2="175" stroke="#71717a" stroke-width="2"/>
  <polygon points="345,175 337,171 337,179" fill="#71717a"/>
  <line x1="305" y1="270" x2="325" y2="270" stroke="#71717a" stroke-width="2"/>
  <line x1="325" y1="270" x2="325" y2="205" stroke="#71717a" stroke-width="2"/>
  <line x1="325" y1="205" x2="345" y2="205" stroke="#71717a" stroke-width="2"/>
  <polygon points="345,205 337,201 337,209" fill="#71717a"/>
  <rect x="470" y="160" width="110" height="60" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="525" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">VL アダプタ</text>
  <text x="525" y="204" text-anchor="middle" font-size="12" fill="#18181b">MLP コネクタ</text>
  <line x1="435" y1="190" x2="470" y2="190" stroke="#71717a" stroke-width="2"/>
  <polygon points="470,190 462,186 462,194" fill="#71717a"/>
  <rect x="610" y="110" width="95" height="160" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="657" y="186" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">DeepSeek</text>
  <text x="657" y="206" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">LLM</text>
  <line x1="580" y1="190" x2="610" y2="190" stroke="#71717a" stroke-width="2"/>
  <polygon points="610,190 602,186 602,194" fill="#71717a"/>
  <rect x="595" y="40" width="120" height="45" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="655" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">テキスト/言語</text>
  <line x1="655" y1="85" x2="655" y2="110" stroke="#71717a" stroke-width="2"/>
  <polygon points="655,110 651,102 659,102" fill="#71717a"/>
  <rect x="595" y="300" width="120" height="45" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="655" y="328" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">自由形式の回答</text>
  <line x1="657" y1="270" x2="657" y2="300" stroke="#71717a" stroke-width="2"/>
  <polygon points="657,300 653,292 661,292" fill="#71717a"/>
</svg><figcaption>全体像。<b>要点</b>は「2 本の視覚エンコーダの特徴を結合 → 1 本の MLP → DeepSeek LLM」という流れ。高解像度の画像も結合後は 576 トークンに収まり、テキストと並べて言語モデルへ渡される。</figcaption></figure>

ここで重要なのは、**視覚モジュールは固定トークン予算（576 トークン）の中で高解像度入力を活かす**よう設計されている点である。トークン数を抑えることで推論コストを管理しつつ、テキストと画像を交互に並べるインターリーブ入力や多ターン対話にも耐えられるようにしている。

## ハイブリッド視覚エンコーダ（SigLIP-L＋SAM-B）

なぜ視覚エンコーダを 2 本にするのか。出発点は「SigLIP のような CLIP 系エンコーダ単体では実世界の問いに十分答えきれない」という観察である。CLIP 系は意味（semantic）表現に最適化されているが、見た目が明確に異なる画像でも似た特徴に潰れてしまう **"CLIP-blind pairs"** の問題を抱える。さらに $224 \times 224$ や $384 \times 384$ といった比較的低い解像度に縛られるため、密な OCR や視覚的グラウンディングのように**低レベルで細かい特徴**を要求するタスクが苦手である。

そこで DeepSeek-VL は、意味をとらえる SigLIP-L に加えて、**視覚専用（vision-only）の自己教師ありエンコーダである SAM-B**（ViTDet ベース）を併用する。役割分担は明確で、

- **SigLIP-L**: $384 \times 384$ の低〜中解像度入力から、画像全体の意味・高レベルな概念をとらえる。
- **SAM-B**: 画像を $1024 \times 1024$ にリサイズして処理し、テキストの細部や小さな物体といった**高解像度の細かい情報**をとらえる。

両者の特徴を結合することで、高解像度を扱いながら意味と細部の両方を保持する。具体的な次元の流れは次の通りである。

- SAM-B は高解像度入力から $64 \times 64 \times 256$ の特徴マップを出力する。
- VL アダプタはこれを $96 \times 96 \times 256$ に補間し、ストライド 2 の畳み込みを 2 回かけて $24 \times 24 \times 1024$ にしたうえで $576 \times 1024$ に整形する。
- 一方 SigLIP-L 側も $576 \times 1024$ の低解像度（意味）特徴を出力する。
- 両者を次元方向に結合すると、**576 個・各 $2048$ 次元の視覚トークン**ができあがる。
- これらは GeLU 活性化を経て埋め込み層を通り、言語モデルと接続される。

トークン数が経路によらず 576 に揃うのは、SAM-B 経路で畳み込みによる空間方向のダウンサンプリングを行い、SigLIP-L 経路の特徴とトークン数を合わせているためである。これにより、高解像度の細部情報をもちながらも LLM へ渡すトークン数を一定に抑えられる。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="ハイブリッド視覚エンコーダの特徴次元の流れ。SAM-B経路とSigLIP-L経路がそれぞれ576かける1024の特徴を出し、結合して576かける2048になる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <text x="20" y="35" font-size="13" font-weight="700" fill="#0e7490">高解像度経路（SAM-B）</text>
  <rect x="20" y="55" width="130" height="55" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="85" y="80" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">SAM-B</text>
  <text x="85" y="98" text-anchor="middle" font-size="12" fill="#18181b">1024角入力</text>
  <rect x="190" y="55" width="120" height="55" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="250" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">64x64x256</text>
  <text x="250" y="98" text-anchor="middle" font-size="12" fill="#18181b">特徴マップ</text>
  <rect x="350" y="55" width="175" height="55" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="437" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">補間→畳込x2→整形</text>
  <text x="437" y="98" text-anchor="middle" font-size="12" fill="#18181b">576x1024</text>
  <line x1="150" y1="82" x2="190" y2="82" stroke="#71717a" stroke-width="2"/>
  <polygon points="190,82 182,78 182,86" fill="#71717a"/>
  <line x1="310" y1="82" x2="350" y2="82" stroke="#71717a" stroke-width="2"/>
  <polygon points="350,82 342,78 342,86" fill="#71717a"/>
  <text x="20" y="245" font-size="13" font-weight="700" fill="#4338ca">意味経路（SigLIP-L）</text>
  <rect x="20" y="265" width="130" height="55" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="85" y="290" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">SigLIP-L</text>
  <text x="85" y="308" text-anchor="middle" font-size="12" fill="#18181b">384角入力</text>
  <rect x="350" y="265" width="175" height="55" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="437" y="290" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">意味特徴</text>
  <text x="437" y="308" text-anchor="middle" font-size="12" fill="#18181b">576x1024</text>
  <line x1="150" y1="292" x2="350" y2="292" stroke="#71717a" stroke-width="2"/>
  <polygon points="350,292 342,288 342,296" fill="#71717a"/>
  <rect x="560" y="150" width="140" height="75" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="630" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">結合</text>
  <text x="630" y="200" text-anchor="middle" font-size="12" fill="#18181b">576x2048</text>
  <line x1="525" y1="82" x2="540" y2="82" stroke="#71717a" stroke-width="2"/>
  <line x1="540" y1="82" x2="540" y2="172" stroke="#71717a" stroke-width="2"/>
  <line x1="540" y1="172" x2="560" y2="172" stroke="#71717a" stroke-width="2"/>
  <polygon points="560,172 552,168 552,176" fill="#71717a"/>
  <line x1="525" y1="292" x2="540" y2="292" stroke="#71717a" stroke-width="2"/>
  <line x1="540" y1="292" x2="540" y2="202" stroke="#71717a" stroke-width="2"/>
  <line x1="540" y1="202" x2="560" y2="202" stroke="#71717a" stroke-width="2"/>
  <polygon points="560,202 552,198 552,206" fill="#71717a"/>
  <text x="630" y="255" text-anchor="middle" font-size="12" fill="#16a34a">GeLU + 埋め込み</text>
  <text x="630" y="273" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">→ DeepSeek LLM へ</text>
</svg><figcaption>視覚エンコーダの次元の流れ。<b>要点</b>は「高解像度（SAM-B）と意味（SigLIP-L）の 2 経路がそれぞれ 576x1024 を出し、結合して 576x2048 にする」こと。高解像度を活かしつつトークン数を 576 に固定できる。</figcaption></figure>

コネクタである **VL アダプタは 2 層のハイブリッド MLP** である。まず高解像度特徴と低解像度特徴を**別々の単層 MLP** で処理し、次元方向に結合してから、もう 1 層の MLP で LLM の入力空間へ写す。言語モデル本体は DeepSeek-LLM で、LLaMA に倣ったマイクロ設計（Pre-Norm の RMSNorm、FFN 活性化に SwiGLU、中間次元 $\frac{8}{3}\,d_{\text{model}}$、Rotary 位置埋め込み、DeepSeek-LLM と同一のトークナイザ）を採用する。学習は視覚理解に注力するため、**損失は言語部分の次トークン予測にのみかける**。

## 実世界志向のデータと言語能力の保持

DeepSeek-VL の「実世界志向」は 2 つの面で具体化される。1 つは**データの作り方**、もう 1 つは**言語能力を壊さない学習**である。

データ面では、事前学習コーパスを Common Crawl・Web コード・電子書籍・教育教材・arXiv 論文など多様なソースから構成し、Web スクリーンショット、PDF、OCR、チャート、専門知識（教科書）といった実世界のシナリオを広くカバーする。さらに SFT（教師ありファインチューニング）のために、GPT-4V や Gemini に対する実ユーザーの利用例をネットから収集し、それを **「認識・変換・分析・推論・評価・安全性」などの分類体系（use case taxonomy）** に体系化する。この分類体系を使って各画像に与えるプロンプトを選ぶことで、ベンチマーク用ではなく**実利用に即した指示チューニングデータ**を作る。同じ体系を評価データの設計にも流用する。この分類体系の最上位カテゴリは、おおむね次のように整理されている。

- **認識（Recognition）**: 全体描写・局所描写・OCR/書き起こしなど、知識を多く要さない理解。
- **変換（Conversion）**: 画像からコードへ、画像からテキストへといった形式変換。
- **分析（Analysis）**: データチャートや専門図、百科事典的知識を使った読み解き。
- **常識推論・論理推論**: 関係・機能・環境・異常などの推論や、数学・物理などの論理。
- **評価・複数画像・安全性**: 画質や類似度の評価、複数画像の比較、反事実質問やプロンプトインジェクションへの耐性。

実利用のプロンプトをこうしたカテゴリに割り当てることで、特定タスクに偏らないバランスの良い指示チューニングデータを組み立てている。

事前学習データの内訳を眺めると、この「実世界志向」がさらに具体的に見えてくる。論文の共同事前学習データは、文脈内学習を促す**交互（interleaved）画像-テキスト**（MMC4・Wiki・Wikihow・自前の PDF/Epub 教科書など）、高品質な**画像キャプション**、表やグラフを読むための**テーブル/チャート**、グラフィカル UI からコードを復元する **Web コード**、環境に埋め込まれた文字を読む**シーンテキスト OCR**、arXiv や電子書籍から作った**文書 OCR**など、多彩なカテゴリで構成される。注目すべきは、これらの視覚関連データに対して**純テキストのコーパスが全体のおよそ 7 割を占める**ように配合されている点である。つまり「言語データを 7 割保つ」という方針は、データ表のレベルから一貫して貫かれている。

もう 1 つの柱が**言語能力の保持**である。VLM は視覚データに偏って学習すると、土台の LLM がもともと持っていた言語能力を急速に失う——いわゆる**破滅的忘却（catastrophic forgetting）**——という問題がある。本論文は、(1) 多モーダルコーパスは言語データに比べて単純で分布が乖離していること、(2) 視覚と言語のモダリティ間に競合（competitive dynamics）があること、の 2 点を忘却の原因と見る。対策はシンプルで、**VL 事前学習に言語データを大量に混ぜる**。実験からは、言語データを**少なくとも 7 割**は保つことが言語知識の保全に不可欠で、最終的には**言語 : 多モーダル ≒ 7 : 3**の比率を採用したと述べられている。加えて、学習が進むにつれて多モーダルの比率を徐々に上げていく **「モダリティ・ウォームアップ（modality warm-up）」** を導入し、両モダリティをバランスよく統合する。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="モダリティ混合比の対比。多モーダル100パーセントは言語を忘却し、言語70パーセント多モーダル30パーセントは両立する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <text x="30" y="40" font-size="15" font-weight="700" fill="#18181b">VL 事前学習でのモダリティ混合比</text>
  <rect x="30" y="70" width="270" height="70" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="165" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">多モーダル 100% / 言語 0%</text>
  <text x="165" y="122" text-anchor="middle" font-size="12" fill="#18181b">視覚に偏った学習</text>
  <rect x="410" y="70" width="280" height="70" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="550" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">言語能力が急激に劣化</text>
  <text x="550" y="122" text-anchor="middle" font-size="12" fill="#18181b">破滅的忘却</text>
  <line x1="300" y1="105" x2="410" y2="105" stroke="#dc2626" stroke-width="2"/>
  <polygon points="410,105 400,100 400,110" fill="#dc2626"/>
  <rect x="30" y="220" width="270" height="80" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="165" y="252" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">言語 70% / 多モーダル 30%</text>
  <text x="165" y="276" text-anchor="middle" font-size="12" fill="#18181b">最終比 およそ 7 : 3</text>
  <rect x="410" y="220" width="280" height="80" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="550" y="252" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">言語力を保ちつつ視覚を獲得</text>
  <text x="550" y="276" text-anchor="middle" font-size="12" fill="#18181b">両モダリティを両立</text>
  <line x1="300" y1="260" x2="410" y2="260" stroke="#16a34a" stroke-width="2"/>
  <polygon points="410,260 400,255 400,265" fill="#16a34a"/>
  <text x="360" y="180" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">モダリティ・ウォームアップ：学習が進むほど多モーダル比を徐々に上げる</text>
</svg><figcaption>言語保持の直感。<b>要点</b>は「多モーダルに振り切る（100:0）と言語を忘却するが、言語を 7 割残す（およそ 7:3）と両立する」こと。比率はウォームアップで徐々に調整する。</figcaption></figure>

なお小規模モデル（1.3B）では、事前学習段階に**ごく少量の指示チューニングデータを混ぜておく**工夫も入る。これは「指示に従う能力がボトルネックになって本来の性能が測れない」事態を避けるためで、評価も選択肢のパープレキシティ（PPL）を比べる **multi-choice PPL** 方式に切り替えている。

## 3 段階学習（ウォームアップ → 共同事前学習 → SFT）

学習は連続する 3 段階で進む。各段階で**どのモジュールを凍結し、どこを学習するか**が変わるのが要点である。

1. **Stage 1: VL アダプタのウォームアップ。** 視覚エンコーダと LLM の両方を**凍結**し、容量の小さい VL アダプタだけを学習する。狙いは、視覚と言語を共通の埋め込み空間で結びつける概念的なリンクを作ることである。データには ShareGPT4V 由来の画像-テキスト対（約 125 万）と文書 OCR レンダリング対（約 250 万）を用いる。この段階ではアダプタの容量が小さいため**データ規模を増やしても効果が乏しく、むしろ悪化しうる**ことが実験で示され、これが次段で LLM を解凍する動機になる。
2. **Stage 2: 共同 VL 事前学習。** 視覚エンコーダは**凍結**したまま、**LLM と VL アダプタを学習**する。ここが多モーダル能力を本格的に獲得する中核段階である。前節の通り、純粋な言語データを大量に混ぜる**共同 language-multimodal 学習**を行い、言語 : 多モーダル ≒ 7 : 3 の比率で忘却を抑える。交互（interleaved）の VL データと純言語シーケンスを併用する。
3. **Stage 3: 教師ありファインチューニング（SFT）。** 指示や対話に従う能力を仕上げ、対話モデル **DeepSeek-VL-Chat** を作る段階。**LLM・VL アダプタ・低解像度の SigLIP-L** を学習対象とする一方、**SAM-B は GPU メモリの制約から凍結**したままにする。応答と特殊トークンだけを教師信号とし、システム/ユーザープロンプトはマスクする。VL のチャットデータと純テキストのチャットデータを混ぜることで、多モーダルと言語の双方で破綻しない versatility を確保する。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="3段階学習。各段階で凍結する部分と学習する部分が異なる様子" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
  <text x="150" y="35" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Stage 1 アダプタ温め</text>
  <text x="360" y="35" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Stage 2 共同VL事前学習</text>
  <text x="570" y="35" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Stage 3 SFT</text>
  <rect x="70" y="80" width="160" height="48" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="150" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">DeepSeek LLM</text>
  <text x="150" y="120" text-anchor="middle" font-size="12" fill="#71717a">凍結</text>
  <rect x="70" y="150" width="160" height="48" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="150" y="173" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">VL アダプタ</text>
  <text x="150" y="190" text-anchor="middle" font-size="12" fill="#4338ca">学習</text>
  <rect x="70" y="220" width="160" height="48" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="150" y="243" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">視覚エンコーダ</text>
  <text x="150" y="260" text-anchor="middle" font-size="12" fill="#71717a">凍結</text>
  <text x="150" y="300" text-anchor="middle" font-size="12" fill="#18181b">画像-テキスト対</text>
  <rect x="280" y="80" width="160" height="48" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">DeepSeek LLM</text>
  <text x="360" y="120" text-anchor="middle" font-size="12" fill="#4338ca">学習</text>
  <rect x="280" y="150" width="160" height="48" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="173" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">VL アダプタ</text>
  <text x="360" y="190" text-anchor="middle" font-size="12" fill="#4338ca">学習</text>
  <rect x="280" y="220" width="160" height="48" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="360" y="243" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">視覚エンコーダ</text>
  <text x="360" y="260" text-anchor="middle" font-size="12" fill="#71717a">凍結</text>
  <text x="360" y="300" text-anchor="middle" font-size="12" fill="#18181b">交互VL + 純言語 (7:3)</text>
  <rect x="490" y="80" width="160" height="48" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="570" y="103" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">DeepSeek LLM</text>
  <text x="570" y="120" text-anchor="middle" font-size="12" fill="#4338ca">学習</text>
  <rect x="490" y="150" width="160" height="48" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="570" y="173" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">VL アダプタ</text>
  <text x="570" y="190" text-anchor="middle" font-size="12" fill="#4338ca">学習</text>
  <rect x="490" y="220" width="160" height="48" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="570" y="240" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">SigLIP-L 学習</text>
  <text x="570" y="258" text-anchor="middle" font-size="12" fill="#71717a">SAM-B 凍結</text>
  <text x="570" y="300" text-anchor="middle" font-size="12" fill="#18181b">VLチャット + 純テキスト</text>
</svg><figcaption>3 段階の学習。<b>要点</b>は「Stage 1 はアダプタだけ、Stage 2 でアダプタと LLM、Stage 3 で SigLIP-L まで学習し SAM-B は凍結」という凍結/学習の切り替え。言語データを混ぜながら段階的に視覚を載せる。</figcaption></figure>

段階ごとにハイパーパラメータも切り替わる。アダプタだけを温める Stage 1 は系列長を短く（$512$）とり、共同 VL 事前学習の Stage 2 では系列長を $4096$ に伸ばしてバッチも大きくし、AdamW・Step スケジューラで長く回す。仕上げの Stage 3 は再び比較的軽い設定で指示追従に合わせ込む、という構成である。学習率や系列長を段階に応じて変える狙いは、各段階の「学習する対象の容量」と「扱うデータの性質」に設定を合わせることにある。

土台の重み出しにも工夫がある。DeepSeek-VL-1B は約 5000 億トークンで学習した DeepSeek-LLM-1B を、DeepSeek-VL-7B は約 2 兆トークンで学習した DeepSeek-LLM-7B を起点とする。学習は HAI-LLM フレームワーク上で行われ、視覚エンコーダとテキスト埋め込みを「結果モデルの第 1 層」とみなすことでパイプライン並列に載せやすくしている、という実装上の工夫も述べられている。計算規模としては、7B は 8 枚の NVIDIA A100 を積んだ 64 ノードで約 5 日、1B は 16 ノードで約 7 日を要したと報告されている。

## まとめと、読解後に答えたい問い

DeepSeek-VL の核心は、**「実世界で使えること」を 3 つの設計判断に翻訳した点**にある。すなわち、(1) 細部と意味を同時に扱う**ハイブリッド視覚エンコーダ（SigLIP-L + SAM-B）**、(2) 実利用の分類体系に基づく**実世界志向のデータ設計**、(3) 言語データを 7 割保ちウォームアップで比率を調整する**言語保持の学習戦略**である。これらを「アダプタ温め → 共同 VL 事前学習 → SFT」の 3 段階で組み上げ、$1024 \times 1024$ の高解像度入力を 576 トークンの固定予算に収めながら、言語能力を損なわずに視覚理解を獲得する。

読解後、自分の言葉で答えられるか確認したい問い:

- **なぜ視覚エンコーダを 2 本にするのか。** SigLIP 単体の "CLIP-blind pairs" と低解像度の限界を、SAM-B の高解像度・低レベル特徴がどう補うかを説明できるか。
- **576 という固定トークン予算は何を狙ったものか。** 高解像度を活かすことと推論コスト・インターリーブ/多ターン対話の両立、というトレードオフを説明できるか。
- **言語能力の保持はなぜ必要で、どう実現したか。** 破滅的忘却の原因（多モーダルの単純さ・モダリティ競合）と、言語 : 多モーダル ≒ 7 : 3 やモダリティ・ウォームアップという対策を結びつけられるか。
- **3 段階で凍結対象が変わる理由は何か。** Stage 1 でアダプタだけ学習する根拠（データ規模を増やしても効かない）、Stage 3 で SAM-B だけ凍結する理由（GPU メモリ）を言えるか。
- **小規模モデルで multi-choice PPL や少量 SFT 混入を使う狙いは何か。** ベンチマークの不安定さと指示追従のボトルネックを、どう切り分けて対処したかを説明できるか。
