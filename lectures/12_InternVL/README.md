# InternVL — Scaling up Vision Foundation Models（大規模 InternViT）

LLaVA や BLIP-2 を学んだ私たちは、「視覚エンコーダは凍結した CLIP/ViT を使い、薄い接続層（線形射影や Q-Former）で LLM へ繋ぐ」という構図に慣れています。InternVL（OpenGVLab, CVPR 2024）はこの常識を一度疑い、**「そもそも視覚エンコーダが小さすぎるのではないか」** という別の問題意識から出発します。LLM が数十〜数百億パラメータに達する一方で、視覚側は数億で止まったまま。この**規模の非対称**を、視覚エンコーダを 6B 規模（InternViT-6B）まで拡大し、さらに言語側へと**段階的に整列**することで埋めにいくのが本モデルの核心です。本ページでは、この「視覚を大きくして、丁寧に橋を架ける」発想を腹落ちさせ、最後に発展研究 ACC（Adaptive Cluster-CLIP）との設計思想の対比へ繋ぎます。

## 問題意識：視覚エンコーダが小さすぎる

InternVL 論文が最初に指摘するのは、視覚と言語のあいだに横たわる二つのギャップです。

- **パラメータ規模の非対称（disparity in parameter scales）**：論文の言葉では、大規模 LLM は今や 1 兆（1000 billion）規模にまで膨らむ一方、VLLM が使う視覚エンコーダは依然として「およそ 10 億」前後にとどまる。視覚側が小さいと、せっかくの LLM の容量を活かしきれない（under-use of LLM's capacity）。
- **表現の不整合（inconsistent representation）**：純視覚データや BERT 系で学習された視覚特徴は、LLM の表現空間と素直に揃わない。
- **非効率な接続（inefficient connection）**：BLIP-2 の Q-Former や LLaVA の線形層といった「glue（糊）」は軽量かつランダム初期化で、マルチモーダルに必要な豊かな相互作用を捉えきれないことがある。

LLaVA 系は 3 番目の「接続」を、BLIP-2 は「接続＋表現整列」を主に攻めました。InternVL はこれらに加えて、**1 番目の「規模そのもの」** に正面から取り組みます。発想（inspiration）は明快で、論文の表現を借りれば「視覚エンコーダを LLM のパラメータ規模まで引き上げ（elevating the vision encoder）、続いて両者の表現を調和させる（harmonizing their representations）」というものです。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="従来は小さな視覚エンコーダと大きなLLMで規模が非対称だが、InternVLはInternViT-6Bで視覚側を拡大して規模を揃えることを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="170" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">従来の VLLM</text><rect x="70" y="200" width="90" height="55" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="115" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ViT</text><text x="115" y="248" text-anchor="middle" font-size="11" fill="#155e75">約0.3〜1B</text><rect x="195" y="80" width="90" height="175" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="240" y="165" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">LLM</text><text x="240" y="183" text-anchor="middle" font-size="11" fill="#3730a3">数十〜数百B</text><text x="178" y="60" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">規模の非対称</text><line x1="345" y1="170" x2="405" y2="170" stroke="#71717a" stroke-width="2"/><polygon points="405,170 395,165 395,175" fill="#71717a"/><text x="555" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">InternVL</text><rect x="440" y="80" width="100" height="175" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="490" y="160" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">InternViT</text><text x="490" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">-6B</text><rect x="560" y="80" width="100" height="175" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="610" y="165" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">LLM</text><text x="610" y="183" text-anchor="middle" font-size="11" fill="#3730a3">数十B</text><text x="550" y="60" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">規模を整列</text></svg><figcaption>従来は視覚エンコーダ（小）と LLM（大）で<b>規模が非対称</b>でした。InternVL は視覚側を <b>InternViT-6B</b> まで拡大し、LLM とつり合う土俵に乗せます。</figcaption></figure>

つまり InternVL は「LLM はそのままに、間の糊を工夫する」のではなく、「**視覚エンコーダ自体を巨大化し、巨大なまま LLM へ整列させる**」という別系統の道を選びます。次節でその全体像を一枚で押さえましょう。

## 全体像（まず一枚で）

InternVL は三つの大きな部品から成ります。

| 部品 | 役割 | 規模・初期化 |
| --- | --- | --- |
| **InternViT-6B** | 大規模視覚エンコーダ（vanilla ViT を 6B まで拡大） | 約 6B パラメータ。ランダム初期化から学習 |
| **QLLaMA** | 言語ミドルウェア（視覚特徴を言語へ橋渡しする中間層） | 約 8B。多言語強化 LLaMA で初期化＋学習可能クエリを追加 |
| **LLM デコーダ** | 応答生成を担う既製の LLM | 例：Vicuna / InternLM（既製品をそのまま接続） |

論文はこれを「**Swiss Army knife（万能ナイフ）型モデル**」と呼びます。InternViT-6B 単体は視覚知覚（分類・セグメンテーション）に使え、InternViT-6B＋QLLaMA は対照学習（ゼロショット検索・分類）に、さらに LLM を繋げばマルチモーダル対話（InternVL-Chat）に使える、という具合に同じ部品を組み替えて多様なタスクへ展開できる設計です。

<figure class="lec-fig"><svg viewBox="0 0 720 300" role="img" aria-label="画像がInternViT-6BからQLLaMAを経てLLMへ流れ、応答や分類を出力するInternVLの全体構成図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="20" y="120" width="70" height="55" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="55" y="153" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">画像</text><line x1="92" y1="147" x2="128" y2="147" stroke="#71717a" stroke-width="2"/><polygon points="128,147 118,142 118,152" fill="#71717a"/><rect x="130" y="95" width="135" height="105" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="197" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">InternViT-6B</text><text x="197" y="160" text-anchor="middle" font-size="11" fill="#155e75">視覚特徴（≈6B）</text><line x1="267" y1="147" x2="303" y2="147" stroke="#71717a" stroke-width="2"/><polygon points="303,147 293,142 293,152" fill="#71717a"/><rect x="305" y="95" width="135" height="105" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="372" y="135" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">QLLaMA</text><text x="372" y="153" text-anchor="middle" font-size="11" fill="#3730a3">言語ミドルウェア</text><text x="372" y="170" text-anchor="middle" font-size="11" fill="#3730a3">学習可能クエリ</text><line x1="442" y1="147" x2="478" y2="147" stroke="#71717a" stroke-width="2"/><polygon points="478,147 468,142 468,152" fill="#71717a"/><rect x="480" y="95" width="135" height="105" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="547" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">LLM デコーダ</text><text x="547" y="160" text-anchor="middle" font-size="11" fill="#3730a3">Vicuna / InternLM</text><line x1="617" y1="147" x2="650" y2="147" stroke="#71717a" stroke-width="2"/><polygon points="650,147 640,142 640,152" fill="#71717a"/><rect x="652" y="120" width="60" height="55" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="682" y="143" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">応答</text><text x="682" y="159" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">分類</text><text x="197" y="225" text-anchor="middle" font-size="11" fill="#71717a">単体で視覚知覚にも使える</text><text x="372" y="225" text-anchor="middle" font-size="11" fill="#71717a">大規模なQ-Former的中間層</text><text x="547" y="225" text-anchor="middle" font-size="11" fill="#71717a">既製LLMをそのまま接続</text></svg><figcaption>全体像。<b>画像 → InternViT-6B → QLLaMA → LLM</b> という流れ。視覚も中間層も大きく取り、LLM とつり合う表現を作ってから橋渡しします。</figcaption></figure>

ここで注目したいのは中央の **QLLaMA** です。LLaVA の線形層や BLIP-2 の Q-Former に相当する位置にありながら、その規模と作りが大きく異なります。節を改めて見ていきましょう。

## QLLaMA という言語ミドルウェア

QLLaMA は「視覚特徴を言語へ橋渡しする中間層」です。役割としては BLIP-2 の Q-Former と同じく、**多数の視覚トークンを少数の学習可能クエリへ圧縮し、テキスト空間へ整える**ものですが、設計思想が異なります。論文によれば、QLLaMA は**多言語強化された LLaMA で初期化**され、そこに**新規の学習可能クエリ（96 個）とクロスアテンション層を追加**したものです。追加分（クエリ＋クロスアテンション）はランダム初期化で約 1B、土台の LLaMA を合わせて全体で約 8B に達します。

論文は、軽量な glue 層に対する QLLaMA の利点を三つ挙げています。

1. **言語知識を初めから持つ**：LLaMA の事前学習済み重みで初期化されるため、InternViT-6B が出す画像トークンを「最初から LLM と揃った表現」へ変換しやすい。ゼロから糊を学ぶ Q-Former との大きな違いです。
2. **桁違いに大きい容量**：QLLaMA の視覚言語整列用パラメータは約 8B で、論文は「QFormer の約 42 倍」と述べます。大規模な視覚特徴を受け止めるだけの容量を持つ、ということです。
3. **対照学習にも生成にも使える**：クエリ特徴を使えば画像テキスト検索などの対照タスクに、テキストを続けて生成すればキャプショニングなどの生成タスクに、と両用できる。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="InternViT-6Bの多数の視覚トークンを学習可能クエリとクロスアテンションで受け、LLaMA重みを通してLLMと整合した少数トークンへ変換するQLLaMAの内部構造図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="25" y="125" width="120" height="90" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="85" y="160" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">InternViT-6B</text><text x="85" y="180" text-anchor="middle" font-size="11" fill="#155e75">視覚特徴</text><text x="85" y="196" text-anchor="middle" font-size="11" fill="#155e75">（多数トークン）</text><rect x="190" y="40" width="135" height="55" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="257" y="63" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">学習可能クエリ ×96</text><text x="257" y="81" text-anchor="middle" font-size="11" fill="#3730a3">（新規・ランダム初期化）</text><rect x="190" y="150" width="135" height="75" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="257" y="183" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">クロスアテンション</text><text x="257" y="201" text-anchor="middle" font-size="11" fill="#3730a3">視覚→クエリへ集約</text><line x1="147" y1="170" x2="186" y2="187" stroke="#71717a" stroke-width="2"/><polygon points="186,187 175,184 178,175" fill="#71717a"/><line x1="257" y1="97" x2="257" y2="146" stroke="#71717a" stroke-width="2"/><polygon points="257,146 252,136 262,136" fill="#71717a"/><rect x="370" y="110" width="150" height="120" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="445" y="158" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">LLaMA 重み</text><text x="445" y="178" text-anchor="middle" font-size="11" fill="#3730a3">で初期化</text><text x="445" y="196" text-anchor="middle" font-size="11" fill="#3730a3">（言語知識を内蔵）</text><line x1="325" y1="180" x2="366" y2="175" stroke="#71717a" stroke-width="2"/><polygon points="366,175 356,171 357,181" fill="#71717a"/><line x1="522" y1="170" x2="558" y2="170" stroke="#71717a" stroke-width="2"/><polygon points="558,170 548,165 548,175" fill="#71717a"/><rect x="560" y="120" width="135" height="100" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="627" y="158" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">LLMと整合した</text><text x="627" y="178" text-anchor="middle" font-size="11" fill="#166534">少数の視覚トークン</text><text x="360" y="280" text-anchor="middle" font-size="12" fill="#71717a">合計 約8B（≒ QFormer の約42倍）。対照学習にも生成にも使える</text></svg><figcaption>QLLaMA の内部。<b>学習可能クエリ＋クロスアテンション</b>で大量の視覚特徴を少数へ集約し、<b>LLaMA 重み</b>の言語知識を通して LLM と揃った視覚トークンへ変換します。</figcaption></figure>

ここまでで「大きな視覚エンコーダ」と「大きな言語ミドルウェア」が揃いました。しかし、これらを最初から end-to-end で繋いで学習するのは、規模も大きくデータも雑多なため不安定です。InternVL はこれを**段階的な整列**で乗り越えます。

## 3段階の段階的整列

InternVL の学習は三つの段階を順に踏みます（論文 Fig.3 の training strategy）。狙いは、**Web 規模のノイズの多いデータから始めて、徐々に高品質データへ移りながら、大きな視覚モデルを LLM へ整列させていく**ことです。各段階で「何が学習され（炎）、何が凍結されるか（雪）」が切り替わる点が要諦です。

| 段階 | 接続される部品 | 損失 | 学習対象 | データの性質 |
| --- | --- | --- | --- | --- |
| ① 対照学習 | InternViT-6B ＋ LLaMA-7B（テキスト側） | CLIP 的な対称交差エントロピー（対照損失） | 両者とも学習 | Web 規模・ノイズの多い画像テキスト |
| ② 生成学習 | InternViT-6B → QLLaMA | BLIP-2 流（ITC＋ITM＋ITG の三損失） | 新規クエリ／クロスアテンションのみ学習、土台は凍結 | フィルタ済みの高品質データ |
| ③ 教師あり SFT | InternViT-6B → QLLaMA →（MLP）→ LLM | 生成損失 | MLP（と必要なら QLLaMA） | 指示・対話の高品質データ（約 400 万件規模） |

各段階の意味を補足します。

- **① 対照学習（Vision-Language Contrastive Training）**：InternViT-6B を、多言語 LLaMA-7B を流用したテキストエンコーダと組み、CLIP と同様に画像テキストペアで対照学習します。LAION-en / LAION-multi / LAION-COCO / COYO / Wukong といった**Web 規模のノイズの多いデータ**を用い、極端に低品質なものだけ除いて大量に回す段階です。ここで InternViT-6B は、言語と整合した強い視覚表現を獲得します。
- **② 生成学習（Vision-Language Generative Training）**：①で得た InternViT-6B と QLLaMA を凍結し、QLLaMA に**新たに足したクエリとクロスアテンション層だけ**を学習します。損失は BLIP-2 と同じ三本立て（画像テキスト対照 ITC、画像テキストマッチング ITM、画像由来テキスト生成 ITG）。データは①よりさらにフィルタした高品質なものへ絞り込みます。
- **③ 教師あり SFT（Supervised Fine-tuning）**：InternViT-6B＋QLLaMA を、MLP 層を介して**既製の LLM デコーダ**（Vicuna や InternLM）へ接続し、指示・対話データで微調整します。QLLaMA と LLM の特徴空間が近いため、LLM デコーダを凍結したまま MLP（必要なら QLLaMA も）だけ学習しても良好に動く、と論文は述べます。

<figure class="lec-fig"><svg viewBox="0 0 720 420" role="img" aria-label="InternVLの3段階の段階的整列。第1段階は対照学習、第2段階は生成学習、第3段階は教師ありSFTで、段階ごとに学習対象と接続部品が変わることを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="15" y="20" width="690" height="115" rx="8" fill="#f4f4f5" stroke="#e4e4e7" stroke-width="2"/><text x="30" y="48" font-size="13" font-weight="700" fill="#18181b">① 対照学習</text><text x="30" y="66" font-size="11" fill="#71717a">Web規模ノイズ</text><rect x="170" y="45" width="160" height="60" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="250" y="72" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">InternViT-6B</text><text x="250" y="90" text-anchor="middle" font-size="11" fill="#dc2626">学習</text><rect x="430" y="45" width="160" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="510" y="72" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">LLaMA-7B（テキスト）</text><text x="510" y="90" text-anchor="middle" font-size="11" fill="#dc2626">学習</text><text x="380" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">対照</text><line x1="332" y1="80" x2="428" y2="80" stroke="#71717a" stroke-width="2"/><polygon points="428,80 418,75 418,85" fill="#71717a"/><polygon points="332,80 342,75 342,85" fill="#71717a"/><rect x="15" y="150" width="690" height="115" rx="8" fill="#f4f4f5" stroke="#e4e4e7" stroke-width="2"/><text x="30" y="178" font-size="13" font-weight="700" fill="#18181b">② 生成学習</text><text x="30" y="196" font-size="11" fill="#71717a">高品質に絞る</text><rect x="170" y="175" width="150" height="60" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="245" y="202" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">InternViT-6B</text><text x="245" y="220" text-anchor="middle" font-size="11" fill="#0e7490">凍結</text><line x1="322" y1="205" x2="368" y2="205" stroke="#71717a" stroke-width="2"/><polygon points="368,205 358,200 358,210" fill="#71717a"/><rect x="370" y="175" width="150" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="445" y="202" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">QLLaMA</text><text x="445" y="220" text-anchor="middle" font-size="11" fill="#dc2626">新規クエリのみ学習</text><text x="615" y="200" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">ITC+ITM</text><text x="615" y="216" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">+ITG</text><line x1="522" y1="205" x2="558" y2="205" stroke="#71717a" stroke-width="2"/><polygon points="558,205 548,200 548,210" fill="#71717a"/><rect x="15" y="280" width="690" height="120" rx="8" fill="#f4f4f5" stroke="#e4e4e7" stroke-width="2"/><text x="30" y="308" font-size="13" font-weight="700" fill="#18181b">③ 教師ありSFT</text><text x="30" y="326" font-size="11" fill="#71717a">指示・対話</text><rect x="150" y="310" width="120" height="60" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="210" y="337" text-anchor="middle" font-size="11" font-weight="700" fill="#155e75">InternViT-6B</text><text x="210" y="355" text-anchor="middle" font-size="11" fill="#0e7490">凍結</text><line x1="272" y1="340" x2="298" y2="340" stroke="#71717a" stroke-width="2"/><polygon points="298,340 288,335 288,345" fill="#71717a"/><rect x="300" y="310" width="110" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="355" y="343" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">QLLaMA</text><line x1="412" y1="340" x2="438" y2="340" stroke="#71717a" stroke-width="2"/><polygon points="438,340 428,335 428,345" fill="#71717a"/><rect x="440" y="320" width="70" height="40" rx="6" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/><text x="475" y="345" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">MLP</text><line x1="512" y1="340" x2="538" y2="340" stroke="#71717a" stroke-width="2"/><polygon points="538,340 528,335 528,345" fill="#71717a"/><rect x="540" y="310" width="150" height="60" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="615" y="337" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">LLM（Vicuna等）</text><text x="615" y="355" text-anchor="middle" font-size="11" fill="#0e7490">凍結も可</text></svg><figcaption>3段階の段階的整列。<b>①対照（CLIP的・大規模ノイズ）→ ②生成（QLLaMAを足して高品質に絞る）→ ③SFT（既製LLMへ接続）</b>。段階ごとに学習対象（赤＝学習／青緑＝凍結）が切り替わります。</figcaption></figure>

この「下流ほど高品質・凍結を増やす」進め方が、6B 規模の視覚モデルを安定して LLM へ整列させる鍵です。結果として InternVL は、視覚知覚（ImageNet 線形評価・ADE20K セグメンテーション）、ゼロショット画像／動画分類・検索、キャプショニング、マルチモーダル対話まで、論文が挙げる多数の汎用視覚言語ベンチマークで強い性能を示します。「視覚を大きくし、丁寧に橋を架ける」という投資が、知覚と VL タスクの両方で効く、というのが InternVL の主張です。

## ACC 研究との関連

ここで、発展研究 **ACC（Adaptive Cluster-CLIP）** と設計思想を対比してみましょう。ACC は、エッジ実時間での物体中心な動画フレーム検索を狙う軽量手法です。要点は次の通りです。

- **軽量な Dense-CLIP 局所特徴**：巨大な視覚エンコーダではなく、**CLIP ResNet-50** ベースの Dense-CLIP から得る局所（パッチ）特徴を使う。
- **クラスタリングで圧縮**：多数の局所特徴をクラスタリングし、**少数の集約ベクトル**へ畳み込む。フレームを「少数の物体中心ベクトル」で表す。
- **クエリ非依存インデックス**：この集約表現は特定のクエリに依存しない。一度作ったインデックスを**多数のクエリで再利用**できる。
- **エッジ実時間**：Jetson AGX Orin 上で実時間相当のインデックス構築を達成する、エッジ志向の設計。

InternVL と ACC は、同じ「視覚特徴を活かす」課題に対して**正反対の投資先**を選んでいます。

| 観点 | InternVL | ACC（Adaptive Cluster-CLIP） |
| --- | --- | --- |
| 視覚エンコーダ | InternViT-6B（約 6B、巨大化） | CLIP ResNet-50（軽量 Dense-CLIP） |
| 狙い | 大規模・高精度（汎用 VL タスク全般） | 軽量・エッジ実時間（物体中心フレーム検索） |
| 特徴の使い方 | 言語へ段階的に整列し LLM で生成 | 局所特徴をクラスタ集約し索引化 |
| クエリとの関係 | クエリ（指示文）に応じて推論 | クエリ非依存の索引を多数クエリで再利用 |
| 計算資源 | 大規模 GPU 学習を前提 | エッジ端末（Jetson）で動く |

つまり、**「大規模視覚・高精度（InternVL）」と「軽量・クエリ非依存・索引再利用（ACC）」** という二つの設計極があります。InternVL は規模に投資して汎用性と精度を、ACC は軽さと再利用性に投資して実時間性を取りにいく。どちらが正しいという話ではなく、**精度を求めて視覚を大きくするのか、運用制約のもとで賢く圧縮・再利用するのか**という、目的に応じた設計判断の違いです。InternVL を読んでおくと、ACC が「なぜあえて巨大エンコーダを避け、軽量 Dense-CLIP の局所特徴を選ぶのか」という設計の裏返しが、より鮮明に見えてきます。

## まとめと、読解後に答えたい問い

InternVL の核心を改めて一言でまとめると、**「視覚エンコーダ（InternViT-6B）を LLM 並みに大きくし、言語ミドルウェア（QLLaMA）を介して 3 段階で丁寧に整列する」** ことで、視覚と言語の規模・表現のギャップを埋めるモデル、です。LLaVA 系が「接続層の工夫」、BLIP-2 が「Q-Former による表現整列」に重心を置いたのに対し、InternVL は「**規模そのもの**」へ正面から投資した別系統の発想だと位置づけられます。

読解後に、自分の言葉で答えられるか確認したい問いを挙げます。

1. InternVL が指摘する視覚と言語の三つのギャップ（規模・表現・接続）のうち、LLaVA や BLIP-2 が主に攻めたのはどれで、InternVL が新たに加えたのはどれか。
2. QLLaMA は BLIP-2 の Q-Former と役割が似ているのに、なぜ「LLaMA 重みで初期化」し「約 8B まで大きく」したのか。その二つの設計がそれぞれ何を狙っているか。
3. 3 段階の整列で、段階が進むほど「データは高品質へ、学習対象は凍結を増やす」方向に動くのはなぜか。最初から end-to-end で繋がないことの利点は何か。
4. 同じ InternViT-6B / QLLaMA を組み替えるだけで、視覚知覚・対照検索・対話までこなせるのはなぜか（「Swiss Army knife」と呼ばれる所以）。
5. ACC の「軽量・クエリ非依存・索引再利用」と InternVL の「大規模・高精度」は、それぞれどんな運用条件で有利になるか。あなたのタスクならどちらの極に寄せるか。
