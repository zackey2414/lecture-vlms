# Janus — Decoupling Visual Encoding for Unified Understanding & Generation

Janus（DeepSeek-AI ほか、arXiv:2410.13848）は、**画像を「理解する」ことと「生成する」ことを、ひとつのモデルで両立させる**ための統合マルチモーダルフレームワークである。ローマ神話で前後に二つの顔を持つ神「ヤヌス」が名前の由来で、相反する二つの方向（理解と生成）を一身に抱えるというモデルの性格をそのまま表している。

この種の統合モデルは以前から存在したが、多くは「画像を1本の視覚エンコーダで符号化し、その表現を理解にも生成にも使い回す」という設計だった（Chameleon などが代表例）。Janus の主張はシンプルで強い。**理解と生成では画像に求める情報の粒度がそもそも違うのだから、視覚エンコーディングの経路を分離（decouple）すべき**、というものである。理解には意味重視のエンコーダ（SigLIP 系）、生成には画像合成向けの VQ トークナイザを充て、その後段で**単一の自己回帰トランスフォーマ**が両者をまとめて処理する。

自己回帰モデルとしての中身は、言語モデルと同じく系列の同時確率を条件付き確率の積に分解する素朴な形である。

$$p(x)=\prod_{i} p\!\left(x_i \mid x_{<i}\right)$$

ここで $x_i$ はテキストトークンであることもあれば画像トークンであることもある。理解では「画像＋質問」を条件にテキストを予測し、生成では「テキスト指示」を条件に画像トークンを予測する。**入口（エンコーダ）は二手に分けるが、出口側の系列モデリングは一本化する**——これが Janus の核である。

## 全体像（まず一枚で）

まず構成要素を俯瞰しよう。Janus は大きく4つのパーツからなる。

- **理解用エンコーダ**（Understanding Encoder）: SigLIP 系の意味的エンコーダ。画像から高レベルの意味的特徴を取り出す。2次元の特徴マップを1次元系列へ平坦化し、2層 MLP の「理解アダプタ」で言語モデルの入力空間へ写像する。
- **生成用トークナイザ**（Generation Encoder）: VQ ベースのトークナイザ（LlamaGen 由来のもの）。画像を離散 ID の列へ変換する。各 ID のコードブック埋め込みを、やはり2層 MLP の「生成アダプタ」で言語モデルの入力空間へ写像する。
- **テキストトークナイザ**: 言語モデル内蔵のトークナイザ。指示文・質問文を離散 ID 化する。
- **単一の自己回帰トランスフォーマ**: 上記すべての特徴を一本の系列として受け取り、テキストと画像トークンの両方を生成する本体（土台は約1.3B の DeepSeek-LLM）。

下図は、二つの入力経路がひとつのトランスフォーマに合流し、そこから理解（テキスト）と生成（画像）の二つの出力が出ていく様子を一枚にまとめたものである。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="理解用エンコーダと生成用トークナイザという二つの入力経路が、単一の自己回帰トランスフォーマに合流し、テキストと画像の二つの出力を生む全体像" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="24" y="88" width="120" height="48" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="84" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">理解用入力</text>
  <text x="84" y="127" text-anchor="middle" font-size="11" fill="#4338ca">画像 ＋ 質問</text>
  <rect x="176" y="88" width="150" height="48" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="251" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">SigLIPエンコーダ</text>
  <text x="251" y="127" text-anchor="middle" font-size="11" fill="#4338ca">意味的な特徴</text>
  <line x1="144" y1="112" x2="168" y2="112" stroke="#4338ca" stroke-width="2"/>
  <polygon points="168,107 176,112 168,117" fill="#4338ca"/>
  <rect x="24" y="224" width="120" height="48" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="84" y="246" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">生成用入力</text>
  <text x="84" y="263" text-anchor="middle" font-size="11" fill="#0e7490">テキスト指示</text>
  <rect x="176" y="224" width="150" height="48" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="251" y="246" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">VQトークナイザ</text>
  <text x="251" y="263" text-anchor="middle" font-size="11" fill="#0e7490">離散画像トークン</text>
  <line x1="144" y1="248" x2="168" y2="248" stroke="#0e7490" stroke-width="2"/>
  <polygon points="168,243 176,248 168,253" fill="#0e7490"/>
  <rect x="372" y="84" width="150" height="192" rx="10" fill="#f4f4f5" stroke="#18181b" stroke-width="2"/>
  <text x="447" y="164" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">単一の</text>
  <text x="447" y="184" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">自己回帰</text>
  <text x="447" y="204" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">トランスフォーマ</text>
  <text x="349" y="104" text-anchor="middle" font-size="10" fill="#71717a">アダプタ</text>
  <text x="349" y="240" text-anchor="middle" font-size="10" fill="#71717a">アダプタ</text>
  <line x1="326" y1="112" x2="364" y2="112" stroke="#4338ca" stroke-width="2"/>
  <polygon points="364,107 372,112 364,117" fill="#4338ca"/>
  <line x1="326" y1="248" x2="364" y2="248" stroke="#0e7490" stroke-width="2"/>
  <polygon points="364,243 372,248 364,253" fill="#0e7490"/>
  <rect x="572" y="88" width="124" height="48" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="634" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">テキスト出力</text>
  <text x="634" y="127" text-anchor="middle" font-size="11" fill="#4338ca">理解の答え</text>
  <rect x="572" y="224" width="124" height="48" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="634" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像トークン→復号</text>
  <text x="634" y="263" text-anchor="middle" font-size="11" fill="#0e7490">生成画像</text>
  <line x1="522" y1="112" x2="564" y2="112" stroke="#4338ca" stroke-width="2"/>
  <polygon points="564,107 572,112 564,117" fill="#4338ca"/>
  <line x1="522" y1="248" x2="564" y2="248" stroke="#0e7490" stroke-width="2"/>
  <polygon points="564,243 572,248 564,253" fill="#0e7490"/>
</svg><figcaption>入口は<b>理解用（SigLIP）と生成用（VQ）の二経路に分離</b>、出口側は<b>単一の自己回帰トランスフォーマに一本化</b>。同じ本体がテキストも画像トークンも出力する。</figcaption></figure>

注目してほしいのは、トランスフォーマが**1つしかない**点である。理解と生成で別々のモデルを学習するのではなく、テキスト対話・画像理解・画像生成のデータを混ぜて同じ本体に流し込む。理解と多モーダル理解のテキスト予測には言語モデル内蔵の予測ヘッドを、画像生成のトークン予測にはランダム初期化した画像ヘッドを使う。特別なアテンションマスクの設計などは要らず、素直な自己回帰の枠組みに収まっている。

## なぜ視覚エンコーディングを分離するのか

ここが Janus の心臓部である。なぜ「1本のエンコーダで両方」ではいけないのか。

理由は、**理解と生成が画像に要求する情報の性質が正反対だから**である。

- **理解タスク**では、エンコーダの役割は高レベルの意味情報（物体カテゴリ、属性、関係）を抽出することにある。出力はさらに複雑な意味的推論にかけられるため、表現の粒度は「抽象的で高次元の意味」に寄ったほうがよい。細部のピクセルは、むしろ削ぎ落としても構わない。
- **生成タスク**では、主眼は局所的なディテールを作り込み、画像全体の整合性を保つことにある。ここで必要なのは、空間的な構造やテクスチャを忠実に再構成できる、低次元で稠密な符号化である。意味だけでは絵は描けない。

この二つを**同じ表現空間に押し込めると衝突（トレードオフ）が起きる**。意味に寄せれば生成のディテールが痩せ、再構成に寄せれば理解の意味抽出が鈍る。実際、従来の統合モデルは理解性能で「理解専用モデル」に明確に見劣りすることが多かった。Janus はこの緊張を、**そもそも経路を分けることで根本から解消する**。理解には理解に最適なエンコーダ、生成には生成に最適なトークナイザを、互いに気兼ねなく選べるようにするわけだ。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="1つのエンコーダに意味重視と再構成重視という相反する要求が同時にかかって衝突する様子と、経路を分離して理解用エンコーダと生成用トークナイザに振り分けることで衝突を解消する様子" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="36" y="56" width="168" height="48" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="120" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">理解の要求</text>
  <text x="120" y="95" text-anchor="middle" font-size="11" fill="#4338ca">意味・高レベル抽象</text>
  <rect x="36" y="150" width="168" height="60" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="120" y="176" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">1つの視覚エンコーダ</text>
  <text x="120" y="194" text-anchor="middle" font-size="11" fill="#dc2626">で両方を担う</text>
  <rect x="36" y="256" width="168" height="48" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="120" y="278" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">生成の要求</text>
  <text x="120" y="295" text-anchor="middle" font-size="11" fill="#0e7490">詳細・低レベル再構成</text>
  <line x1="120" y1="104" x2="120" y2="144" stroke="#4338ca" stroke-width="2"/>
  <polygon points="114,144 120,151 126,144" fill="#4338ca"/>
  <line x1="120" y1="256" x2="120" y2="216" stroke="#0e7490" stroke-width="2"/>
  <polygon points="114,216 120,209 126,216" fill="#0e7490"/>
  <text x="120" y="234" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">衝突・トレードオフ</text>
  <line x1="214" y1="180" x2="300" y2="180" stroke="#16a34a" stroke-width="3"/>
  <polygon points="300,172 314,180 300,188" fill="#16a34a"/>
  <text x="264" y="166" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">経路を分離</text>
  <rect x="326" y="92" width="190" height="54" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="421" y="114" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">理解エンコーダ：SigLIP</text>
  <text x="421" y="132" text-anchor="middle" font-size="11" fill="#4338ca">意味を抽出</text>
  <rect x="326" y="216" width="190" height="54" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="421" y="238" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">生成トークナイザ：VQ</text>
  <text x="421" y="256" text-anchor="middle" font-size="11" fill="#0e7490">詳細を保持</text>
  <text x="421" y="188" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">衝突を解消</text>
  <rect x="560" y="110" width="130" height="142" rx="10" fill="#f4f4f5" stroke="#18181b" stroke-width="2"/>
  <text x="625" y="174" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">単一</text>
  <text x="625" y="194" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">トランスフォーマ</text>
  <line x1="516" y1="119" x2="552" y2="119" stroke="#4338ca" stroke-width="2"/>
  <polygon points="552,114 560,119 552,124" fill="#4338ca"/>
  <line x1="516" y1="243" x2="552" y2="243" stroke="#0e7490" stroke-width="2"/>
  <polygon points="552,238 560,243 552,248" fill="#0e7490"/>
</svg><figcaption><b>左：1本のエンコーダに意味と再構成という相反する要求が同時にかかり衝突</b>。<b>右：経路を分離</b>して各タスクに最適な符号化を選び、衝突を解消したうえで単一トランスフォーマへ。</figcaption></figure>

この「分離」の効用は、論文のアブレーションでも裏づけられている。理解と生成を1本の VQ トークナイザで賄う構成は、生成は満足にこなせても理解ベンチで大きく劣る。逆に意味的トークナイザに寄せても、両立構成では理解専用構成に届かない。これに対し、SigLIP（理解）＋ VQ（生成）に分離した Janus は、理解性能を大きく引き上げつつ生成も保つ。**衝突の正体は「共有エンコーダ」にあった**、という診断が効いている格好だ。

分離のもう一つの利点は**拡張性**である。経路が独立しているので、理解側だけより強力なエンコーダに差し替えたり、生成側だけより細かい粒度のトークナイザに替えたりが自由にできる。原理的には点群・触覚・EEG など、新しいモダリティ用のエンコーダを足して同じトランスフォーマに合流させることも見据えられている。

## 単一の自己回帰トランスフォーマで理解と生成

入口で分けた特徴は、最終的に**ひとつの系列**へ連結され、共通のトランスフォーマが次トークン予測で処理する。理解と生成の違いは「何を条件にして何を予測するか」だけに整理される。

- **理解**: 画像トークン（SigLIP 由来）と質問のテキストトークンを条件に、答えのテキストトークンを順に予測する。
- **生成**: テキスト指示のトークンを条件に、画像トークン（VQ コードブックの ID）を順に予測する。生成された ID 列は最後に画像デコーダで絵に戻す。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="理解では画像と質問を条件にテキストを予測し、生成ではテキスト指示を条件に画像トークンを予測する、共通の次トークン予測系列の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="24" y="60" font-size="13" font-weight="700" fill="#18181b">理解：画像 ＋ 質問 → テキストを生成</text>
  <rect x="40" y="74" width="70" height="36" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="75" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像</text>
  <rect x="122" y="74" width="70" height="36" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="157" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像</text>
  <rect x="204" y="74" width="70" height="36" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="239" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">質問</text>
  <rect x="286" y="74" width="70" height="36" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="321" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">質問</text>
  <line x1="362" y1="68" x2="362" y2="116" stroke="#71717a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <rect x="368" y="74" width="70" height="36" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="403" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">答え</text>
  <rect x="450" y="74" width="70" height="36" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="485" y="97" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">答え</text>
  <text x="200" y="132" text-anchor="middle" font-size="11" fill="#71717a">条件（入力）</text>
  <text x="444" y="132" text-anchor="middle" font-size="11" fill="#71717a">生成（予測）</text>
  <line x1="540" y1="92" x2="690" y2="92" stroke="#71717a" stroke-width="2"/>
  <polygon points="682,87 692,92 682,97" fill="#71717a"/>
  <text x="616" y="83" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">次トークン予測</text>
  <text x="24" y="196" font-size="13" font-weight="700" fill="#18181b">生成：テキスト指示 → 画像トークンを生成</text>
  <rect x="40" y="210" width="70" height="36" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="75" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">指示</text>
  <rect x="122" y="210" width="70" height="36" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="157" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">指示</text>
  <rect x="204" y="210" width="70" height="36" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="239" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">指示</text>
  <line x1="280" y1="204" x2="280" y2="252" stroke="#71717a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <rect x="286" y="210" width="70" height="36" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="321" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像</text>
  <rect x="368" y="210" width="70" height="36" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="403" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像</text>
  <rect x="450" y="210" width="70" height="36" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="485" y="233" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">画像</text>
  <text x="157" y="268" text-anchor="middle" font-size="11" fill="#71717a">条件（入力）</text>
  <text x="403" y="268" text-anchor="middle" font-size="11" fill="#71717a">生成（予測）</text>
  <line x1="540" y1="228" x2="690" y2="228" stroke="#71717a" stroke-width="2"/>
  <polygon points="682,223 692,228 682,233" fill="#71717a"/>
  <text x="616" y="219" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">次トークン予測</text>
</svg><figcaption><b>理解も生成も「左から右へ次トークンを予測する」同じ系列モデリング</b>に統一。違いは条件と予測対象の種類だけで、因果マスクの上を実線（条件）から破線（予測）へと進む。</figcaption></figure>

学習目的は言語モデルと同じく、素朴な交差エントロピー（自己回帰の対数尤度最大化）である。

$$\mathcal{L} = -\sum_{i} \log p_\theta\!\left(x_i \mid x_{<i}\right)$$

理解・テキスト理解ではテキスト系列に、生成では画像系列に損失を計算する。タスクごとに損失の重みを変えるといった小細工はせず、設計を単純に保っている点も Janus らしい。

推論時はそのまま次トークン予測でサンプリングする。ただし画像生成では、テキスト条件あり・なしのロジットを混ぜる**分類器フリーガイダンス（CFG）**を使い、指示への忠実度を高める。

$$l_g = l_u + s\,(l_c - l_u)$$

ここで $l_c$ は条件付きロジット、$l_u$ は無条件ロジット、$s$ はガイダンス強度である。学習時に一定確率でテキスト条件をパッドトークンへ置き換えておくことで、無条件生成の能力をあらかじめ仕込んでおく。

### 学習は3段階

統合モデルゆえ、いきなり全部を同時に学習すると不安定になりやすい。Janus は段階を踏む。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="3段階の学習手順。第1段でアダプタと画像ヘッドのみ学習、第2段でLLMを解放して統合事前学習、第3段で指示チューニングを行う様子を、各モジュールが学習対象か凍結かで色分けした図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="130" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage I</text>
  <text x="360" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage II</text>
  <text x="590" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Stage III</text>
  <rect x="55" y="62" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="130" y="83" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">ヘッド</text>
  <rect x="55" y="112" width="150" height="40" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="130" y="137" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">LLM</text>
  <rect x="55" y="170" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="130" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">アダプタ</text>
  <rect x="55" y="220" width="150" height="34" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="130" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">エンコーダ</text>
  <text x="130" y="278" text-anchor="middle" font-size="11" fill="#18181b">アダプタと画像ヘッドのみ学習</text>
  <rect x="285" y="62" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="360" y="83" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">ヘッド</text>
  <rect x="285" y="112" width="150" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="360" y="137" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">LLM</text>
  <rect x="285" y="170" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="360" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">アダプタ</text>
  <rect x="285" y="220" width="150" height="34" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="360" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">エンコーダ</text>
  <text x="360" y="278" text-anchor="middle" font-size="11" fill="#18181b">LLMを解放し統合事前学習</text>
  <rect x="515" y="62" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="83" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">ヘッド</text>
  <rect x="515" y="112" width="150" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="137" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">LLM</text>
  <rect x="515" y="170" width="150" height="32" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="191" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">アダプタ</text>
  <rect x="515" y="220" width="150" height="34" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="590" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">エンコーダ</text>
  <text x="590" y="278" text-anchor="middle" font-size="11" fill="#18181b">指示チューニング（SFT）</text>
  <rect x="248" y="312" width="22" height="16" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="300" y="325" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">学習</text>
  <rect x="372" y="312" width="22" height="16" rx="3" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="424" y="325" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">凍結</text>
</svg><figcaption><b>第1段で橋渡し（アダプタ＋画像ヘッド）だけを学習し、第2段でLLMを解放して理解・生成を統合事前学習、第3段で指示チューニング</b>。緑が学習対象、灰が凍結（第3段でも生成エンコーダのみ凍結）。</figcaption></figure>

第1段は、視覚エンコーダと LLM を凍結したまま、理解アダプタ・生成アダプタ・画像ヘッドだけを学習し、視覚と言語を埋め込み空間でつなぐ「橋渡し」を作る。第2段で LLM を解放し、テキスト・画像理解・画像生成のデータを混ぜて統合事前学習を行う（まず ImageNet で基本的なピクセル依存性を掴ませ、その後に一般的なテキスト→画像生成へ進む）。第3段は指示チューニング（SFT）で、対話・指示追従の質を高める。このとき生成エンコーダ以外を更新する。**理解用と生成用に別モデルを作らず、最後まで一つのモデルとして仕上げる**のがポイントだ。

## まとめと、読解後に答えたい問い

- **一言で**: Janus は「理解には意味重視のエンコーダ（SigLIP）、生成には再構成重視の VQ トークナイザ」と**視覚エンコーディングを分離**し、その後段を**単一の自己回帰トランスフォーマに一本化**した統合モデルである。
- **なぜ効くか**: 1本のエンコーダで意味抽出と詳細再構成を兼ねると衝突する。経路を分けることで衝突を解消し、各タスクに最適な符号化を独立に選べる。これが理解性能の底上げにつながった。
- **どう統一されるか**: 入口は二経路でも、出口側は同じ次トークン予測。理解はテキストを、生成は画像トークンを予測するだけで、目的関数は素朴な交差エントロピーに収まる。
- **設計思想**: 単純・柔軟・拡張可能。アダプタは2層 MLP、特別なマスクも不要。理解側・生成側を独立に強化でき、新モダリティの追加も見据えられる。

読み終えたら、次の問いに自分の言葉で答えられるか確かめてほしい。

1. 「理解」と「生成」が画像に求める情報は、それぞれどう違うのか。なぜ1本のエンコーダで両立すると衝突するのかを、粒度の観点から説明できるか。
2. Janus が「分離」するのは入口のどこまでで、どこから先は「共有」なのか。図のどの境界が分離／一本化の境目かを指させるか。
3. 同じ自己回帰トランスフォーマが、理解（テキスト出力）と生成（画像トークン出力）をどう切り替えているのか。条件と予測対象という観点で言い換えられるか。
4. なぜ理解用ヘッドは LLM 内蔵、画像用ヘッドはランダム初期化なのか。両者で予測する対象の語彙が異なることと結びつけて説明できるか。
5. 3段階学習で、各段階が何を凍結し何を学習するのか。第1段でアダプタだけを先に学ぶことに、どんな安定化の意味があるか。
6. 「経路を分離した」効用は理解性能にとどまらない。拡張性の観点から、Janus の設計がなぜ将来のモダリティ追加に強いと言えるか。
