# CogVLM — Visual Expert for Pretrained Language Models

LLaVA 系では、凍結した大規模言語モデル（LLM）の入力空間へ画像特徴を写すだけの **projector（shallow alignment）** で視覚と言語をつないだ。学習は速く収束する一方、視覚と言語が「浅く」しか混ざらないため、表現力に限界がある。CogVLM はこの前提を崩し、トランスフォーマの**各層**に**画像トークン専用の重み（visual expert）**を差し込むことで、**深い融合（deep fusion）**を実現する。しかも視覚エンコーダ（EVA-CLIP）由来の画像トークンと、もとの LLM が処理するテキストトークンとで、層の内部で重みを使い分けるため、**もとの言語能力（NLP）を犠牲にしない**。本章では、その仕組みと狙いを読み解く。

CogVLM-17B はオフザシェルフの言語モデル（Vicuna1.5-7B）と視覚エンコーダから組み立てられる。鍵は、追加学習する重みを「入口の projector 一枚」ではなく「全層に薄く広げた視覚専用パス」として持たせた点にある。


## 全体像（まず一枚で）

入力は2系統に分かれる。画像は **ViT エンコーダ（EVA-CLIP）** でパッチ特徴に変換され、**MLP アダプタ（2層, SwiGLU）** でテキスト特徴と同じ次元・空間へ写されて**画像トークン**になる。テキストは通常どおり**単語埋め込み**で**テキストトークン**になる。両者を1本の系列に **concat** し、**visual expert を内蔵したトランスフォーマ N 層**へ流して、次トークンを予測する。

ここまでは LLaVA とよく似た外形だ。違いは「トランスフォーマ N 層」の**中身**にある。LLaVA では projector が画像をならした後の LLM 本体は**触らない**が、CogVLM は各層の内部に画像トークン専用の経路を増設している。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="CogVLMの全体構成。画像はViTエンコーダとMLPアダプタを通って画像トークンになり、テキストは単語埋め込みでテキストトークンになる。両者を結合して、visual expertを内蔵したトランスフォーマN層に入力し、次トークンを予測する。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="20" y="58" width="96" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="68" y="86" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">パッチ画像</text>
  <rect x="146" y="58" width="112" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="202" y="80" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">ViT エンコーダ</text>
  <text x="202" y="96" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0e7490">EVA-CLIP</text>
  <rect x="288" y="58" width="112" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="344" y="80" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">MLP アダプタ</text>
  <text x="344" y="96" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0e7490">SwiGLU</text>
  <line x1="116" y1="81" x2="142" y2="81" stroke="#71717a" stroke-width="2"/>
  <polygon points="142,77 150,81 142,85" fill="#71717a"/>
  <line x1="258" y1="81" x2="284" y2="81" stroke="#71717a" stroke-width="2"/>
  <polygon points="284,77 292,81 284,85" fill="#71717a"/>

  <rect x="20" y="190" width="96" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="68" y="218" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力テキスト</text>
  <rect x="146" y="190" width="112" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="202" y="218" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">単語埋め込み</text>
  <line x1="116" y1="213" x2="142" y2="213" stroke="#71717a" stroke-width="2"/>
  <polygon points="142,209 150,213 142,217" fill="#71717a"/>

  <rect x="430" y="118" width="108" height="76" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="484" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">concat</text>
  <text x="484" y="170" text-anchor="middle" font-size="11.5" font-weight="700" fill="#71717a">画像＋テキスト列</text>
  <line x1="400" y1="81" x2="426" y2="128" stroke="#0e7490" stroke-width="2"/>
  <polygon points="421,122 428,132 414,131" fill="#0e7490"/>
  <line x1="258" y1="213" x2="426" y2="184" stroke="#4338ca" stroke-width="2"/>
  <polygon points="420,180 428,186 419,189" fill="#4338ca"/>

  <rect x="568" y="106" width="132" height="100" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="634" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Transformer</text>
  <text x="634" y="166" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">× N 層</text>
  <text x="634" y="186" text-anchor="middle" font-size="11.5" font-weight="700" fill="#4338ca">visual expert 内蔵</text>
  <line x1="538" y1="156" x2="564" y2="156" stroke="#71717a" stroke-width="2"/>
  <polygon points="564,152 572,156 564,160" fill="#71717a"/>

  <rect x="568" y="22" width="132" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="634" y="50" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">次トークン予測</text>
  <line x1="634" y1="106" x2="634" y2="72" stroke="#71717a" stroke-width="2"/>
  <polygon points="630,72 634,64 638,72" fill="#71717a"/>
</svg>
<figcaption>画像は ViT と MLP アダプタを経て画像トークンに、テキストは単語埋め込みでテキストトークンになる。両者を1本の系列にして N 層へ。<b>要点</b>: 外形は projector 型と似るが、N 層の内部に視覚専用パスが入っている点が決定的に違う。</figcaption>
</figure>


## shallow alignment（projector）の限界

projector 型（線形層や Q-Former）の発想はシンプルだ。凍結した LLM の**入力空間**に画像特徴を写し、あとはテキストと同じように流すだけ。学習はすぐ収束する。しかし、ここには構造的なミスマッチが潜む。

- **LLM の重みはテキスト処理用に最適化されている。** 各層の重みはテキストトークンの分布を前提に学習されており、画像特徴には**直接対応する単語埋め込みが存在しない**。
- **入口で揃えても、深い層でズレる。** projector が入口の分布をうまく合わせても、画像特徴が多層の変換を経るうちに、**深い層が期待する入力分布から逸脱（deviate）**していく。浅い整合は文字どおり「浅い」。
- **タスク特異な情報を表層的にしか符号化できない。** キャプションの文体や長さといった、タスクに固有の制御情報は、固定された写像だけでは視覚特徴に**表面的にしか**埋め込めない。

「では LLM ごと学習すればよいのでは」という反論はもっともだ。実際 PaLI や Qwen-VL は事前学習・SFT の段で LLM を直接訓練する。だがこれには別の代償がある。LLM は大規模な**テキストのみ**のコーパスで事前学習されており、画像テキストペア（LAION や COYO）とは分布が大きく異なる。画像テキストで LLM を訓練し直すと、**もとの得意分野が劣化する（catastrophic forgetting）**。論文の図では、言語側を直接学習すると純テキスト指標が学習ステップとともに急落していく様子が示される（具体値は割愛、傾向として大幅に低下）。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="shallow alignmentの限界。projectorは画像特徴を凍結言語モデルの入力空間に写すだけで、テキスト分布で学習済みの深い層に到達すると分布がズレる。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="24" y="150" width="104" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="76" y="183" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">画像特徴</text>
  <rect x="158" y="150" width="150" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="233" y="174" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">projector</text>
  <text x="233" y="192" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">線形 / Q-Former</text>
  <line x1="128" y1="178" x2="154" y2="178" stroke="#71717a" stroke-width="2"/>
  <polygon points="154,174 162,178 154,182" fill="#71717a"/>

  <rect x="372" y="276" width="232" height="46" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="488" y="304" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語モデル層 1（凍結）</text>
  <rect x="372" y="216" width="232" height="46" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="488" y="244" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語モデル層 2（凍結）</text>
  <rect x="372" y="156" width="232" height="46" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="488" y="184" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語モデル層 3（凍結）</text>
  <rect x="372" y="96" width="232" height="46" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="488" y="124" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語モデル層 4（凍結）</text>
  <line x1="308" y1="178" x2="368" y2="222" stroke="#0e7490" stroke-width="2"/>
  <polygon points="362,216 370,226 356,225" fill="#0e7490"/>

  <rect x="624" y="96" width="80" height="106" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"/>
  <text x="664" y="138" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">分布の</text>
  <text x="664" y="156" text-anchor="middle" font-size="12.5" font-weight="700" fill="#dc2626">ミスマッチ</text>
  <line x1="624" y1="150" x2="606" y2="170" stroke="#dc2626" stroke-width="2"/>
  <polygon points="612,166 604,172 614,174" fill="#dc2626"/>

  <text x="76" y="60" font-size="12.5" font-weight="700" fill="#71717a">入口で写すだけ＝浅い接続。深い層はテキスト分布のまま学習されており、</text>
  <text x="76" y="78" font-size="12.5" font-weight="700" fill="#71717a">写しただけの視覚特徴が深部に届くとズレが顕在化する。</text>
</svg>
<figcaption>projector は凍結言語モデルの入口に画像を写すだけ。深い層はテキスト分布のまま学習されているため、視覚特徴が深部へ進むほど期待分布から外れていく。<b>要点</b>: 浅い接続は言語能力を守れるが融合が浅い、直接学習は融合が深いが言語能力を壊す。両立が課題。</figcaption>
</figure>

つまりここには明確な**トレードオフ**がある。「浅く接続して言語能力を守る」か、「深く学習して融合を得るが言語能力を失う」か。CogVLM はこの二者択一を、**層内でトークンの種別ごとに重みを分ける**ことで回避する。


## visual expert module（deep fusion）

中核が **visual expert module** だ。トランスフォーマの**各層**に、**画像トークン専用の QKV 射影と FFN（＝視覚エキスパート）**を増設する。その形（次元）はもとの言語モデルの QKV・FFN と**同一**で、**言語モデルの重みで初期化**される（ランダム初期化より良い、という結果が得られている）。

肝は「**層の内部で、トークンの種別ごとに別の重みを当てる**」ことだ。系列を画像隠れ状態 $X_I$ とテキスト隠れ状態 $X_T$ に分け、画像には視覚エキスパートの重み $W_I$、テキストにはもとの言語重み $W_T$ を適用してから結合する。注意機構は次のように書ける。

$$\mathrm{Attention}(X, W_I, W_T) = \mathrm{softmax}\!\left(\frac{\mathrm{Tril}(QK^\top)}{\sqrt{D}}\right) V$$

$$Q = \mathrm{concat}(X_I W_I^Q,\; X_T W_T^Q),\quad K = \mathrm{concat}(X_I W_I^K,\; X_T W_T^K),\quad V = \mathrm{concat}(X_I W_I^V,\; X_T W_T^V)$$

ここで $W_I$ は視覚エキスパート、$W_T$ はもとの言語モデルの QKV 行列、$\mathrm{Tril}(\cdot)$ は下三角（因果）マスクを表す。Q・K・V を作る段で種別ごとに重みを分け、結合してから**共通の注意**を取るので、画像とテキストは**同じ注意空間で相互作用**できる。FFN も同様に種別ごとに分かれる。

$$\mathrm{FFN}(X) = \mathrm{concat}\big(\mathrm{FFN}_I(X_I),\; \mathrm{FFN}_T(X_T)\big)$$

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="visual expert moduleの1層の内部。画像トークンは視覚QKV射影W_Iと視覚FFNを通り、テキストトークンは言語QKV射影W_Tと言語FFNを通る。注意機構はQKVを結合して共通に計算される。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="74" y="18" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="114" y="18" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="154" y="18" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="298" y="18" width="34" height="32" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="338" y="18" width="34" height="32" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="131" y="64" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0e7490">画像トークン</text>
  <text x="335" y="64" text-anchor="middle" font-size="11.5" font-weight="700" fill="#4338ca">テキストトークン</text>

  <rect x="70" y="84" width="160" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="150" y="112" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">視覚 QKV 射影 W_I</text>
  <rect x="280" y="84" width="160" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="112" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語 QKV 射影 W_T</text>
  <line x1="131" y1="50" x2="140" y2="80" stroke="#0e7490" stroke-width="2"/>
  <polygon points="136,75 142,84 144,73" fill="#0e7490"/>
  <line x1="335" y1="50" x2="352" y2="80" stroke="#4338ca" stroke-width="2"/>
  <polygon points="348,75 354,84 356,73" fill="#4338ca"/>

  <rect x="110" y="152" width="290" height="42" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="255" y="178" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">Multi-Head Attention（因果マスク, Q K V を結合）</text>
  <line x1="150" y1="130" x2="170" y2="148" stroke="#71717a" stroke-width="2"/>
  <polygon points="166,143 172,151 159,149" fill="#71717a"/>
  <line x1="360" y1="130" x2="338" y2="148" stroke="#71717a" stroke-width="2"/>
  <polygon points="350,149 337,151 343,143" fill="#71717a"/>

  <rect x="70" y="226" width="160" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="150" y="254" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">視覚 FFN（FFN_I）</text>
  <rect x="280" y="226" width="160" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="254" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">言語 FFN（FFN_T）</text>
  <line x1="220" y1="194" x2="160" y2="222" stroke="#71717a" stroke-width="2"/>
  <polygon points="166,217 156,224 169,225" fill="#71717a"/>
  <line x1="290" y1="194" x2="350" y2="222" stroke="#71717a" stroke-width="2"/>
  <polygon points="344,217 354,224 341,225" fill="#71717a"/>

  <rect x="74" y="300" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="114" y="300" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="154" y="300" width="34" height="32" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="298" y="300" width="34" height="32" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="338" y="300" width="34" height="32" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <line x1="150" y1="272" x2="140" y2="298" stroke="#0e7490" stroke-width="2"/>
  <polygon points="138,290 139,300 145,291" fill="#0e7490"/>
  <line x1="360" y1="272" x2="350" y2="298" stroke="#4338ca" stroke-width="2"/>
  <polygon points="348,290 349,300 355,291" fill="#4338ca"/>

  <rect x="470" y="86" width="232" height="160" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="586" y="112" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">層内のルーティング</text>
  <rect x="486" y="124" width="20" height="20" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="516" y="139" font-size="11.5" font-weight="700" fill="#0e7490">視覚エキスパート</text>
  <text x="516" y="156" font-size="11.5" font-weight="700" fill="#71717a">W_I・FFN_I（追加・学習）</text>
  <rect x="486" y="172" width="20" height="20" rx="4" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="516" y="187" font-size="11.5" font-weight="700" fill="#4338ca">言語モデル</text>
  <text x="516" y="204" font-size="11.5" font-weight="700" fill="#71717a">W_T・FFN_T（元の重み）</text>
  <text x="486" y="228" font-size="11.5" font-weight="700" fill="#18181b">パラメータ約2倍 / FLOPs ほぼ不変</text>
</svg>
<figcaption>1層の内部。画像トークンは視覚 QKV と視覚 FFN を、テキストトークンは言語 QKV と言語 FFN を通り、注意は Q K V を結合して共通に計算する。<b>要点</b>: 各トークンは片方の経路しか通らないため、パラメータは約2倍でも計算量（FLOPs）はほぼ増えない。</figcaption>
</figure>

補足を3点。第一に、注意マスクは画像トークンを含め**因果（causal）マスク**を採用する。双方向の full mask の方が情報は多く使えそうだが、実験では因果マスクの方が良い（LLM 本来の構造に合うため、と解釈される）。第二に、**位置埋め込み**は RoPE 上で**すべての画像トークンが単一の position id を共有**する。画像は数百〜数千トークンになり得るうえ、位置情報は ViT 内ですでに符号化済みだからだ。第三に、視覚エキスパートはパラメータを約2倍にするが、**各トークンは片方の経路しか通らない**ため、推論あたりの計算量（FLOPs）はほぼ変わらない。


## 言語能力を保ったまま深く融合

なぜ「深く融合」しても言語能力が落ちないのか。答えは経路設計にある。**もとの言語モデルの重み $W_T$・$\mathrm{FFN}_T$ は固定**され、追加した視覚エキスパートは**画像トークンにしか作用しない**。したがって、**画像を含まない入力（テキストのみ）では、振る舞いはもとの言語モデルと完全に同一**になる。NLP のベンチマーク上で性能を落とす理由がそもそも生じない。

この設計は LoRA・P-Tuning との対比で理解すると腑に落ちる。projector による浅い接続は、入力に「視覚という接頭辞（prefix）」を差し込む P-Tuning に似ている。一方 visual expert は、各層の重みを画像トークン向けに**低ランクではなく専用パスとして**差し替える点で LoRA に近く、各層で適応するぶん**より安定**に働く。視覚エキスパートが学習を通じて、各注意ヘッドが捉える意味側面に画像特徴を整列させることで、入口だけでなく**全層で**視覚と言語が混ざり合う。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="言語能力の保持。テキストのみの入力ではW_TとFFN_Tだけを通り元の言語モデルと同一の出力になる。画像とテキストの入力では画像にW_I・FFN_I、テキストにW_T・FFN_Tが適用され、言語重みを保ったまま深い融合が起きる。" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="28" y="40" font-size="13" font-weight="700" fill="#4338ca">ケース1: テキストのみの入力</text>
  <rect x="28" y="56" width="120" height="50" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="88" y="86" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">テキストのみ</text>
  <rect x="226" y="56" width="190" height="50" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="321" y="80" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">各層は W_T・FFN_T のみ通る</text>
  <text x="321" y="97" text-anchor="middle" font-size="11" font-weight="700" fill="#71717a">視覚エキスパートは作用しない</text>
  <rect x="494" y="56" width="200" height="50" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="594" y="80" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">元の言語モデルと同一の出力</text>
  <text x="594" y="97" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">NLP 性能を維持</text>
  <line x1="148" y1="81" x2="222" y2="81" stroke="#71717a" stroke-width="2"/>
  <polygon points="222,77 230,81 222,85" fill="#71717a"/>
  <line x1="416" y1="81" x2="490" y2="81" stroke="#71717a" stroke-width="2"/>
  <polygon points="490,77 498,81 490,85" fill="#71717a"/>

  <line x1="28" y1="160" x2="694" y2="160" stroke="#e4e4e7" stroke-width="2"/>

  <text x="28" y="206" font-size="13" font-weight="700" fill="#0e7490">ケース2: 画像＋テキストの入力</text>
  <rect x="28" y="222" width="120" height="50" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="88" y="252" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">画像＋テキスト</text>
  <rect x="226" y="222" width="190" height="50" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="321" y="246" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0e7490">画像 = W_I・FFN_I</text>
  <text x="321" y="263" text-anchor="middle" font-size="11.5" font-weight="700" fill="#4338ca">テキスト = W_T・FFN_T</text>
  <rect x="494" y="222" width="200" height="50" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="594" y="246" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">深い融合（deep fusion）</text>
  <text x="594" y="263" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">言語重みは保たれたまま</text>
  <line x1="148" y1="247" x2="222" y2="247" stroke="#71717a" stroke-width="2"/>
  <polygon points="222,243 230,247 222,251" fill="#71717a"/>
  <line x1="416" y1="247" x2="490" y2="247" stroke="#71717a" stroke-width="2"/>
  <polygon points="490,243 498,247 490,251" fill="#71717a"/>

  <text x="28" y="320" font-size="12" font-weight="700" fill="#71717a">言語側の重みを一切動かさないので、視覚を足しても NLP が劣化しない。LoRA に似て各層で適応するため安定。</text>
</svg>
<figcaption>テキストのみなら言語重みだけを通り元のモデルと完全一致、画像があれば視覚エキスパートが画像トークンにだけ作用する。<b>要点</b>: 言語側の重みを固定したまま全層で融合するため、深い融合と言語能力の保持が両立する。</figcaption>
</figure>

学習面でも、この経路設計は素直に効く。視覚エキスパートを**言語モデルの重みで初期化**すると、ランダム初期化より一貫して良い。テキストで事前学習されたトランスフォーマの重みは、視覚トークンを処理する出発点としても有効だ、という示唆である。視覚エンコーダ（EVA-CLIP）は事前学習段ではほぼ固定的に扱い、後段でわずかに（低い学習率で）適応させる程度に留めることで、安定性を保つ。


## まとめと、読解後に答えたい問い

- CogVLM は、projector による **shallow alignment** に代えて、各層に **visual expert module** を増設して **deep fusion** を実現する。
- 仕組みの核は、**層内でトークン種別ごとに重みを分ける**こと。画像トークンには視覚専用の QKV 射影と FFN（$W_I$, $\mathrm{FFN}_I$）、テキストトークンには元の言語重み（$W_T$, $\mathrm{FFN}_T$）を当て、注意は Q・K・V を結合して共通に計算する。
- 言語側の重みは固定されるため、**画像のない入力では元の言語モデルと同一**。視覚を足しても NLP 能力を犠牲にしない。
- 視覚エンコーダは EVA-CLIP、画像特徴は MLP アダプタ（SwiGLU）でテキスト空間へ写す。注意は因果マスク、画像トークンは単一 position id を共有。
- パラメータは約2倍だが各トークンは片方の経路しか通らないため **FLOPs はほぼ不変**。視覚エキスパートは言語モデル重みで初期化すると良い。

読み終えたら、次の問いに自分の言葉で答えられるか確かめてほしい。

1. shallow alignment の「浅さ」は具体的にどの段階で問題になるのか。入口の整合が取れていても深い層でズレるのはなぜか。
2. LLM を直接学習する方式（PaLI / Qwen-VL 型）の利点と、catastrophic forgetting という代償を説明できるか。
3. visual expert が、画像とテキストで**注意を共有しつつ**重みを分けられるのはなぜか。式の Q・K・V の concat がそれをどう担保しているか。
4. 画像トークンのない入力で、なぜ振る舞いが元の言語モデルと完全に一致すると言えるのか。
5. パラメータが約2倍でも FLOPs がほぼ増えないのはなぜか。各トークンが通る経路の本数で説明できるか。
6. projector を P-Tuning、visual expert を LoRA に例える比喩は、どこまで的確でどこから崩れるか。
7. 因果マスクの採用と、全画像トークンが単一 position id を共有する設計には、それぞれどんな狙いがあるか。
