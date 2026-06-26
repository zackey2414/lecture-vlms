# InstructBLIP — instruction-aware Q-Former によるゼロショット指示追従

InstructBLIP（Towards General-purpose Vision-Language Models with Instruction Tuning, arXiv:2305.06500, Salesforce 2023）は、BLIP-2 を土台に「指示チューニング（instruction tuning）」を視覚言語へ持ち込んだモデルです。最大の貢献は、**指示文を LLM だけでなく Q-Former にも与える** という一点に集約されます。BLIP-2 を既習の読者にとって本稿の狙いは、この小さな配線変更が「タスクに応じて視覚特徴を作り変える」という能力をどう生み、なぜ未見タスクへのゼロショット汎化を押し上げるのかを腹落ちさせることです。

## 全体像（まず一枚で）

InstructBLIP の構成要素は BLIP-2 とほぼ同じです。凍結した画像エンコーダ（ViT-g/14）が画像を埋め込み、Q-Former が少数の学習可能なクエリでそこから視覚特徴を取り出し、それを soft prompt として凍結 LLM（FlanT5 もしくは Vicuna）へ渡します。学習対象は **Q-Former だけ**で、画像エンコーダと LLM は終始凍結されます。BLIP-2 と決定的に違うのは、**指示文が Q-Former にも分岐入力される** 点です。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="InstructBLIPの全体構成。凍結画像エンコーダからinstruction-aware Q-Formerを経て凍結LLMへ進み、指示文がQ-FormerとLLMの両方に入力される図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="300" y="30" width="330" height="42" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="465" y="49" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">指示文（例: この画像について詳しく説明して）</text>
<text x="465" y="65" text-anchor="middle" font-size="11" fill="#71717a">同じ指示が二手に分岐する</text>
<line x1="375" y1="72" x2="375" y2="151" stroke="#0e7490" stroke-width="2"/>
<polygon points="375,159 370,148 380,148" fill="#0e7490"/>
<line x1="600" y1="72" x2="600" y2="156" stroke="#0e7490" stroke-width="2"/>
<polygon points="600,164 595,153 605,153" fill="#0e7490"/>
<rect x="20" y="165" width="70" height="55" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="55" y="197" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">入力画像</text>
<rect x="110" y="160" width="110" height="65" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="165" y="190" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">画像エンコーダ</text>
<text x="165" y="208" text-anchor="middle" font-size="11" fill="#4338ca">（凍結）</text>
<rect x="300" y="155" width="150" height="80" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="375" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">instruction-aware</text>
<text x="375" y="202" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">Q-Former</text>
<text x="375" y="220" text-anchor="middle" font-size="11" fill="#0e7490">（学習対象）</text>
<rect x="470" y="170" width="60" height="45" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="500" y="197" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">全結合</text>
<rect x="565" y="160" width="125" height="65" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="627" y="190" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">LLM</text>
<text x="627" y="208" text-anchor="middle" font-size="11" fill="#4338ca">（凍結）</text>
<line x1="90" y1="192" x2="106" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="114,192 103,187 103,197" fill="#71717a"/>
<line x1="220" y1="192" x2="296" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="304,192 293,187 293,197" fill="#71717a"/>
<line x1="450" y1="192" x2="466" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="474,192 463,187 463,197" fill="#71717a"/>
<line x1="530" y1="192" x2="561" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="569,192 558,187 558,197" fill="#71717a"/>
<rect x="565" y="270" width="125" height="40" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="627" y="294" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">応答テキスト</text>
<line x1="627" y1="225" x2="627" y2="261" stroke="#71717a" stroke-width="2"/>
<polygon points="627,269 622,258 632,258" fill="#71717a"/>
</svg><figcaption>BLIP-2 と同じ「凍結エンコーダ＋Q-Former＋凍結 LLM」だが、<b>指示文が Q-Former にも分岐入力</b>される。学習されるのは Q-Former のみで、視覚と言語の巨大な重みは凍結したまま指示追従を獲得する。</figcaption></figure>

つまり InstructBLIP は、巨大モデルを再学習せず、橋渡し役の Q-Former だけを指示で条件づけることで汎用性を得ます。学習コストの軽さ（画像エンコーダ・LLM 凍結）と汎化性能を両立させる設計思想は BLIP-2 から受け継いだもので、本論文はそこへ「指示認識」という一手を加えたものと読むのが素直です。

## instruction-aware Q-Former

BLIP-2 の Q-Former は、画像から特徴を取り出すときに**タスクを一切見ません**（instruction-agnostic）。同じ画像なら、説明生成でも空間推論でも OCR でも、Q-Former が吐き出す視覚特徴は常に同じです。静的な視覚表現を作り、後段の LLM がそれを文脈に応じて解釈する、という分業でした。

InstructBLIP はここに介入します。Q-Former の入力に、学習可能なクエリ埋め込みに加えて **指示テキストのトークン**を並べ、両者を Q-Former 内部の **自己注意（self-attention）層**で相互作用させます。クエリは画像埋め込みとはクロスアテンションで、指示とは自己注意で交わるため、「いま何を問われているか」を踏まえて画像のどこに注意を割くかを変えられます。結果として、抽出される視覚特徴そのものが**指示に応じて作り変わる**（instruction-aware visual feature extraction）。同じ画像でも、空間関係を問われればレイアウトに、文字を問われれば文字領域に寄った特徴が LLM へ渡る、という具合です。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="指示なしのBLIP-2のQ-Formerと指示ありのInstructBLIPのQ-Formerを左右で対比した図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="15" y="48" width="330" height="296" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="2"/>
<text x="180" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">BLIP-2: 指示なし Q-Former</text>
<rect x="40" y="78" width="120" height="40" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="100" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">画像埋め込み</text>
<rect x="190" y="78" width="120" height="40" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="250" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">クエリ</text>
<rect x="60" y="158" width="230" height="70" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="175" y="188" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Q-Former</text>
<text x="175" y="208" text-anchor="middle" font-size="11" fill="#71717a">自己注意＝クエリのみ</text>
<rect x="80" y="268" width="190" height="44" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="175" y="288" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">静的な視覚特徴</text>
<text x="175" y="304" text-anchor="middle" font-size="11" fill="#71717a">全タスクで同一</text>
<line x1="100" y1="118" x2="124" y2="156" stroke="#71717a" stroke-width="2"/>
<polygon points="128,162 117,158 124,150" fill="#71717a"/>
<line x1="250" y1="118" x2="226" y2="156" stroke="#71717a" stroke-width="2"/>
<polygon points="222,162 226,150 233,158" fill="#71717a"/>
<line x1="175" y1="228" x2="175" y2="264" stroke="#71717a" stroke-width="2"/>
<polygon points="175,272 170,261 180,261" fill="#71717a"/>
<rect x="375" y="48" width="330" height="296" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="2"/>
<text x="540" y="38" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">InstructBLIP: 指示あり Q-Former</text>
<rect x="393" y="78" width="92" height="40" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="439" y="103" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">画像埋め込み</text>
<rect x="495" y="78" width="80" height="40" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="535" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">クエリ</text>
<rect x="585" y="78" width="100" height="40" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="635" y="103" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">指示文</text>
<rect x="415" y="158" width="270" height="70" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="550" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">Q-Former</text>
<text x="550" y="206" text-anchor="middle" font-size="11" fill="#0e7490">自己注意でクエリ × 指示が相互作用</text>
<rect x="440" y="268" width="220" height="44" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
<text x="550" y="288" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">指示に応じた視覚特徴</text>
<text x="550" y="304" text-anchor="middle" font-size="11" fill="#0e7490">タスク適応的</text>
<line x1="439" y1="118" x2="468" y2="156" stroke="#71717a" stroke-width="2"/>
<polygon points="472,162 461,158 468,150" fill="#71717a"/>
<line x1="535" y1="118" x2="540" y2="156" stroke="#71717a" stroke-width="2"/>
<polygon points="541,162 535,151 546,153" fill="#71717a"/>
<line x1="635" y1="118" x2="615" y2="156" stroke="#0e7490" stroke-width="2"/>
<polygon points="611,162 615,150 622,158" fill="#0e7490"/>
<line x1="550" y1="228" x2="550" y2="264" stroke="#71717a" stroke-width="2"/>
<polygon points="550,272 545,261 555,261" fill="#71717a"/>
</svg><figcaption>BLIP-2（左）は指示を見ずに<b>静的な視覚特徴</b>を作る。InstructBLIP（右）は<b>指示文を Q-Former の自己注意にも入れ</b>、クエリが指示と交わることで<b>指示に応じた視覚特徴</b>を抽出する。配線は小さく、効果は大きい。</figcaption></figure>

設計上のうれしさは、Q-Former のクロスアテンション（画像）と自己注意（クエリ＋指示）という既存の構造をそのまま使い、入力にテキストを足すだけで「タスク適応的な特徴抽出」を実現している点です。論文のアブレーション（Table 2）では、この instruction-aware 化を外すと held-in・held-out の双方で性能が明確に落ち、とりわけ **空間推論（ScienceQA など）や時間推論（iVQA など）** で低下が大きいと報告されています。指示が「画像のどこを見るべきか」を誘導する役割を、定量的にも裏づけた結果と読めます。

## 指示チューニングのデータ設計

汎用性は仕掛けだけでなくデータからも来ます。InstructBLIP は公開済みの視覚言語データセットを広く集め、本文によれば **26 データセット・11 タスクカテゴリ** を統一的な指示フォーマットへ変換します。各タスクには 10〜15 種の自然言語テンプレートを人手で用意し、データセットの素の入出力を「指示＋応答」の形に書き換えます（短答系には *short* / *briefly* を混ぜて常に短い出力へ過適合しないようにする、シーンテキスト系には OCR トークンを補助情報として添える、といった工夫も入ります）。

評価設計の肝は **held-in / held-out の分離**です。26 を **13 の held-in（指示チューニングに使用）** と **13 の held-out（ゼロショット評価に使用）** に分け、さらに **4 つのタスクカテゴリを丸ごと未学習**として残し、タスクレベルのゼロショットも測れるようにします。データ汚染を避けるため、評価データが別データセットの学習側に混入しないよう選定されています。

もう一つの工夫が **balanced sampling（均衡サンプリング）** です。データセットごとに規模が桁違いに異なるため、一様に混ぜると小さなデータに過適合し大きなデータに過少適合します。そこで、サイズ $S_d$ のデータから一サンプルが選ばれる確率を **規模の平方根に比例**させます。

$$p_d = \dfrac{\sqrt{S_d}}{\sum_i \sqrt{S_i}}$$

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="26データセットをheld-inとheld-outに分け、held-in側にbalanced samplingを適用するデータ設計の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="210" y="22" width="300" height="48" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="360" y="44" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">公開 VL データセット 26 種 / 11 タスクカテゴリ</text>
<text x="360" y="61" text-anchor="middle" font-size="11" fill="#6366f1">自然言語の指示テンプレートに変換</text>
<line x1="320" y1="74" x2="210" y2="110" stroke="#71717a" stroke-width="2"/>
<polygon points="203,114 211,105 215,116" fill="#71717a"/>
<line x1="400" y1="74" x2="510" y2="110" stroke="#71717a" stroke-width="2"/>
<polygon points="517,114 505,116 509,105" fill="#71717a"/>
<rect x="40" y="115" width="300" height="64" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>
<text x="190" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#854d0e">held-in（13 データセット）</text>
<text x="190" y="160" text-anchor="middle" font-size="12" fill="#854d0e">→ 指示チューニング学習</text>
<rect x="380" y="115" width="300" height="64" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="530" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">held-out（13 データセット）</text>
<text x="530" y="160" text-anchor="middle" font-size="12" fill="#71717a">→ ゼロショット評価</text>
<line x1="190" y1="179" x2="190" y2="201" stroke="#71717a" stroke-width="2"/>
<polygon points="190,209 185,198 195,198" fill="#71717a"/>
<line x1="530" y1="179" x2="530" y2="201" stroke="#71717a" stroke-width="2"/>
<polygon points="530,209 525,198 535,198" fill="#71717a"/>
<rect x="40" y="211" width="300" height="100" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="190" y="235" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">balanced sampling</text>
<text x="190" y="258" text-anchor="middle" font-size="12" fill="#15803d">抽出確率 ∝ √(データ規模)</text>
<text x="190" y="280" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">p_d = √S_d / Σ √S_i</text>
<text x="190" y="300" text-anchor="middle" font-size="11" fill="#15803d">小データ過学習・大データ過少学習を緩和</text>
<rect x="380" y="211" width="300" height="100" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
<text x="530" y="237" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">学習に一切現れないデータ・タスクで評価</text>
<text x="530" y="262" text-anchor="middle" font-size="12" fill="#0e7490">4 タスクカテゴリは丸ごと未学習</text>
<text x="530" y="287" text-anchor="middle" font-size="12" fill="#0e7490">指示追従のゼロショット</text>
<text x="530" y="304" text-anchor="middle" font-size="12" fill="#0e7490">汎化能力を測定</text>
</svg><figcaption>26 データセットを <b>held-in（学習）/ held-out（評価）</b>に分け、4 タスクカテゴリは丸ごと未学習に残す。<b>balanced sampling</b> は抽出確率を規模の平方根に比例させ、データ規模の偏りによる過学習・過少学習を抑える。</figcaption></figure>

平方根則に加え、最適化を整えるために一部データセットの重みを手で微調整します（多肢選択中心の A-OKVQA は重みを下げ、自由記述の OKVQA は上げる、など）。Table 2 のアブレーションでは、balanced sampling を外すと学習の進み方がデータセット間でバラつき、held-in・held-out のいずれも不安定化・低下します。複数タスクの「歩調を揃える」ことが汎化に効く、という主張です。

## ゼロショット汎化

これらを合わせると、InstructBLIP は **held-out の 13 ベンチマークでゼロショット SoTA 級**に達し、原型である BLIP-2 を有意に上回り、はるかに大きな Flamingo 系も超えます（本文 Table 1）。注目すべきは、学習時に時系列の動画データを見ていないにもかかわらず、動画 QA のような未見タスクカテゴリでも強い、という点です。フレームを個別に画像エンコーダ＋Q-Former へ通し、得た特徴を連結して LLM に渡すだけで、指示認識が新タスクへの橋渡しになっています。

なぜ指示認識が汎化を上げるのか。直観的には二段構えで効きます。第一に、視覚側が指示で条件づくと、**タスクに不要な視覚情報を落とし必要な情報を残した**特徴が LLM へ届くため、後段の言語推論が楽になります。第二に、多様な指示テンプレートで学習することで、モデルが「指示の表面形ではなく意図」に反応するようになり、未見の言い回し・未見タスクへ転移しやすくなります。

この主張を切り分けたのが本文 Figure 4 の「指示チューニング vs マルチタスク学習」です。指示なしのマルチタスク学習でも **held-in** は同等に解けますが、**held-out** ではマルチタスクは BLIP-2 並みに留まり、指示チューニングだけが大きく伸びます。つまり held-in の当てはめは「そのデータで訓練したか」で決まり、ゼロショット汎化を分けるのは「指示で訓練したか」だ、というのが論文の結論です。指示は単なる入力整形ではなく、未見への一般化の鍵だということです。

## まとめと、読解後に答えたい問い

InstructBLIP の核心は、BLIP-2 の凍結エンコーダ・凍結 LLM・学習対象 Q-Former という枠を保ったまま、**指示文を Q-Former にも与える**一手で「指示に応じた視覚特徴抽出」を獲得した点にあります。そこへ、26 データセットを held-in/held-out に分けた指示チューニングと、規模の平方根に比例させる balanced sampling を組み合わせ、**held-out のゼロショットで SoTA 級**を実現しました。指示は入力整形ではなく汎化の鍵であり、その効きどころは空間・時間推論のように「どこを見るか」が問われるタスクで顕著でした。

読解後に、次の問いへ自分の言葉で答えられるか確かめてください。

- 指示文を **LLM だけ**に与える BLIP-2 と、**Q-Former にも**与える InstructBLIP で、抽出される視覚特徴はどう変わるか。その差はアブレーションのどのタスクで最も表れたか。
- balanced sampling の確率はなぜ「規模そのもの」ではなく「規模の平方根」に比例させるのか。一様サンプリングや規模比例だと何が起きるか。
- held-in は指示チューニングでもマルチタスク学習でも同等に解けるのに、held-out では差がつくのはなぜか（Figure 4 の主張を自分で再構成できるか）。
- 動画 QA のように学習で見ていないタスクへ汎化できたのは、アーキテクチャ（指示認識）とデータ設計（テンプレートの多様性）のどちらがどう寄与した結果と説明できるか。
