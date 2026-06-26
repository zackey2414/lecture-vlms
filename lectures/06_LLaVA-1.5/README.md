# LLaVA-1.5 — Improved Baselines（MLP projector・336px・標準ベースライン）

LLaVA-1.5 は、初代 LLaVA の枠組み（視覚エンコーダ → 軽量コネクタ → LLM）をほぼそのまま保ったまま、**ごく少数の設計変更**だけで多数のベンチマークを SoTA 級へ押し上げた論文です（*Improved Baselines with Visual Instruction Tuning*, arXiv:2310.03744）。変更は本質的に 3 つ —— コネクタを **線形1層から2層MLP** へ、視覚エンコーダを **CLIP ViT-L/14@336px** へ、そして **学術VQA系データ＋応答フォーマット指示プロンプト** の追加。派手な再サンプラ（Q-Former 等）も巨大事前学習も使わず、**公開データのみ**で短時間に再現できる点が、本モデルを「みんなが乗っかる標準ベースライン」にしました。本ページでは、CLIP と深層学習を既知とする読者が「LLaVA から何が変わって、なぜそれが効いたか」を腹落ちできるよう順に解きほぐします。

## 全体像と、LLaVA からの差分

LLaVA 系の骨格は一貫して「**画像を CLIP 系の視覚エンコーダで特徴化 → コネクタで言語埋め込み空間へ写像 → 既存 LLM が視覚トークンとユーザ指示を一緒に読んで自由文を生成**」というシンプルな構成です。LLaVA-1.5 はこの骨格を一切壊しません。壊さないまま、コネクタ・解像度・学習データという「LLaVA の枠組みにそのまま足せる3つのつまみ」を回しただけ、というのが論文の主張の核です。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="LLaVA-1.5の全体アーキテクチャ。入力画像を視覚エンコーダで符号化し、2層MLPコネクタを介して言語モデルに渡し、ユーザ指示とともに応答を生成する流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="20" y="140" width="96" height="70" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="68" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">入力画像</text>
<line x1="116" y1="175" x2="135" y2="175" stroke="#71717a" stroke-width="2"/>
<polygon points="144,175 135,170 135,180" fill="#71717a"/>
<rect x="148" y="130" width="156" height="92" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="226" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">視覚エンコーダ</text>
<text x="226" y="192" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">CLIP ViT-L/14 @336</text>
<text x="320" y="160" text-anchor="middle" font-size="11" fill="#71717a">視覚特徴</text>
<line x1="304" y1="175" x2="327" y2="175" stroke="#71717a" stroke-width="2"/>
<polygon points="336,175 327,170 327,180" fill="#71717a"/>
<rect x="398" y="118" width="84" height="18" rx="4" fill="#16a34a"/>
<text x="440" y="131" text-anchor="middle" font-size="10" font-weight="700" fill="#ffffff">変更点①</text>
<rect x="340" y="140" width="140" height="72" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="410" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">コネクタ</text>
<text x="410" y="192" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">2層MLP (GELU)</text>
<text x="496" y="160" text-anchor="middle" font-size="11" fill="#71717a">視覚トークン</text>
<line x1="480" y1="176" x2="503" y2="176" stroke="#71717a" stroke-width="2"/>
<polygon points="512,176 503,171 503,181" fill="#71717a"/>
<rect x="516" y="120" width="176" height="120" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="604" y="174" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">大規模言語モデル</text>
<text x="604" y="196" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">Vicuna v1.5 (7B/13B)</text>
<line x1="604" y1="120" x2="604" y2="95" stroke="#71717a" stroke-width="2"/>
<polygon points="604,86 599,95 609,95" fill="#71717a"/>
<text x="604" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">応答テキスト（短答 / 自由文）</text>
<rect x="516" y="276" width="176" height="52" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="604" y="300" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">ユーザ指示</text>
<text x="604" y="318" text-anchor="middle" font-size="11" fill="#52525b">（プロンプト）</text>
<line x1="604" y1="276" x2="604" y2="249" stroke="#71717a" stroke-width="2"/>
<polygon points="604,240 599,249 609,249" fill="#71717a"/>
</svg><figcaption>骨格は LLaVA と同一で、<b>コネクタ・解像度・学習データ</b>の3点だけを差し替えます。視覚エンコーダと LLM は学習済みを流用し、その間を結ぶ<b>軽量コネクタ</b>が橋渡しの主役。</figcaption></figure>

「LLaVA → 1.5」の差分を一望すると、次のように整理できます。

| 観点 | LLaVA（初代） | LLaVA-1.5 |
| --- | --- | --- |
| コネクタ | 線形 1層（`W·z`） | **2層 MLP（GELU 非線形）** |
| 視覚エンコーダ | CLIP ViT-L/14（224px） | **CLIP ViT-L/14@336px** |
| 学習データ | 会話/詳細記述/推論の指示データ中心 | **＋学術VQA系**（VQAv2 / GQA / OKVQA / A-OKVQA / OCRVQA など）＋region系 |
| 応答制御 | 短答が苦手・yes/no に偏りがち | **応答フォーマット指示プロンプト**で短答/長文を制御 |
| ベースLLM | Vicuna（初期系列） | **Vicuna v1.5（7B / 13B）** |
| 学習枠組み | 2段階（特徴整列の事前学習 → 指示チューニング） | **同じ2段階**を踏襲 |

重要なのは、これらが **LLaVA の枠組みに「直交する（orthogonal）」改善**――すなわち枠組みを壊さずそのまま足せる――である点です（論文の言う orthogonal はこの意味で、主眼は MLP コネクタと学術データの追加）。だからこそ各変更の寄与を表（アブレーション）で一段ずつ積み上げて検証できています。

## 変更点①：線形 → 2層 MLP projector

初代 LLaVA のコネクタは、視覚特徴 `z` を言語埋め込み空間へ写す **単一の線形写像 `W·z`** でした。これは驚くほど強力かつデータ効率的でしたが、写像が線形である以上、視覚特徴と言語トークンの対応づけに表現力の上限があります。LLaVA-1.5 は、自己教師あり学習で「線形→MLP に替えると表現が良くなる」知見にならい、コネクタを **2層 MLP（間に GELU 非線形）** へ置き換えました。

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="線形1層projectorと2層MLP projectorの構造対比" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="30" y="42" font-size="13" font-weight="700" fill="#3730a3">LLaVA（従来）：線形 1層</text>
<rect x="30" y="55" width="120" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="90" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">視覚特徴 z</text>
<line x1="150" y1="83" x2="181" y2="83" stroke="#71717a" stroke-width="2"/>
<polygon points="190,83 181,78 181,88" fill="#71717a"/>
<rect x="194" y="55" width="130" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="259" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">線形 W·z</text>
<line x1="324" y1="83" x2="355" y2="83" stroke="#71717a" stroke-width="2"/>
<polygon points="364,83 355,78 355,88" fill="#71717a"/>
<rect x="368" y="55" width="150" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="443" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">言語トークン</text>
<text x="545" y="80" font-size="11" fill="#71717a">1層・軽量だが</text>
<text x="545" y="96" font-size="11" fill="#71717a">表現力に上限</text>
<text x="30" y="180" font-size="13" font-weight="700" fill="#166534">LLaVA-1.5：2層 MLP（GELU）</text>
<rect x="30" y="195" width="120" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="90" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">視覚特徴 z</text>
<line x1="150" y1="223" x2="181" y2="223" stroke="#71717a" stroke-width="2"/>
<polygon points="190,223 181,218 181,228" fill="#71717a"/>
<rect x="192" y="195" width="84" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="234" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">線形 W₁</text>
<rect x="282" y="195" width="70" height="56" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="317" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">GELU</text>
<rect x="358" y="195" width="84" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="400" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">線形 W₂</text>
<line x1="442" y1="223" x2="470" y2="223" stroke="#71717a" stroke-width="2"/>
<polygon points="479,223 470,218 470,228" fill="#71717a"/>
<rect x="482" y="195" width="140" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="552" y="228" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">言語トークン</text>
<text x="482" y="272" font-size="11" fill="#16a34a">非線形で表現力↑・追加コストはわずか</text>
</svg><figcaption><b>線形1層</b>に GELU を挟んで<b>2層MLP</b>にするだけ。パラメータ増はごく小さいのに、視覚→言語の写像が非線形になり、マルチモーダル能力が底上げされます。<b>軽さと表現力の両立</b>がこの変更の旨味。</figcaption></figure>

ここで効いているのは「視覚エンコーダや LLM を太らせる」のではなく、**両者をつなぐ細い管だけを少し賢くする**という発想です。視覚側・言語側はどちらも学習済みの強力なモデルなので、橋渡しの自由度を一段上げるだけで、両者の整合がよくなる。追加コストはほぼ無視できるレベルに留まります。

## 変更点②：解像度 336 と学術VQAデータ

### 解像度を 336px へ

初代 LLaVA は CLIP の 224px 入力を使っていました。LLaVA-1.5 は、CLIP で利用可能な **最高解像度である ViT-L/14@336px** に視覚エンコーダを差し替えます。解像度が上がるとパッチが細かくなり、視覚トークン数が増えて、画像中の細部や文字を「見える」ようになります。論文は、この高解像度化が **OCR・細部知覚で効き、幻覚（hallucination）の低減**にも寄与すると報告しています。

<figure class="lec-fig"><svg viewBox="0 0 720 300" role="img" aria-label="入力解像度224と336のパッチ密度の対比。336の方がパッチが細かく視覚トークンが増える" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="70" y="60" width="140" height="140" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<line x1="105" y1="60" x2="105" y2="200" stroke="#6366f1" stroke-width="1"/>
<line x1="140" y1="60" x2="140" y2="200" stroke="#6366f1" stroke-width="1"/>
<line x1="175" y1="60" x2="175" y2="200" stroke="#6366f1" stroke-width="1"/>
<line x1="70" y1="95" x2="210" y2="95" stroke="#6366f1" stroke-width="1"/>
<line x1="70" y1="130" x2="210" y2="130" stroke="#6366f1" stroke-width="1"/>
<line x1="70" y1="165" x2="210" y2="165" stroke="#6366f1" stroke-width="1"/>
<text x="140" y="225" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">224px：粗いパッチ</text>
<text x="140" y="245" text-anchor="middle" font-size="11" fill="#71717a">（従来のCLIP入力）</text>
<line x1="255" y1="130" x2="415" y2="130" stroke="#71717a" stroke-width="2"/>
<polygon points="424,130 415,125 415,135" fill="#71717a"/>
<text x="335" y="116" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">高解像度化（ViT-L/14@336）</text>
<rect x="430" y="60" width="140" height="140" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
<line x1="453" y1="60" x2="453" y2="200" stroke="#06b6d4" stroke-width="1"/>
<line x1="477" y1="60" x2="477" y2="200" stroke="#06b6d4" stroke-width="1"/>
<line x1="500" y1="60" x2="500" y2="200" stroke="#06b6d4" stroke-width="1"/>
<line x1="523" y1="60" x2="523" y2="200" stroke="#06b6d4" stroke-width="1"/>
<line x1="547" y1="60" x2="547" y2="200" stroke="#06b6d4" stroke-width="1"/>
<line x1="430" y1="83" x2="570" y2="83" stroke="#06b6d4" stroke-width="1"/>
<line x1="430" y1="107" x2="570" y2="107" stroke="#06b6d4" stroke-width="1"/>
<line x1="430" y1="130" x2="570" y2="130" stroke="#06b6d4" stroke-width="1"/>
<line x1="430" y1="153" x2="570" y2="153" stroke="#06b6d4" stroke-width="1"/>
<line x1="430" y1="177" x2="570" y2="177" stroke="#06b6d4" stroke-width="1"/>
<text x="500" y="225" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">336px：細かいパッチ</text>
<text x="500" y="245" text-anchor="middle" font-size="11" fill="#71717a">OCR・細部・幻覚低減に効く</text>
<text x="648" y="124" text-anchor="middle" font-size="11" fill="#71717a">視覚トークン</text>
<text x="648" y="140" text-anchor="middle" font-size="11" fill="#71717a">が増える</text>
</svg><figcaption>同じ ViT-L/14 でも入力を <b>336px</b> にするとパッチが細かくなり、文字や小さな対象まで「見える」。<b>OCR・細部知覚</b>が伸び、<b>幻覚も減る</b>のが定性的な効果です。なお、さらに上の解像度はグリッド分割（LLaVA-1.5-HD）で扱います。</figcaption></figure>

### 学術VQA系データと応答フォーマット指示

初代 LLaVA は会話・詳細記述・複雑推論といった「長め・自由文」の指示データが中心で、**短答（単語/句）を求める学術VQAが苦手**でした。質問に対して安易に "yes" と答えてしまう、といった偏りも見られます。LLaVA-1.5 は **VQAv2 / GQA / OKVQA / A-OKVQA / OCRVQA / TextCaps** など学術タスク指向のデータを指示チューニングに加え、視覚知識と短答能力を補強します（さらに Visual Genome / RefCOCO など region 系で細かな位置把握も強化）。

ただしデータを足すだけでは、短答データに引っぱられて自由会話まで短文化してしまう危険があります。これを **応答フォーマット指示プロンプト**で解きます。VQA の質問末尾に、出力形式をはっきり指示する一文を付ける —— たとえば短答を求めたいときは次の一文を添える、という単純な手です。

```text
# 短答（VQA系）を求めるとき、質問の末尾に付けるプロンプト
Answer the question using a single word or phrase.

# A-OKVQA を多肢選択にしたときの指示例
Answer with the option's letter from the given choices directly.
```

このひと工夫で、「曖昧な書式指定（`Q: ... A: ...` のような）」が原因の短文偏重を避けつつ、ユーザの意図に応じて短答と長文を**出し分け**られるようになります。論文では、VQAv2 を加えるだけでも形式指示によって MME スコアが大きく改善する（およそ 809.6 → 1323.8）ことを示しており、形式制御が「データを活かすための前提」だと分かります。

| プロンプト種別 | 質問 | 応答の傾向 |
| --- | --- | --- |
| 通常プロンプト | シャツの色は？ | 文で説明（長め） |
| 曖昧プロンプト（`Q:…A:`） | 同上 | 形式が定まらず不安定 |
| **形式指示プロンプト** | 同上 ＋「単語/句で答えよ」 | **短答（例：黄色）** に安定 |

なお基盤 LLM は **Vicuna v1.5（7B / 13B）** に更新され、学習は LLaVA の **2段階**（① 画像–テキスト対で視覚特徴を言語埋め込み空間へ整列させる事前学習 → ② 指示チューニング）をそのまま踏襲します。事前学習に約 558K 対、指示チューニングに約 665K の指示データを使い、最終 13B でも **計約 1.2M の公開データ**のみ。8×A100 一台で**おおむね 1 日**で学習を終えられる規模感です。

## なぜ「最小の変更」で効くのか

LLaVA-1.5 の魅力は、性能の高さそのものよりも「**最小の変更で、説明可能なまま、誰でも再現できる**」という三拍子にあります。

- **足し込み式ゆえの分かりやすさ**：コネクタ・解像度・データはいずれも枠組みを壊さず足せる改善なので、寄与をアブレーションで一段ずつ積み上げて説明できる。「どれが効いたのか」が霧の中に消えない。
- **再サンプラも巨大事前学習も不要**：Q-Former のような特別な視覚再サンプラや、数億〜十数億規模の画像–テキスト事前学習を使う手法に対し、LLaVA-1.5 は**最もシンプルな全結合コネクタ**と**公開データのみ**で互角以上に戦える。視覚エンコーダ（CLIP 等）が既に web 規模で学習済みである以上、巨大な追加事前学習は必ずしも要らない、という示唆でもある。
- **学術的に手頃**：公開データのみ・短時間学習・単純構造なので、研究室レベルで丸ごと再現でき、改造の出発点にしやすい。この「手頃さ」こそが、LLaVA-1.5 を **事実上の標準ベースライン**へ押し上げた最大の理由です。

つまり LLaVA-1.5 は「より大きく・より複雑に」ではなく、「**既に強い部品をシンプルに正しくつなぐ**」方向で MLLM の到達点を引き上げ、その再現容易性によって後続研究の共通の足場になった —— と理解するのが要点です。

## まとめと、読解後に答えたい問い

LLaVA-1.5 の核心は「**骨格は変えず、コネクタ（線形→2層MLP）・解像度（→336px）・データ（学術VQA＋応答フォーマット指示）の3点だけを足す**」こと。これらは枠組みを壊さず足せて、公開データのみ・短時間学習で再現でき、多数ベンチで SoTA 級に届く。だからこそ MLLM 研究の**標準ベースライン**になりました。

読解後、自分の言葉で答えられるか確認したい問い。

- 線形1層を2層MLP（GELU）に替えると、なぜ「軽さを保ったまま」表現力が上がるのか。視覚側・言語側を太らせる案と何が違うか。
- 学術VQAデータを足す効果は、なぜ**応答フォーマット指示プロンプト**とセットで初めて活きるのか。形式指示がない場合に何が壊れるか。
- 224 → 336px の高解像度化は、どんなタスク（OCR・細部・幻覚）でどう効くと説明できるか。さらに上の解像度はどう扱う（グリッド分割）か。
- 「最小の変更」「公開データのみ」「短時間学習」が、性能以上に**標準ベースライン化**へ効いた理由を説明できるか。
