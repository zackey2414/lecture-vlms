# Janus-Pro — Unified Multimodal Understanding & Generation with Scaling

Janus-Pro（arXiv:2501.17811、DeepSeek-AI）は、前章の **Janus**——「理解」と「生成」で視覚エンコーディングを**分離**しつつ、単一の自己回帰トランスフォーマで両タスクを**統合**するモデル——をそのまま土台にして、性能を一段引き上げた発展版である。アーキテクチャは Janus と同一で、新規モジュールは増えていない。では何が変わったのか。論文の主張はきわめてシンプルで、**(1) 学習戦略の最適化・(2) 学習データの拡張・(3) モデル規模の拡大**という三点だけである。本章はこの「三点でどこまで伸ばせるか」を腹落ちさせることを目的とする。Janus の枠組み（分離エンコード・統合トランスフォーマ）は既習前提とし、差分に集中する。

---

## Janus からの強化点（全体像）

まず構成を最小限おさらいする。Janus-Pro も Janus と同じく、**理解側は SigLIP エンコーダ**で画像から高次の意味特徴を取り出し（2次元グリッドを1次元へ平坦化し、2層 MLP のアダプタで LLM の入力空間へ射影）、**生成側は VQ トークナイザ**（コードブックサイズ $16{,}384$、ダウンサンプル係数 $16$）で画像を離散 ID 列へ変換し、別の2層 MLP アダプタで埋め込みへ写す。両者を連結した系列を**単一の自己回帰トランスフォーマ**へ流し、テキストは LLM 内蔵の予測ヘッド、画像はランダム初期化した別ヘッドで予測する。**「視覚エンコードを分離して、両タスクの表現の衝突を避ける」**という Janus の核心はそのまま受け継がれている。

Janus は 1.5B スケールで「分離エンコードが効く」ことを実証した先駆けだったが、**学習データが少なく容量も小さい**ため、短いプロンプトでの生成が不安定で、画質も振るわないという弱点があった。Janus-Pro はアーキテクチャをいじらず、この弱点を以下の三本柱で潰しにいく。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="Janusを土台にして、学習戦略・データ・モデル規模の三点で強化しJanus-Proに至る全体像" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="20" y="50" width="180" height="260" rx="10" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="110" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">Janus（前章の到達点）</text>
<rect x="36" y="90" width="70" height="52" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
<text x="71" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">理解</text>
<text x="71" y="130" text-anchor="middle" font-size="11" fill="#4338ca">SigLIP</text>
<rect x="114" y="90" width="70" height="52" rx="6" fill="#cffafe" stroke="#06b6d4" stroke-width="2"/>
<text x="149" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">生成</text>
<text x="149" y="130" text-anchor="middle" font-size="11" fill="#0e7490">VQ</text>
<rect x="36" y="156" width="148" height="44" rx="6" fill="#f4f4f5" stroke="#18181b" stroke-width="2"/>
<text x="110" y="183" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">単一ARトランスフォーマ</text>
<text x="110" y="230" text-anchor="middle" font-size="11" fill="#71717a">視覚エンコードを分離</text>
<text x="110" y="256" text-anchor="middle" font-size="11" fill="#dc2626">課題: 1.5Bのみ／データ少</text>
<text x="110" y="278" text-anchor="middle" font-size="11" fill="#dc2626">短プロンプトで不安定</text>
<rect x="252" y="66" width="210" height="64" rx="9" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="357" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">1. 学習戦略の最適化</text>
<text x="357" y="113" text-anchor="middle" font-size="11" fill="#0e7490">段階の分け方・ステップ配分</text>
<rect x="252" y="148" width="210" height="64" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="357" y="174" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">2. データ拡張</text>
<text x="357" y="195" text-anchor="middle" font-size="11" fill="#15803d">理解データ＋合成美的データ</text>
<rect x="252" y="230" width="210" height="64" rx="9" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="357" y="256" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">3. モデル規模拡大</text>
<text x="357" y="277" text-anchor="middle" font-size="11" fill="#4338ca">1.5B から 7B へ</text>
<rect x="520" y="50" width="180" height="260" rx="10" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
<text x="610" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">Janus-Pro</text>
<rect x="536" y="100" width="148" height="50" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="610" y="122" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">理解の精度↑</text>
<text x="610" y="140" text-anchor="middle" font-size="11" fill="#15803d">指示追従↑</text>
<rect x="536" y="162" width="148" height="50" rx="6" fill="#cffafe" stroke="#06b6d4" stroke-width="2"/>
<text x="610" y="184" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">生成が安定・美的↑</text>
<text x="610" y="202" text-anchor="middle" font-size="11" fill="#0e7490">短文でも崩れにくい</text>
<text x="610" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">1.5B / 7B の2サイズ</text>
<line x1="200" y1="178" x2="248" y2="178" stroke="#71717a" stroke-width="2"/>
<polygon points="248,178 238,173 238,183" fill="#71717a"/>
<line x1="464" y1="178" x2="516" y2="178" stroke="#71717a" stroke-width="2"/>
<polygon points="516,178 506,173 506,183" fill="#71717a"/>
</svg><figcaption>アーキテクチャは Janus と<b>完全に同一</b>。<b>学習戦略・データ・モデル規模</b>の三点だけで、理解の精度と生成の安定性・美的品質を同時に底上げするのが Janus-Pro の主張。</figcaption></figure>

「新しい仕掛けを足さず、レシピと素材と規模で押し切る」という思想なので、各強化点が**なぜ効くのか**を理解することが本章のすべてになる。順に見ていく。

---

## 学習戦略の最適化

Janus は **3段階**で学習していた。**Stage I** はアダプタと画像ヘッドだけを訓練、**Stage II** は理解エンコーダ・生成エンコーダを除く全体で統合事前学習、**Stage III** は理解エンコーダのパラメータも解凍して教師ありファインチューニング（SFT）を行う、という流れである。Janus-Pro はこの 3段階の骨格は保ちつつ、**段階の役割分担とステップ配分**を見直した。

### Stage II の「ImageNet 偏重」という無駄

論文がまず問題視するのは Stage II の中身である。Janus は PixArt にならい、Stage II の text-to-image 学習を二つに割っていた。**前半は ImageNet データ**を使い、画像カテゴリ名をプロンプトとして与えて「画素同士の依存（ピクセル依存）」をモデル化させる。**後半は通常の text-to-image データ**で密な記述から画像を生成させる。問題は配分で、**Stage II の text-to-image ステップの約 $2/3$（66.67%）が前半の ImageNet 部分**に充てられていた。論文は、これが**最適でなく、計算的にも非効率**だと結論する。

### 二つの修正 — Stage I を延ばし、Stage II を集中させる

そこで Janus-Pro は二点を変える。

- **Stage I の延長**: Stage I の訓練ステップを増やし、ImageNet 上で十分に学習させる。著者らは、**LLM のパラメータを固定したままでも**、モデルはピクセル依存を十分にモデル化でき、カテゴリ名から妥当な画像を生成できることを観察した。つまり「画素の文法」を覚える仕事は、早い安価な段階に前倒しできる。
- **Stage II の集中（Focused Training）**: Stage II からは ImageNet データを**外し**、通常の text-to-image データだけで「密な記述に基づく生成」を直接訓練する。これにより Stage II が text-to-image データをより効率的に使えるようになり、訓練効率と最終性能の双方が改善する。

直感的には、**「基礎的な画素の練習（Stage I）」と「記述に沿った本番の生成（Stage II）」を段階で切り分けた**のがポイントだ。Janus では両者が Stage II 内で混在し、計算量の多くが基礎練習に吸われていた。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="Janusの3段学習とJanus-Proの改良。Stage Iを延長しImageNetを吸収、Stage IIをtext-to-imageに集中、Stage IIIのデータ比を調整する対比図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="30" y="44" font-size="13" font-weight="700" fill="#71717a">Janus（従来の3段）</text>
<rect x="30" y="56" width="190" height="92" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="125" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">Stage I</text>
<text x="125" y="102" text-anchor="middle" font-size="11" fill="#71717a">アダプタ＋画像ヘッド</text>
<text x="125" y="122" text-anchor="middle" font-size="11" fill="#71717a">のみ訓練（短め）</text>
<rect x="265" y="56" width="190" height="92" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="360" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">Stage II</text>
<text x="360" y="102" text-anchor="middle" font-size="11" fill="#dc2626">ImageNet 約2/3</text>
<text x="360" y="122" text-anchor="middle" font-size="11" fill="#dc2626">＋ T2I 約1/3（非効率）</text>
<rect x="500" y="56" width="190" height="92" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="595" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">Stage III（SFT）</text>
<text x="595" y="102" text-anchor="middle" font-size="11" fill="#71717a">理解エンコーダも解凍</text>
<text x="595" y="122" text-anchor="middle" font-size="11" fill="#71717a">データ比 7:3:10</text>
<line x1="220" y1="102" x2="263" y2="102" stroke="#71717a" stroke-width="2"/>
<polygon points="263,102 253,97 253,107" fill="#71717a"/>
<line x1="455" y1="102" x2="498" y2="102" stroke="#71717a" stroke-width="2"/>
<polygon points="498,102 488,97 488,107" fill="#71717a"/>
<text x="30" y="218" font-size="13" font-weight="700" fill="#4338ca">Janus-Pro（改良）</text>
<rect x="30" y="230" width="190" height="92" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="125" y="254" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">Stage I（延長）</text>
<text x="125" y="276" text-anchor="middle" font-size="11" fill="#15803d">ImageNetを十分に吸収</text>
<text x="125" y="296" text-anchor="middle" font-size="11" fill="#15803d">画素依存を前倒しで習得</text>
<rect x="265" y="230" width="190" height="92" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="360" y="254" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">Stage II（集中）</text>
<text x="360" y="276" text-anchor="middle" font-size="11" fill="#0e7490">ImageNetを外す</text>
<text x="360" y="296" text-anchor="middle" font-size="11" fill="#0e7490">T2Iのみ・密な記述で生成</text>
<rect x="500" y="230" width="190" height="92" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="595" y="254" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">Stage III（SFT）</text>
<text x="595" y="276" text-anchor="middle" font-size="11" fill="#4338ca">データ比を調整</text>
<text x="595" y="296" text-anchor="middle" font-size="11" fill="#4338ca">7:3:10 から 5:1:4 へ</text>
<line x1="220" y1="276" x2="263" y2="276" stroke="#4338ca" stroke-width="2"/>
<polygon points="263,276 253,271 253,281" fill="#4338ca"/>
<line x1="455" y1="276" x2="498" y2="276" stroke="#4338ca" stroke-width="2"/>
<polygon points="498,276 488,271 488,281" fill="#4338ca"/>
</svg><figcaption>従来は Stage II の生成ステップの<b>約2/3が ImageNet</b>に吸われていた。Janus-Pro は<b>Stage I を延長</b>して画素依存を先に習得し、<b>Stage II を本番の text-to-image に集中</b>させる。役割分担を段階で切り分けるのが要点。</figcaption></figure>

### Stage III のデータ比の再調整

最後に Stage III の SFT で、**マルチモーダル理解データ : 純テキストデータ : text-to-image データ**の比率を **$7:3:10$ から $5:1:4$** へ変えた。text-to-image データの割合をやや**減らす**ことで、生成能力を強く保ったまま、**理解性能を改善**できると報告している。生成側を欲張りすぎず、理解側に配分を戻したわけだ。表で示された訓練の規模感としては、Stage I が約 $20\mathrm{K}$ ステップ、Stage II が約 $360\mathrm{K}$ ステップ（早期終了で $270\mathrm{K}$ 付近）、Stage III が $1.5\mathrm{B}$ で $80\mathrm{K}$・$7\mathrm{B}$ で $40\mathrm{K}$ ステップ程度であり、最適化器は AdamW（$\beta_1{=}0.9,\ \beta_2{=}0.95$）、学習率は段階ごとに $1.0\times10^{-3}\to1.0\times10^{-4}\to4.0\times10^{-5}$ と下げていく定数スケジュールが採られている。

---

## データとモデル規模の拡張

### 理解データ — DeepSeek-VL2 系を引き込む

理解側では、Stage II の事前学習データに **約 9000 万（$\sim 90\mathrm{M}$）サンプル**を追加した。内訳は画像キャプション（YFCC など）に加え、**表・図・文書理解**（Docmatix など）といった、構造のあるドキュメント系データである。さらに Stage III の SFT では DeepSeek-VL2 由来のデータ——MEME 理解、中国語の会話データ、対話体験を高めるためのデータ——を取り込んだ。これらにより扱えるタスクの幅が広がり、会話の質も向上したとされる。要するに、**理解の伸びはおおむね「良質で多様なデータをどれだけ流せたか」**で説明される。

### 生成データ — 合成の「美的」データで安定させる

生成側で論文が指摘するのは、**Janus が使っていた実画像データの質の低さとノイズ**である。これが text-to-image 生成を不安定にし、美的に劣る出力を生んでいた。Janus-Pro はここに **約 7200 万（$\sim 72\mathrm{M}$）サンプルの合成美的データ**を投入し、統合事前学習における**実データ : 合成データの比を $1:1$** に揃えた（プロンプトは公開されているものを利用）。実験的には、**合成データで学習するとモデルの収束が速く**、出力はより安定し、美的品質も大きく改善した。「綺麗に揃った教師」を半分混ぜることで、生成の振る舞いが落ち着くという、直感に沿った結果である。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="理解データの追加と合成美的データの投入、そして1.5Bから7Bへのモデル規模拡大を示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="20" y="50" width="210" height="270" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="125" y="76" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">理解データの拡張</text>
<rect x="38" y="92" width="174" height="40" rx="6" fill="#bbf7d0" stroke="#16a34a" stroke-width="2"/>
<text x="125" y="117" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">Stage II に 約90M 追加</text>
<text x="125" y="158" text-anchor="middle" font-size="11" fill="#15803d">画像キャプション（YFCC）</text>
<text x="125" y="180" text-anchor="middle" font-size="11" fill="#15803d">表・図・文書（Docmatix）</text>
<text x="125" y="216" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">Stage III に追加</text>
<text x="125" y="238" text-anchor="middle" font-size="11" fill="#15803d">MEME・中国語会話</text>
<text x="125" y="260" text-anchor="middle" font-size="11" fill="#15803d">対話強化データ</text>
<text x="125" y="296" text-anchor="middle" font-size="11" fill="#71717a">DeepSeek-VL2 系を参照</text>
<rect x="250" y="50" width="200" height="270" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="350" y="76" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">生成データの拡張</text>
<rect x="288" y="150" width="48" height="120" rx="4" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="312" y="288" text-anchor="middle" font-size="11" fill="#71717a">実データ</text>
<rect x="362" y="150" width="48" height="120" rx="4" fill="#06b6d4" stroke="#0e7490" stroke-width="2"/>
<text x="386" y="288" text-anchor="middle" font-size="11" fill="#0e7490">合成 約72M</text>
<text x="350" y="120" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">実 : 合成 = 1 : 1</text>
<text x="350" y="312" text-anchor="middle" font-size="11" fill="#0e7490">収束↑・安定・美的↑</text>
<rect x="470" y="50" width="230" height="270" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="585" y="76" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">モデル規模の拡大</text>
<rect x="500" y="160" width="56" height="60" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="528" y="195" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">1.5B</text>
<rect x="612" y="110" width="72" height="160" rx="6" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
<text x="648" y="195" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">7B</text>
<line x1="560" y1="190" x2="608" y2="190" stroke="#4338ca" stroke-width="2"/>
<polygon points="608,190 598,185 598,195" fill="#4338ca"/>
<text x="585" y="298" text-anchor="middle" font-size="11" fill="#4338ca">層 24→30 / 埋込 2048→4096</text>
<text x="585" y="314" text-anchor="middle" font-size="11" fill="#4338ca">損失の収束が速い</text>
</svg><figcaption>理解側は<b>約90M の追加データ</b>（キャプション・文書系）で底上げ。生成側は<b>約72M の合成美的データ</b>を入れ<b>実:合成=1:1</b>に。さらに<b>1.5B から 7B へ</b>拡大すると、両タスクとも損失の収束が顕著に速くなる。</figcaption></figure>

### モデル規模 — 1.5B と 7B の2サイズ

Janus は分離エンコードの有効性を 1.5B の LLM で検証したが、Janus-Pro は基盤 LLM を **DeepSeek-LLM の 1.5B と 7B** の二系列に広げた（論文の表では小型側を「1B」と略記）。設定上の差は素直で、埋め込み次元 $2048\to4096$、層数 $24\to30$、アテンションヘッド $16\to32$ と増え、コンテキスト長 $4096$・語彙 $100\mathrm{K}$ は共通である。重要な観察は、**大きい LLM を使うと、理解・生成の双方で損失の収束が明確に速くなる**点だ。これは「分離エンコード＋統合トランスフォーマ」という設計が**素直にスケールする**ことの証拠であり、三本柱の中でも効果が大きい。実装は HAI-LLM フレームワークで、解像度はすべて $384\times384$、訓練は 1.5B / 7B でそれぞれ 16 / 32 ノード（各 8 枚の A100 40GB）を用いて約 9 / 14 日を要したと報告されている。

---

## 理解・生成ベンチでの改善

三本柱の効果は、**理解ベンチと text-to-image 生成ベンチの双方**で確認できる。Janus（1.5B）を基準に、Janus-Pro-1B、Janus-Pro-7B へと段階的に伸びていくのが基本的な読み筋である。

**理解側**では、総合的なマルチモーダル理解ベンチ MMBench のスコアが Janus の $69.4$ から、Janus-Pro-1B で $75.5$、Janus-Pro-7B で **$79.2$** へと上がり、TokenFlow（$68.9$）や MetaMorph（$75.2$）といった統合モデルを上回る。MMMU でも $30.5 \to 36.3 \to 41.0$ と素直に伸びる。論文は、**理解の好成績を「視覚エンコードの分離」が両タスクの衝突を和らげている**ことに帰し、より大きな TokenFlow-XL（13B）に対しても、GQA を除く全ベンチで Janus-Pro-7B が上回ると述べる。

**生成側**では、構成的な指示追従を測る GenEval の総合スコアが Janus の $0.61$ から、Janus-Pro-1B で $0.73$、Janus-Pro-7B で **$0.80$** へ。これは Stable Diffusion 3 Medium（$0.74$）や DALL-E 3（$0.67$）、Transfusion（$0.63$）といった、生成専用モデルを含めて上回る水準である。密で長いプロンプトへの忠実度を測る DPG-Bench でも $79.68 \to 82.63 \to 84.19$ と一貫して改善し、表中の他手法を上回る。**理解専用でも生成専用でもない「統合モデル」が、両軸で競争力を持つ**ことを示せたのが、この章の最大の成果だ。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="理解のMMBenchと生成のGenEvalで、JanusからJanus-Pro-1B、Janus-Pro-7Bへとスコアが段階的に伸びる棒グラフ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<text x="195" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">理解: MMBench スコア</text>
<line x1="60" y1="300" x2="340" y2="300" stroke="#71717a" stroke-width="2"/>
<rect x="80" y="115" width="50" height="185" rx="4" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="105" y="108" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">69.4</text>
<text x="105" y="318" text-anchor="middle" font-size="11" fill="#71717a">Janus</text>
<rect x="150" y="99" width="50" height="201" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
<text x="175" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">75.5</text>
<text x="175" y="318" text-anchor="middle" font-size="11" fill="#4338ca">Pro-1B</text>
<rect x="220" y="89" width="50" height="211" rx="4" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
<text x="245" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">79.2</text>
<text x="245" y="318" text-anchor="middle" font-size="11" fill="#4338ca">Pro-7B</text>
<text x="535" y="40" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">生成: GenEval 総合（％）</text>
<line x1="400" y1="300" x2="680" y2="300" stroke="#71717a" stroke-width="2"/>
<rect x="420" y="137" width="50" height="163" rx="4" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="445" y="130" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">61</text>
<text x="445" y="318" text-anchor="middle" font-size="11" fill="#71717a">Janus</text>
<rect x="490" y="105" width="50" height="195" rx="4" fill="#cffafe" stroke="#06b6d4" stroke-width="2"/>
<text x="515" y="98" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">73</text>
<text x="515" y="318" text-anchor="middle" font-size="11" fill="#0e7490">Pro-1B</text>
<rect x="560" y="87" width="50" height="213" rx="4" fill="#06b6d4" stroke="#0e7490" stroke-width="2"/>
<text x="585" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">80</text>
<text x="585" y="318" text-anchor="middle" font-size="11" fill="#0e7490">Pro-7B</text>
</svg><figcaption>理解（MMBench）も生成（GenEval）も、<b>Janus → Pro-1B → Pro-7B</b> と段階的に伸びる。1.5B のままでも三本柱で明確に改善し、7B でさらに上振れする。MMMU は 30.5→41.0、DPG-Bench は 79.7→84.2 と同傾向。</figcaption></figure>

注意したいのは、Janus-Pro-1B が**規模は据え置き**のまま Janus を明確に上回っている点だ。これは伸びの相当部分が**学習戦略とデータ**から来ていることを示しており、7B へのスケールはそこに**上積み**として効いている、という二段の読み方ができる。

---

## まとめと、読解後に答えたい問い

Janus-Pro の貢献は、**「分離エンコード＋統合トランスフォーマ」という Janus の設計を一切いじらず、レシピ（学習戦略）・素材（データ）・規模（モデル）の三点だけで、理解と生成を同時に底上げできる**ことを実証したことに尽きる。具体的には、(1) Stage I を延長して画素依存を前倒し習得させ、Stage II を本番の text-to-image に集中させ、Stage III のデータ比を理解寄りに調整、(2) 理解に約 90M、生成に約 72M の合成美的データを足して実:合成を $1:1$ に、(3) 基盤 LLM を 1.5B から 7B へ拡大、という三手である。

一方で論文が認める限界もはっきりしている。入力解像度が $384\times384$ に制限されるため、**OCR のような細粒度タスクでは不利**であり、生成側も低解像度と VQ トークナイザの再構成損失が重なって、意味的には豊かでも**細部（小さな顔領域など）が甘くなる**。解像度の引き上げが次の自然な一手だと示唆されている。

読解後に、次の問いに自分の言葉で答えられるか確認してほしい。

- なぜ Janus の Stage II は「非効率」だったのか。ImageNet 部分を Stage I へ移すと、何が安く済むのか。
- Stage III のデータ比を $7:3:10$ から $5:1:4$ に変えた狙いは何か。生成と理解のどちらに配分を寄せ、その結果どちらが伸びたか。
- 生成データで「実:合成 = $1:1$」にすると、なぜ収束が速くなり出力が安定すると考えられるか。合成データの何が効いているのか。
- 1.5B → 7B のスケールで「両タスクの損失収束が速くなった」ことは、分離エンコード設計のどんな性質を裏づけるか。
- Janus-Pro-1B が同規模の Janus を上回った事実から、伸びの要因（戦略・データ・規模）をどう切り分けて理解できるか。
- $384\times384$ の解像度制約は、理解（OCR）と生成（細部）でそれぞれどんな形で表面化するか。解像度を上げる以外に緩和策は考えられるか。
