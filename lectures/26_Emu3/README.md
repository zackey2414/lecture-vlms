# Emu3 — Next-Token Prediction is All You Need

Chameleon（24章）で、画像もテキストも離散トークンに変換して 1 つのトランスフォーマで早期融合（early-fusion）する考え方を学んだ。**Emu3**（BAAI, arXiv:2409.18869）はその路線をさらに推し進め、画像・テキストに加えて**動画**まで離散トークン化し、生成（text-to-image / text-to-video）と理解（VQA・キャプション）の両方を、たった 1 つの「次トークン予測」モデルで統一する。タイトルが宣言する通り「次トークン予測こそすべて（Next-Token Prediction is All You Need）」であり、**拡散モデルも CLIP も合成的な設計も一切使わない**のが思想の核である。

この章では、Emu3 がどのように視覚情報を離散トークンへ落とし込み、なぜそれだけで理解も生成もこなせるのか、そして Chameleon との異同を整理する。

## 全体像（まず一枚で）

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="動画・画像・テキストを離散トークンに変換し、単一トランスフォーマが次トークン予測で出力するEmu3の全体像" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="m1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#4338ca"/></marker>
  </defs>
  <rect x="15" y="45" width="84" height="54" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="57" y="78" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">動画</text>
  <rect x="15" y="152" width="84" height="54" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="57" y="185" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">画像</text>
  <rect x="15" y="259" width="84" height="54" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="57" y="292" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">テキスト</text>
  <rect x="140" y="118" width="108" height="122" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="194" y="166" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3"><tspan x="194">視覚／テキスト</tspan><tspan x="194" dy="19">トークナイザ</tspan></text>
  <text x="194" y="220" text-anchor="middle" font-size="12" fill="#4338ca">連続値 → 離散トークン</text>
  <line x1="99" y1="72" x2="137" y2="140" stroke="#4338ca" stroke-width="2" marker-end="url(#m1)"/>
  <line x1="99" y1="179" x2="137" y2="179" stroke="#4338ca" stroke-width="2" marker-end="url(#m1)"/>
  <line x1="99" y1="286" x2="137" y2="218" stroke="#4338ca" stroke-width="2" marker-end="url(#m1)"/>
  <text x="345" y="148" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">離散トークン列（共通の語彙）</text>
  <rect x="286" y="165" width="14" height="30" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="303" y="165" width="14" height="30" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="320" y="165" width="14" height="30" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="337" y="165" width="14" height="30" rx="3" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <rect x="354" y="165" width="14" height="30" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="371" y="165" width="14" height="30" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="388" y="165" width="14" height="30" rx="3" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <line x1="248" y1="179" x2="284" y2="179" stroke="#4338ca" stroke-width="2" marker-end="url(#m1)"/>
  <line x1="404" y1="179" x2="428" y2="179" stroke="#4338ca" stroke-width="2" marker-end="url(#m1)"/>
  <rect x="430" y="78" width="128" height="202" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="494" y="158" text-anchor="middle" font-size="14" font-weight="700" fill="#166534"><tspan x="494">単一トランスフォーマ</tspan><tspan x="494" dy="20">（デコーダ専用）</tspan><tspan x="494" dy="20">次トークン予測</tspan></text>
  <text x="596" y="150" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">脱トークン化</text>
  <line x1="558" y1="150" x2="630" y2="95" stroke="#16a34a" stroke-width="2"/>
  <line x1="558" y1="180" x2="630" y2="180" stroke="#16a34a" stroke-width="2"/>
  <line x1="558" y1="210" x2="630" y2="265" stroke="#16a34a" stroke-width="2"/>
  <rect x="632" y="70" width="78" height="48" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="671" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">動画</text>
  <rect x="632" y="156" width="78" height="48" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="671" y="186" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像</text>
  <rect x="632" y="242" width="78" height="48" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="671" y="272" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">テキスト</text>
</svg>
<figcaption>動画・画像・テキストをすべて同じ離散トークン列にしてから、ゼロから学習した 1 つのデコーダで次トークンを予測する。<b>要点</b>＝入力も出力も同じトークン空間なので、生成と理解を 1 つのモデルに畳み込める。</figcaption>
</figure>

Emu3 の主張は驚くほど単純である。言語モデルが文を

$$p(x)=\prod_i p(x_i\mid x_{<i})$$

と左から順に生成するのと**まったく同じ枠組み**を、画像にも動画にも適用する。鍵になるのは、連続値のピクセルを離散トークンに変換する**視覚トークナイザ**だ。一度すべてが離散トークン列になってしまえば、「これはテキスト」「これは画像」「これは動画」という区別は消え、語彙が拡張された 1 つの巨大な系列予測問題に還元される。

- **入力**: 言語・画像・動画を混ぜた系列データ。
- **モデル**: Llama-2 系のデコーダ専用トランスフォーマを**ゼロから**学習（約 8B パラメータ）。主な改造は、離散視覚トークンを収めるための埋め込み層・語彙の拡張だけ。
- **出力**: 次トークンを 1 つずつ予測し、視覚トークンなら脱トークン化（detokenize）して画像・動画に戻す。

専用の理解用エンコーダ（CLIP）も、専用の生成器（拡散 UNet）も外付けしない。「トークン」という一点に収束させることで設計が劇的に単純化され、学習・推論ともにスケールさせやすくなる、というのが論文の立場である。

## 画像・動画のトークン化

理解も生成も次トークン予測に乗せる前提として、まず**画像・動画を離散トークンに変換できなければならない**。Emu3 はこのための視覚トークナイザを自前で学習する。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="視覚トークナイザが画像や動画をエンコーダとコードブックで離散トークン列に変換し、行区切りとフレーム区切りの特殊トークンを挟む流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="m2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#0e7490"/></marker>
  </defs>
  <rect x="15" y="135" width="118" height="86" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="74" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b"><tspan x="74">画像 512×512</tspan><tspan x="74" dy="19">動画 4×512×512</tspan></text>
  <text x="74" y="210" text-anchor="middle" font-size="11" fill="#0e7490">連続値のピクセル</text>
  <rect x="158" y="128" width="118" height="100" rx="10" fill="#ecfeff" stroke="#06b6d4" stroke-width="2"/>
  <text x="217" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#155e63"><tspan x="217">エンコーダ</tspan><tspan x="217" dy="19">(MoVQGAN系)</tspan><tspan x="217" dy="18" font-size="11" font-weight="400">3D畳み込みで時間も</tspan></text>
  <rect x="300" y="120" width="126" height="120" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="363" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">コードブック</text>
  <text x="363" y="234" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">32,768 ベクトル</text>
  <g fill="#c7d2fe" stroke="#4338ca" stroke-width="1">
    <rect x="318" y="158" width="20" height="16" rx="2"/><rect x="342" y="158" width="20" height="16" rx="2"/><rect x="366" y="158" width="20" height="16" rx="2"/><rect x="390" y="158" width="20" height="16" rx="2"/>
    <rect x="318" y="178" width="20" height="16" rx="2"/><rect x="342" y="178" width="20" height="16" rx="2"/><rect x="366" y="178" width="20" height="16" rx="2"/><rect x="390" y="178" width="20" height="16" rx="2"/>
    <rect x="318" y="198" width="20" height="16" rx="2"/><rect x="342" y="198" width="20" height="16" rx="2"/><rect x="366" y="198" width="20" height="16" rx="2"/><rect x="390" y="198" width="20" height="16" rx="2"/>
  </g>
  <line x1="133" y1="178" x2="156" y2="178" stroke="#0e7490" stroke-width="2" marker-end="url(#m2)"/>
  <line x1="276" y1="178" x2="298" y2="178" stroke="#0e7490" stroke-width="2" marker-end="url(#m2)"/>
  <text x="447" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#155e63">最近傍ID</text>
  <line x1="426" y1="178" x2="466" y2="178" stroke="#0e7490" stroke-width="2" marker-end="url(#m2)"/>
  <rect x="470" y="92" width="238" height="176" rx="10" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="589" y="116" text-anchor="middle" font-size="13" font-weight="700" fill="#155e63">離散トークン格子（1画像=4096個）</text>
  <g font-size="11" font-weight="700" fill="#18181b" text-anchor="middle">
    <rect x="482" y="128" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="497" y="142">172</text>
    <rect x="516" y="128" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="531" y="142">88</text>
    <rect x="550" y="128" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="565" y="142">904</text>
    <rect x="584" y="128" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="599" y="142">31</text>
    <text x="650" y="142" fill="#dc2626">[EOL]</text>
    <rect x="482" y="154" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="497" y="168">560</text>
    <rect x="516" y="154" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="531" y="168">12</text>
    <rect x="550" y="154" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="565" y="168">733</text>
    <rect x="584" y="154" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="599" y="168">47</text>
    <text x="650" y="168" fill="#dc2626">[EOL]</text>
    <rect x="482" y="180" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="497" y="194">9</text>
    <rect x="516" y="180" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="531" y="194">421</text>
    <rect x="550" y="180" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="565" y="194">66</text>
    <rect x="584" y="180" width="30" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="1"/><text x="599" y="194">300</text>
    <text x="650" y="194" fill="#dc2626">[EOL]</text>
  </g>
  <text x="525" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">[EOF]</text>
  <text x="618" y="232" text-anchor="middle" font-size="11" fill="#155e63">フレーム区切り</text>
  <text x="589" y="256" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">圧縮率：空間 8×8 ・ 時間 4×（=4×8×8）</text>
</svg>
<figcaption>エンコーダで特徴を抽出し、コードブック内の最近傍ベクトルのID列に量子化する。行末に [EOL]、フレーム末に [EOF] を挿入して 2 次元・時系列構造を保持する。<b>要点</b>＝平坦なトークン列と、画像・動画の幾何構造を相互変換できる。</figcaption>
</figure>

視覚トークナイザは **SBER-MoVQGAN** をベースに学習され、次の特徴を持つ。

- **コードブックサイズは 32,768**。512×512 の画像を **4096 個**の離散トークンに、4×512×512 の動画クリップも同様に離散トークン列へ符号化する。
- 圧縮率は**空間方向 8×8、時間方向 4×**（まとめて 4×8×8）。動画に対応するため、エンコーダ／デコーダに 3D 畳み込みカーネルを持つ時間残差層を追加している。
- 学習は L2 損失・LPIPS 知覚損失・GAN 損失・commitment 損失の組み合わせで end-to-end に行う。

平坦な 1 次元トークン列から元の 2 次元画像・時系列動画に戻せるよう、視覚トークンの中に**行区切り [EOL]** と**フレーム区切り [EOF]** という特殊トークンを差し込む。これにより、トランスフォーマにとっては単なる系列でありながら、脱トークン化のときに正しい解像度・フレーム構成へ復元できる。

事前学習では、テキストと視覚をひとつの「文書」にまとめた系列フォーマットを定義する（定性的に示す）。

```
[BOS] {キャプション} [SOV] {メタ情報} [SOT] {視覚トークン} [EOV] [EOS]
```

- [SOV]／[SOT]／[EOV] が視覚部分の「開始・本体開始・終了」を示す。
- **メタ情報**には、画像なら解像度、動画なら解像度・フレームレート・長さがプレーンテキストで入る。
- ここが重要なところで、拡散モデルのように外部テキストエンコーダを使って条件を注入するのではなく、**条件もすべて同じトークン列の中に文字として書く**。テキスト条件付けが「特別な仕組み」ではなく、ただの前置きトークンになる。

## 次トークン予測だけで理解も生成も

すべてがトークンになった後の学習目標は、**標準のクロスエントロピーによる次トークン予測のみ**である。拡散の denoising も、CLIP の対照学習も、対の損失設計も存在しない。視覚トークンの数が膨大で系列を支配しないよう、視覚トークンに対する損失には重み 0.5 を掛ける、という調整だけが入る。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="同じ次トークン予測で、トークンの並べ方を変えるだけで生成と理解を切り替える様子" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="m3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#16a34a"/></marker>
  </defs>
  <text x="20" y="40" font-size="14" font-weight="700" fill="#3730a3">生成（text → image / video）：条件テキストを先に置き、視覚トークンを予測</text>
  <g font-size="12" font-weight="700" text-anchor="middle">
    <rect x="20" y="58" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="43" y="83" fill="#3730a3">[BOS]</text>
    <rect x="72" y="58" width="118" height="40" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="131" y="83" fill="#18181b">キャプション(条件)</text>
    <rect x="196" y="58" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="219" y="83" fill="#3730a3">[SOV]</text>
    <rect x="248" y="58" width="50" height="40" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="273" y="83" fill="#18181b">メタ</text>
    <rect x="304" y="58" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="327" y="83" fill="#3730a3">[SOT]</text>
    <rect x="356" y="58" width="178" height="40" rx="6" fill="#cffafe" stroke="#16a34a" stroke-width="3"/><text x="445" y="83" fill="#166534">視覚トークン（予測対象）</text>
    <rect x="540" y="58" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="563" y="83" fill="#3730a3">[EOV]</text>
    <rect x="592" y="58" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="615" y="83" fill="#3730a3">[EOS]</text>
  </g>
  <line x1="356" y1="116" x2="534" y2="116" stroke="#16a34a" stroke-width="2" marker-end="url(#m3)"/>
  <text x="445" y="132" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">左から順に視覚トークンを生成 → 脱トークン化で画像・動画へ</text>
  <line x1="20" y1="170" x2="700" y2="170" stroke="#e4e4e7" stroke-width="2"/>
  <text x="20" y="208" font-size="14" font-weight="700" fill="#155e63">理解（VQA・キャプション）：画像を文脈にして、答えのテキストを予測</text>
  <g font-size="12" font-weight="700" text-anchor="middle">
    <rect x="20" y="226" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="43" y="251" fill="#3730a3">[BOS]</text>
    <rect x="72" y="226" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="95" y="251" fill="#3730a3">[SOV]</text>
    <rect x="124" y="226" width="50" height="40" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="149" y="251" fill="#18181b">メタ</text>
    <rect x="180" y="226" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="203" y="251" fill="#3730a3">[SOT]</text>
    <rect x="232" y="226" width="160" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="312" y="251" fill="#155e63">視覚トークン（文脈）</text>
    <rect x="398" y="226" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="421" y="251" fill="#3730a3">[EOV]</text>
    <rect x="450" y="226" width="150" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/><text x="525" y="251" fill="#166534">答えテキスト（予測対象）</text>
    <rect x="606" y="226" width="46" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="629" y="251" fill="#3730a3">[EOS]</text>
  </g>
  <line x1="450" y1="284" x2="600" y2="284" stroke="#16a34a" stroke-width="2" marker-end="url(#m3)"/>
  <text x="525" y="300" text-anchor="middle" font-size="12" font-weight="700" fill="#16a34a">画像トークンを文脈として回答を生成</text>
  <text x="360" y="338" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">どちらも同じモデル・同じ損失。違うのはトークンの並べ方だけ。</text>
</svg>
<figcaption>生成は「テキスト条件 → 視覚トークン」、理解は「視覚トークン → 答えテキスト」。並べ替えるだけで予測対象が切り替わる。<b>要点</b>＝タスクの違いがデータ整形の違いに吸収され、アーキテクチャは一つで済む。</figcaption>
</figure>

生成と理解の切り替えは、**トークンの並べ方だけ**で決まる。

- **生成（text-to-image / text-to-video）**: テキスト条件を前に置き、続く視覚トークンを予測させる。予測した視覚トークンを脱トークン化すれば画像・動画になる。
- **理解（VQA・キャプション）**: 答えのテキストを視覚トークンの後ろ、[EOV] の後に移す。画像トークンを文脈として、答えのテキストを予測させる。

同じ重み・同じ目的関数のまま、入力の順序を変えるだけで双方向をこなす点が美しい。動画も「次フレームのトークンを因果的に予測する」だけなので、与えた動画の続きを生成する**未来予測（video extension）**も同じ仕組みで自然に実現できる。Sora のような動画拡散モデルがノイズから映像を生むのに対し、Emu3 は文章を書き継ぐように映像を 1 トークンずつ書き継ぐ。

論文では、**生成で SDXL を、理解で LLaVA-1.6 を、動画で OpenSora を上回る**と報告されている（具体の優劣はベンチマーク次第なので、ここでは定性的に押さえる）。専用設計に頼らずとも、単一の next-token-prediction モデルが task-specific モデルに匹敵・凌駕しうる、というのが Emu3 の中心的な実証である。

なお後段の調整（post-training）も、追加の損失設計ではなく**同じ次トークン予測の枠内**で行われる。生成側は高品質データでの品質ファインチューニング（QFT。学習解像度を 512 から 720 へ引き上げ）と DPO による人間嗜好への整合、理解側は image-to-text 学習と instruction tuning の 2 段、という構成である。

## Chameleon との位置づけ

Emu3 と Chameleon（24章）は、ともに「**離散トークン＋次トークン予測＋早期融合**」という同じ系譜にある。だからこそ、何が同じで何が違うのかを明確にしておきたい。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="拡散系・CLIP併用系と、Chameleonおよびemu3の離散トークン早期融合方式の比較" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="m4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="16" y="30" font-size="13" font-weight="700" fill="#dc2626">拡散系（SDXL 等）：外部テキストエンコーダ ＋ 反復ノイズ除去の生成器</text>
  <g font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">
    <rect x="16" y="42" width="70" height="44" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="51" y="69">テキスト</text>
    <rect x="118" y="42" width="138" height="44" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/><text x="187" y="69">外部テキストエンコーダ</text>
    <rect x="288" y="42" width="150" height="44" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/><text x="363" y="69">拡散UNet（反復除去）</text>
    <rect x="470" y="42" width="70" height="44" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="505" y="69">画像</text>
  </g>
  <line x1="86" y1="64" x2="116" y2="64" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="256" y1="64" x2="286" y2="64" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="438" y1="64" x2="468" y2="64" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <text x="16" y="120" font-size="13" font-weight="700" fill="#0e7490">CLIP併用系（LLaVA 等）：CLIPエンコーダ ＋ LLM、理解専用で生成は別物</text>
  <g font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">
    <rect x="16" y="132" width="70" height="44" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="51" y="159">画像</text>
    <rect x="118" y="132" width="128" height="44" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/><text x="182" y="159">CLIPエンコーダ</text>
    <rect x="278" y="132" width="90" height="44" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="323" y="159">LLM</text>
    <rect x="400" y="132" width="120" height="44" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="460" y="159">テキスト答え</text>
  </g>
  <line x1="86" y1="154" x2="116" y2="154" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="246" y1="154" x2="276" y2="154" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="368" y1="154" x2="398" y2="154" stroke="#71717a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="16" y1="196" x2="704" y2="196" stroke="#e4e4e7" stroke-width="2"/>
  <text x="16" y="222" font-size="13" font-weight="700" fill="#166534">離散トークンの早期融合（Chameleon / Emu3）：1つのトランスフォーマで完結</text>
  <g font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">
    <rect x="16" y="236" width="150" height="70" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="91" y="266"><tspan x="91">入力</tspan><tspan x="91" dy="18">テキスト/画像/動画</tspan></text>
    <rect x="200" y="236" width="240" height="70" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/><text x="320" y="266" fill="#166534"><tspan x="320">単一トランスフォーマ</tspan><tspan x="320" dy="18">次トークン予測・early-fusion</tspan></text>
    <rect x="474" y="236" width="150" height="70" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="549" y="266"><tspan x="549">出力</tspan><tspan x="549" dy="18">画像/動画/テキスト</tspan></text>
  </g>
  <line x1="166" y1="271" x2="198" y2="271" stroke="#16a34a" stroke-width="2" marker-end="url(#m4)"/>
  <line x1="440" y1="271" x2="472" y2="271" stroke="#16a34a" stroke-width="2" marker-end="url(#m4)"/>
  <text x="638" y="262" font-size="11" font-weight="700" fill="#166534" text-anchor="middle"><tspan x="638">Emu3 は</tspan><tspan x="638" dy="16">＋動画</tspan><tspan x="638" dy="16">＋生成と理解</tspan><tspan x="638" dy="16">の統一</tspan></text>
  <text x="320" y="332" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">拡散も CLIP も外付けしない／encoder-free・decoder-only</text>
</svg>
<figcaption>拡散系は外部エンコーダと拡散生成器を、CLIP併用系は CLIP と LLM を組み合わせる。Chameleon と Emu3 は離散トークンの早期融合で 1 モデルに統合する。<b>要点</b>＝Emu3 は動画と「生成・理解の双方向」まで同じ枠に取り込む。</figcaption>
</figure>

**同じところ**: どちらも視覚を離散トークンに落とし、テキストと混ぜて 1 つのトランスフォーマで次トークン予測する early-fusion。理解と生成を別々のモデルに分けない発想を共有する。

**Emu3 が踏み込んだところ**:

1. **動画を加えた**。Chameleon が主に画像とテキストの混在を扱うのに対し、Emu3 は動画クリップも離散トークン化して同じ系列に乗せ、動画生成・未来予測まで一つのモデルで行う。
2. **生成と理解の両方で task-specific モデルを上回ることを狙った**。「統一できる」だけでなく、SDXL（生成）や LLaVA-1.6（理解）のような専用モデルに匹敵・凌駕することを実証目標に据えた。
3. **拡散・CLIP・合成的設計を完全に排した**。decoder-only かつ encoder-free を貫き、外付け部品をゼロにする。理解においても、CLIP のような事前学習済みエンコーダや専用 LLM に依存しない「encoder-free」な手法として位置づけられる。

系譜の中では、前身の Emu / Emu2 が視覚埋め込みを回帰する形だったのに対し、Emu3 は**完全に離散トークンの分類**（＝言語と同一の next-token prediction）へ統一した点が転換である。また、拡散と自己回帰を混ぜる TransFusion や Show-o とも異なり、Emu3 は両者を混ぜずに「純粋な次トークン予測だけ」で押し切ろうとする。Chameleon が拓いた早期融合の道を、動画と双方向タスクまで拡張したのが Emu3、と捉えるとよい。

## まとめと、読解後に答えたい問い

**まとめ**:

- Emu3 は、画像・動画・テキストをすべて離散トークン化し、**ゼロから学習した単一のトランスフォーマで次トークン予測 $p(x)=\prod_i p(x_i\mid x_{<i})$ だけ**を行う。
- 視覚トークナイザ（MoVQGAN 系、コードブック 32,768）が、512×512 の画像を 4096 トークンへ、動画を時間方向も含めて離散化する。[EOL]／[EOF] で幾何・時系列構造を保つ。
- **拡散も CLIP も合成的設計も使わない**。生成と理解の違いは、トークンの並べ方（条件を前に置くか、答えを後ろに置くか）に吸収される。
- 専用設計に頼らずとも、生成（対 SDXL）・理解（対 LLaVA-1.6）・動画（対 OpenSora）で競争力を示し、「次トークン予測こそすべて」という主張を裏づけた。
- Chameleon と同じ離散トークン早期融合の系譜にありつつ、**動画**と**生成・理解の双方向統一**まで取り込んだのが Emu3 の立ち位置である。

**読解後に答えたい問い**:

1. 視覚トークナイザの**コードブックサイズや圧縮率**を変えると、再構成品質と系列長（＝計算コスト）はどうトレードオフするか。なぜ Emu3 はこの設定を選んだのか。
2. 視覚トークンの損失に**重み 0.5** を掛けるのはなぜか。重みを変えると理解・生成のバランスはどう動くと予想されるか。
3. テキスト条件を「外部エンコーダ」ではなく「同じ系列内のトークン」として与えることの利点と弱点は何か。長い・細かいプロンプトに対してどう効くか。
4. 拡散モデルが得意とする高解像度・高忠実度の生成に対し、離散トークンの自己回帰生成が抱える原理的な難しさ（量子化誤差、系列長、サンプリング順序など）はどこにあるか。
5. Chameleon と比べて Emu3 が**動画**を扱えることは、単なるモダリティ追加か、それとも「次トークン予測で世界を予測する」という方向性にとって本質的か。
