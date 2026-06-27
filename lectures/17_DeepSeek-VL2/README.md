# DeepSeek-VL2 — MoE Vision-Language Models（動的タイリング＋MoE）

DeepSeek-VL2（*Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding*, arXiv:2412.10302, DeepSeek-AI 2024）は、前章 **DeepSeek-VL** の後継です。基本は LLaVA 系（視覚エンコーダ → VL アダプタ → 言語モデル）を踏襲しつつ、2 つの大きな更新で**性能と効率を同時に**押し上げました。ひとつは高解像・任意アスペクト比に対応する **動的タイリング（dynamic tiling）** の視覚符号化、もうひとつは **MLA（Multi-head Latent Attention）** を備えた **DeepSeekMoE** 言語モデルです。狙いは一貫して「**少ない活性パラメータで、高解像・高性能**」。モデルは Tiny / Small / フルの 3 規模で展開されます。本ページは、前章で「固定解像度の二重視覚エンコーダ＋密な LLM」を学んだ読者が、**密 → MoE**・**固定解像度 → 動的タイリング**という 2 軸の転換を腹落ちさせることを狙います。

## 全体像（まず一枚で）

DeepSeek-VL2 は 3 つの中核モジュールからなります ―― (1) **視覚エンコーダ**（単一の SigLIP-SO400M-384）、(2) **VL アダプタ**（2 層 MLP）、(3) **MoE 言語モデル**（DeepSeekMoE＋MLA）。高解像画像はまず動的タイリングで複数タイルに分割され、共有の ViT で符号化、VL アダプタで言語空間へ写像され、テキスト指示とともに MoE 言語モデルへ流れて自己回帰で生成されます。「箱の並び」は LLaVA 系のままで、新しさは**視覚側のタイリング**と**言語側の MoE＋MLA**という 2 つの差分にあります。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="DeepSeek-VL2 の全体像。入力の高解像画像を動的タイリングでタイルとサムネイルに分割し、共有SigLIP-SO400M ViTで符号化、2層MLPのVLアダプタで視覚トークンへ写像し、テキスト指示とともにDeepSeekMoE言語モデル（MLA付き）へ入力して自己回帰でテキストを生成する。MoEは一部の専門家のみ活性化し、活性パラメータは総パラメータより遥かに小さい" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="360" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">LLaVA 系の 3 部品：視覚エンコーダ → VL アダプタ → MoE 言語モデル</text><rect x="20" y="56" width="104" height="60" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="72" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">入力画像</text><text x="72" y="101" text-anchor="middle" font-size="10" fill="#0e7490">高解像・任意比</text><line x1="124" y1="86" x2="146" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="154,86 143,81 143,91" fill="#71717a"/><rect x="150" y="56" width="132" height="60" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="216" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">動的タイリング</text><text x="216" y="101" text-anchor="middle" font-size="10" fill="#0e7490">タイル＋サムネイル</text><line x1="282" y1="86" x2="304" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="312,86 301,81 301,91" fill="#71717a"/><rect x="308" y="56" width="150" height="60" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="383" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">SigLIP-SO400M</text><text x="383" y="101" text-anchor="middle" font-size="10" fill="#4338ca">共有 ViT（384²）</text><line x1="458" y1="86" x2="480" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="488,86 477,81 477,91" fill="#71717a"/><rect x="484" y="56" width="216" height="60" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="592" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">VL アダプタ（2 層 MLP）</text><text x="592" y="101" text-anchor="middle" font-size="10" fill="#4338ca">2×2 pixel shuffle → 視覚トークン</text><line x1="592" y1="116" x2="592" y2="202" stroke="#71717a" stroke-width="2"/><polygon points="592,210 587,199 597,199" fill="#71717a"/><rect x="150" y="212" width="470" height="82" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="385" y="246" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">DeepSeekMoE 言語モデル（MLA 付き）</text><text x="385" y="270" text-anchor="middle" font-size="11" fill="#4338ca">一部の専門家のみ活性／KV を潜在ベクトルに圧縮</text><rect x="20" y="228" width="110" height="50" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="75" y="258" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">テキスト指示</text><line x1="130" y1="253" x2="146" y2="253" stroke="#71717a" stroke-width="2"/><polygon points="150,253 141,248 141,258" fill="#71717a"/><rect x="636" y="226" width="64" height="54" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="668" y="249" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">テキスト</text><text x="668" y="265" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">生成</text><line x1="620" y1="253" x2="628" y2="253" stroke="#71717a" stroke-width="2"/><polygon points="636,253 627,248 627,258" fill="#71717a"/><text x="385" y="332" text-anchor="middle" font-size="11" fill="#71717a">活性パラメータ ≪ 総パラメータ（全体活性：Tiny 1.0B ／ Small 2.8B ／ VL2 4.5B）</text></svg><figcaption>構成は LLaVA 系の 3 部品のまま。差分は <b>視覚側の動的タイリング</b> と <b>言語側の MoE＋MLA</b> の 2 つに集約されます。</figcaption></figure>

前章 DeepSeek-VL からの設計差を整理すると、視覚側と言語側でそれぞれ転換が起きていることが分かります。

| 観点 | DeepSeek-VL（前章） | DeepSeek-VL2 |
|---|---|---|
| 視覚エンコーダ | ハイブリッド：SigLIP（384²）＋ SAM-B（1024²）の二重構成、固定解像度 | **単一 SigLIP-SO400M-384 ＋ 動的タイリング** |
| 解像度・アスペクト比 | 固定（最大 1024²） | **任意比・高解像**（候補解像度から余白最小を選択） |
| 言語モデル | 密（dense）LLM | **DeepSeekMoE（疎活性）** |
| Attention | 通常の MHA | **MLA**（KV を潜在ベクトルに圧縮、rank=512）※Tiny のみ MHA |
| 規模展開 | 1.3B / 7B 級の密モデル | **Tiny / Small / VL2**（全体活性 1.0B / 2.8B / 4.5B） |

スケールは 3 種で、いずれも視覚エンコーダは共通の SigLIP-SO400M（約 0.4B）です。MoE では**総パラメータ**と**活性パラメータ**（1 トークンあたり実際に計算される量）を区別するのが要点で、活性は総より遥かに小さく抑えられています（PDF Table 2）。

| モデル | 総パラメータ（LLM） | 活性パラメータ（LLM） | 全体の活性 | 学習規模 |
|---|---|---|---|---|
| DeepSeek-VL2-Tiny | 3B | 0.57B | 1.0B | 約 7 日 / 16 ノード |
| DeepSeek-VL2-Small | 16B | 2.4B | 2.8B | 約 10 日 / 33 ノード |
| DeepSeek-VL2 | 27B | 4.1B | 4.5B | 約 14 日 / 42 ノード |

> 各ノードは 8×NVIDIA A100。基盤の言語モデルは DeepSeekMoE 3B/16B/27B のベースモデルで、これに視覚エンコーダと VL アダプタを接続して学習します。

DeepSeek-VL2 を 1 枚で捉えるなら、各モジュールの責務はこう分かれます。

- **視覚エンコーダ（SigLIP-SO400M）**：動的タイリングで作ったタイル群を 1 枚ずつ符号化する。全タイルで重みを共有するため、タイル数が増えても新たなパラメータは要らない。
- **VL アダプタ（2 層 MLP）**：視覚埋め込みを言語モデルの埋め込み空間へ写像する「橋」。pixel shuffle によるトークン圧縮もここで担う。
- **MoE 言語モデル（DeepSeekMoE＋MLA）**：視覚トークンとテキストトークンを 1 本の系列として受け取り、自己回帰でテキストを生成する本体。

学習はどの段でも**次トークン予測の損失をテキストトークンにのみ**課します。視覚トークンは予測対象ではなく「条件付けの文脈」であり、勾配はテキスト生成に役立つ形で視覚情報を取り込ませる役割に徹します。この「視覚は文脈・損失はテキスト」という原則は LLaVA や前章 DeepSeek-VL、InternVL 系と共通で、DeepSeek-VL2 の新規性はあくまで視覚側のタイリングと言語側の MoE＋MLA にあります。

位置づけを整理しておくと、DeepSeek-VL2 は 2 つの系譜の合流点にあります。視覚側のタイル分割は、InternVL や Qwen-VL 系が採る **AnyRes（動的解像度）** の流れにあり、本章はその DeepSeek 版です。言語側の DeepSeekMoE＋MLA は、DeepSeek の**言語モデル系列**（MoE と KV 圧縮で効率を追う流儀）を視覚言語モデルへ持ち込んだものです。つまり「高解像の視覚処理」と「効率的な MoE 言語モデル」という、別々に発展してきた 2 つの工夫を 1 つの VLM に束ねたのが DeepSeek-VL2 だ、と捉えると見通しが良くなります。

## 動的タイリング（高解像度の視覚処理）

前章 DeepSeek-VL の視覚側は、SigLIP（粗い特徴・384²）と SAM-B（細かい特徴・1024²）を組み合わせたハイブリッドでしたが、**固定解像度 1024²** という上限が、極端なアスペクト比や超高解像（InfographicVQA・密な OCR・細かい視覚グラウンディング）で足かせになっていました。DeepSeek-VL2 はこれを捨て、**単一の SigLIP-SO400M-384** を使いながら、入力をタイルに割って高解像を扱う **動的タイリング** に切り替えます。AnyRes 系（スライス＝タイル方式）の発想で、ViT の局所注意の良さを保ちつつ、解像度に伴う計算量の二乗スケーリングを避けられます。

手順はこうです。まず候補解像度の集合を用意します。

$$C_R = \{(m\cdot 384,\ n\cdot 384) \mid m,n\in\mathbb{N},\ 1\le m,n,\ mn\le 9\}$$

入力サイズ $(H, W)$ に対して、各候補へリサイズしたときに必要な**余白（padding）面積**を計算し、**余白が最小**になる解像度 $(m_i\cdot 384,\ n_i\cdot 384)$ を選びます（長辺を目標に合わせ、短辺をアスペクト比保持のままパディング）。選んだ解像度の画像を $m_i\times n_i$ 枚の **ローカルタイル**（各 384²）に分割し、さらに**画像全体を 1 枚に縮小した全体サムネイル**を加えます。共有 ViT は計 $(1 + m_i\times n_i)$ 枚のタイルを処理し、各タイルから $27\times 27 = 729$ 個・$1152$ 次元の視覚埋め込みを得ます。

前章のハイブリッド構成（SigLIP＋SAM-B）を手放せたのは、この**タイル化が高解像の役目を肩代わり**するからです。高解像を「大きな入力を一発で処理する」のではなく「小さな 384² タイルを複数並べる」方式に置き換えたため、細部の解像度はタイル数で稼げます。エンコーダは 1 つで済み、全タイルが重みを共有するので、解像度を上げてもパラメータは増えません。

具体例で見ましょう。横長の入力なら $3\times 2$ のように横に多いタイル配置が余白最小になり、縦長なら逆に縦長配置が選ばれます。文字が密な書類のスキャンのように縦横が極端に大きい画像でも、$mn\le 9$ の範囲でアスペクト比に合うタイル数が選ばれ、長辺を解像度に合わせて細部を保てます。固定 1024² では潰れていた細かな文字や罫線も、タイル単位で 384² の実効解像度を確保できるのが利点です。なお論文では、極端なアスペクト比と巨大画像を含む InfographicVQA の評価時に限り、候補解像度を $mn\le 18$ まで広げています（PDF 5 節脚注）。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="動的タイリングの流れ。高解像で任意アスペクト比の入力画像を、余白が最小になる候補解像度に合わせて複数のローカルタイル（各384の2乗）に分割し、さらに画像全体を縮小した全体サムネイルを1枚加える。各タイルを共有のSigLIP ViTで符号化して27かける27で729個の埋め込みを得たのち、2かける2のpixel shuffleで14かける14の196トークンに圧縮し、view_separatorとtile_newlineの区切りトークンを挿入して言語モデルへ渡す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="360" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">高解像画像を「ローカルタイル＋全体サムネイル」に分割して符号化</text><rect x="24" y="74" width="140" height="190" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="94" y="58" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">入力画像（高解像・任意比）</text><line x1="164" y1="168" x2="200" y2="168" stroke="#71717a" stroke-width="2"/><polygon points="208,168 197,163 197,173" fill="#71717a"/><text x="186" y="150" text-anchor="middle" font-size="10" fill="#71717a">余白最小の</text><text x="186" y="162" text-anchor="middle" font-size="10" fill="#71717a">解像度を選択</text><rect x="214" y="74" width="86" height="62" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="257" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">全体</text><text x="257" y="117" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">サムネイル</text><text x="257" y="152" text-anchor="middle" font-size="10" fill="#6366f1">全体を 1 枚に縮小</text><rect x="214" y="170" width="42" height="42" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><rect x="258" y="170" width="42" height="42" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><rect x="214" y="214" width="42" height="42" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><rect x="258" y="214" width="42" height="42" rx="4" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="257" y="284" text-anchor="middle" font-size="10" fill="#0e7490">ローカルタイル m×n（各 384²）</text><line x1="306" y1="168" x2="342" y2="168" stroke="#71717a" stroke-width="2"/><polygon points="350,168 339,163 339,173" fill="#71717a"/><rect x="354" y="86" width="200" height="40" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="454" y="111" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">共有 ViT で各タイルを符号化</text><line x1="454" y1="126" x2="454" y2="142" stroke="#71717a" stroke-width="2"/><polygon points="454,150 449,139 459,139" fill="#71717a"/><rect x="354" y="150" width="200" height="40" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="454" y="175" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">27×27 = 729 埋め込み / 1152 次元</text><line x1="454" y1="190" x2="454" y2="206" stroke="#71717a" stroke-width="2"/><polygon points="454,214 449,203 459,203" fill="#71717a"/><rect x="354" y="214" width="200" height="40" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="454" y="239" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">2×2 pixel shuffle → 14×14 = 196</text><line x1="554" y1="234" x2="586" y2="234" stroke="#71717a" stroke-width="2"/><polygon points="594,234 583,229 583,239" fill="#71717a"/><rect x="598" y="206" width="104" height="56" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="650" y="228" text-anchor="middle" font-size="10" font-weight="700" fill="#18181b">区切りトークン挿入</text><text x="650" y="244" text-anchor="middle" font-size="9" fill="#71717a">view_separator / tile_newline</text><text x="360" y="340" text-anchor="middle" font-size="11" fill="#71717a">3 枚を超える複数画像の入力時は動的タイリングを無効化（文脈長と効率のため）</text></svg><figcaption>高解像画像を <b>余白最小の候補解像度</b> でローカルタイルに割り、<b>全体サムネイル</b>を 1 枚添えて共有 ViT で符号化。pixel shuffle で 1 タイル <b>196 トークン</b> に圧縮します。</figcaption></figure>

タイルあたり 729 トークンは多すぎるため、VL アダプタは **$2\times 2$ pixel shuffle** で各タイルを $27\times 27 \to 14\times 14 = 196$ トークンに圧縮します。pixel shuffle は隣り合う $2\times 2$ の空間パッチをチャネル方向に畳み込む操作で、空間解像度を半分にする代わりにチャネル次元を増やし、**トークン数を $1/4$** にします。情報を捨てずに系列長だけ縮められるため、タイルを増やしても言語モデル側の負荷を抑えられます。

さらにレイアウトを言語モデルへ伝えるため、特別トークンを差し込みます。全体サムネイル（$14\times 14$）には各行末に `tile_newline` を足して $14\times 15 = 210$ トークンとし、ローカルタイル群（$m_i\cdot 14$ 行 × $n_i\cdot 14$ 列の 2 次元配置）にも行末改行を入れ、サムネイルとローカルタイルの境目に `view_separator` を 1 つ挿入します。最終的な視覚トークン列の長さは次の通りです。

$$210\ +\ 1\ +\ m_i\cdot 14\cdot(n_i\cdot 14 + 1)$$

この列を 2 層 MLP で言語モデルの埋め込み空間へ射影します。視覚トークン列の中身を分解すると、レイアウト情報がトークン列にどう焼き込まれるかが見えます。

- **全体サムネイル**：$14\times 14$ の本体に各行末の `tile_newline` を足して $14\times 15 = 210$ トークン。まず画像の「全体像」を低解像で先に与える。
- **`view_separator`**：サムネイルとローカルタイル群の境目に 1 つ。粗い全体と細かい局所の切り替わりを明示する。
- **ローカルタイル群**：$m_i\cdot 14$ 行 × $n_i\cdot 14$ 列の 2 次元配置に、各行の終端を示す改行トークンを加える。タイルの並び順＝画像内の空間配置が保たれる。

このようにトークン列そのものに行・タイルの境界を埋め込むことで、1 次元の系列しか受け取らない言語モデルにも 2 次元の空間レイアウトが伝わります。なお、画像が**3 枚以上**入る場合は、文脈長と計算効率の管理のために動的タイリングを無効化します。動的タイリングの効きどころ（PDF 2 節）を整理します。

- **任意アスペクト比・高解像**に、固定解像度エンコーダ 1 つで対応できる
- ViT の**局所注意**を保つため、解像度に対する計算量の二乗爆発を避けられる
- タイル化＋pixel shuffle で**視覚トークン数を管理可能**に保つ（1 タイル 196 トークン）
- OCR・文書/表/図表理解・視覚グラウンディングなど**細部が効くタスク**に強い

## MoE 言語モデル（MLA 付き）

言語側は DeepSeek の **DeepSeekMoE** をベースに、**MLA（Multi-head Latent Attention）** を組み込んだ構成です。MoE（Mixture-of-Experts）の肝は、フィードフォワード層を多数の小さな**専門家（expert）**に分け、入力トークンごとに **Router（ゲーティング）が一部の専門家だけを起動**する点にあります。すべての専門家を毎回計算しないので、**総パラメータは大きいまま、1 トークンが実際に通る（＝活性）パラメータは小さく**できます。DeepSeekMoE はさらに、常に起動する少数の **共有専門家（shared experts）** と、Top-K で選ばれる **ルーティング専門家（routed experts）** を分けるのが特徴です。

DeepSeekMoE の「専門家の分け方」には 2 つの工夫があります。ひとつは**細粒度の専門家分割（fine-grained segmentation）**で、FFN を多数の小さな専門家に細かく割ることで、トークンごとの専門家の組み合わせ自由度を上げます。もうひとつは**共有専門家の分離（shared expert isolation）**で、どのトークンも共通して必要とする知識を共有専門家にまとめ、ルーティング専門家には「その入力ならでは」の処理を任せます。これにより、Top-K で選ぶ少数の専門家に専門性が宿りやすくなります。

活性パラメータの感覚を数値で掴みましょう。フルモデルは 72 個のルーティング専門家から **Top-6** を選び、加えて常時活性の共有専門家 2 個を使います。1 トークンが通るのは実質「6＋2 個ぶんの FFN ＋ 注意層」であり、総 27B のうち活性は約 4.1B ―― つまり毎ステップ**総量の 1/6 ほど**しか計算しません。容量（知識の貯蔵）は総パラメータが担い、推論コストは活性パラメータが決める、という分業が成立します。

<figure class="lec-fig"><svg viewBox="0 0 720 380" role="img" aria-label="MoE言語モデルのルーティング。入力トークンをRouter（ゲーティング）が受け取り、72個のルーティング専門家のうちTop-6だけを活性化する。図では代表として3つの専門家が緑で活性、ほかは灰色で休止して計算されない。加えて2個の共有専門家は常に活性。活性した専門家のみを計算して重み付き合成し次層へ渡す。活性パラメータは総パラメータより遥かに小さく、フルモデルでおよそ4.1Bと27Bの関係になる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="360" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">Router がトークンごとに「一部の専門家だけ」を起動する</text><rect x="20" y="168" width="118" height="64" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="79" y="205" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">入力トークン</text><line x1="138" y1="200" x2="156" y2="200" stroke="#71717a" stroke-width="2"/><polygon points="164,200 153,195 153,205" fill="#71717a"/><rect x="166" y="160" width="120" height="80" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="226" y="194" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">Router</text><text x="226" y="214" text-anchor="middle" font-size="11" fill="#4338ca">Top-6 を選択</text><text x="430" y="50" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">Routed Experts ×72（Top-6 のみ活性）</text><rect x="355" y="62" width="160" height="30" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="435" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">Expert 7 ◀ 活性</text><rect x="355" y="98" width="160" height="30" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="435" y="118" text-anchor="middle" font-size="11" fill="#71717a">休止（計算しない）</text><rect x="355" y="134" width="160" height="30" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="435" y="154" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">Expert 23 ◀ 活性</text><rect x="355" y="170" width="160" height="30" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="435" y="190" text-anchor="middle" font-size="11" fill="#71717a">休止（計算しない）</text><rect x="355" y="206" width="160" height="30" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="435" y="226" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">Expert 58 ◀ 活性</text><rect x="355" y="242" width="160" height="30" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="435" y="262" text-anchor="middle" font-size="11" fill="#71717a">休止（計算しない）</text><rect x="355" y="286" width="160" height="34" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="435" y="308" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">Shared Experts ×2（常時活性）</text><line x1="286" y1="190" x2="351" y2="77" stroke="#16a34a" stroke-width="2"/><polygon points="355,77 343,77 349,87" fill="#16a34a"/><line x1="286" y1="196" x2="351" y2="149" stroke="#16a34a" stroke-width="2"/><polygon points="355,149 344,144 344,154" fill="#16a34a"/><line x1="286" y1="202" x2="351" y2="221" stroke="#16a34a" stroke-width="2"/><polygon points="355,221 344,216 344,226" fill="#16a34a"/><line x1="286" y1="208" x2="351" y2="113" stroke="#a1a1aa" stroke-width="1.2" stroke-dasharray="4 3"/><line x1="286" y1="212" x2="351" y2="185" stroke="#a1a1aa" stroke-width="1.2" stroke-dasharray="4 3"/><line x1="286" y1="216" x2="351" y2="257" stroke="#a1a1aa" stroke-width="1.2" stroke-dasharray="4 3"/><rect x="545" y="150" width="150" height="80" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="620" y="184" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">活性した専門家のみ</text><text x="620" y="204" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">計算 → 合成 → 次層</text><line x1="515" y1="77" x2="543" y2="165" stroke="#16a34a" stroke-width="2"/><polygon points="545,170 533,166 540,158" fill="#16a34a"/><line x1="515" y1="149" x2="543" y2="180" stroke="#16a34a" stroke-width="2"/><polygon points="545,184 534,181 538,172" fill="#16a34a"/><line x1="515" y1="221" x2="543" y2="200" stroke="#16a34a" stroke-width="2"/><polygon points="545,196 534,196 539,205" fill="#16a34a"/><line x1="515" y1="303" x2="543" y2="222" stroke="#6366f1" stroke-width="1.6" stroke-dasharray="4 3"/><text x="360" y="362" text-anchor="middle" font-size="11" fill="#71717a">活性パラメータ ≪ 総パラメータ（DeepSeek-VL2：活性 約 4.1B / 総 27B）</text></svg><figcaption>Router は<b>トークンごとに Top-6 の専門家だけ</b>を起動（図は代表 3 つ）。常時活性の<b>共有専門家</b>と合わせて計算し、活性パラメータは総パラメータより遥かに小さく保たれます。</figcaption></figure>

各スケールのアーキテクチャ構成（PDF Table 1）を見ると、Tiny だけが通常の MHA で、Small とフルが MLA を使う点、フルモデルだけ専門家数とルーティングが強化されている点が読み取れます。

| 構成 | Tiny | Small | DeepSeek-VL2 |
|---|---|---|---|
| 語彙数 | 129,280 | 102,400 | 129,280 |
| 埋め込み次元 | 1,280 | 2,048 | 2,560 |
| 層数 | 12 | 27 | 30 |
| ヘッド数 | 10 | 16 | 32 |
| Attention | MHA | MLA（rank=512） | MLA（rank=512） |
| Routed / Shared / Top-K | 64 / 2 / 6 | 64 / 2 / 6 | 72 / 2 / 6 |
| ルーティング関数 | Softmax | Softmax | Sigmoid |
| 専門家バイアス補正 | × | × | ✓ |

スケールが上がるほど機構が「強い側」へ寄っているのが読み取れます。Tiny は容量が小さく、MLA や Sigmoid ルーティング・バイアス補正のような追加機構の旨味が出にくいため、素直な **MHA＋Softmax** に留めます。一方フルモデルは専門家数を 64→72 に増やしつつ、**Sigmoid ルーティング＋専門家バイアス補正**で多数の専門家を均して使い切る設計です。つまり「規模に応じて、効率機構をどこまで盛るか」を変えているわけです。ここで効いている設計を 2 つに分けて押さえます。

**MLA（Multi-head Latent Attention）** は、推論時に層ごとに肥大しがちな **KV キャッシュ**（Key-Value）を、**低ランクの潜在ベクトル（rank=512）に圧縮**してから保持する仕組みです。キャッシュするのは潜在ベクトルだけで、注意計算時に Key・Value を復元します。これにより**メモリ占有が減り、スループットが上がる**ため、動的タイリングで視覚トークンが増えても推論コストを抑えられます。Tiny は容量が小さいため通常の MHA を採用し、Small とフルが MLA を使います。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="MLAによるKVキャッシュ圧縮の対比。上段の従来MHAは層ごとにKeyとValueを丸ごとキャッシュするためメモリが大きく、長文や多タイルでKVが肥大してスループットが低下する。下段のMLAは隠れ状態を下方投影してrank512の小さな潜在ベクトルcに圧縮し、それだけをキャッシュする。注意計算の直前に上方投影でKeyとValueを復元するため、キャッシュは潜在cのみでメモリが小さく高スループットになる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="30" y="40" font-size="13" font-weight="700" fill="#dc2626">従来 MHA：K・V を丸ごとキャッシュ</text><rect x="30" y="54" width="360" height="56" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/><text x="210" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">K キャッシュ ＋ V キャッシュ</text><text x="210" y="98" text-anchor="middle" font-size="11" fill="#dc2626">ヘッド数 × 次元ぶん、メモリ大</text><text x="410" y="78" font-size="11" fill="#71717a">長文・多タイルで</text><text x="410" y="94" font-size="11" fill="#71717a">KV が肥大 → 遅い</text><line x1="30" y1="132" x2="690" y2="132" stroke="#e4e4e7" stroke-width="2"/><text x="30" y="166" font-size="13" font-weight="700" fill="#15803d">MLA：K・V を低ランク潜在ベクトルに圧縮（rank=512）</text><rect x="24" y="184" width="116" height="56" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="82" y="216" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">隠れ状態</text><line x1="140" y1="212" x2="166" y2="212" stroke="#71717a" stroke-width="2"/><polygon points="174,212 163,207 163,217" fill="#71717a"/><rect x="176" y="184" width="116" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="234" y="216" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">下方投影</text><line x1="292" y1="212" x2="318" y2="212" stroke="#71717a" stroke-width="2"/><polygon points="326,212 315,207 315,217" fill="#71717a"/><rect x="328" y="184" width="116" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="386" y="210" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">潜在 c</text><text x="386" y="227" text-anchor="middle" font-size="10" fill="#15803d">rank=512（小）</text><text x="386" y="262" text-anchor="middle" font-size="10" font-weight="700" fill="#16a34a">★ これだけキャッシュ</text><line x1="444" y1="212" x2="470" y2="212" stroke="#71717a" stroke-width="2"/><polygon points="478,212 467,207 467,217" fill="#71717a"/><rect x="480" y="184" width="116" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="538" y="210" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">上方投影</text><text x="538" y="227" text-anchor="middle" font-size="10" fill="#4338ca">K・V を復元</text><line x1="596" y1="212" x2="622" y2="212" stroke="#71717a" stroke-width="2"/><polygon points="630,212 619,207 619,217" fill="#71717a"/><rect x="632" y="184" width="64" height="56" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="664" y="216" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">Attn</text><text x="360" y="312" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">キャッシュは潜在 c のみ → メモリ小・高スループット</text></svg><figcaption>MLA は K・V を <b>rank=512 の潜在ベクトル</b>に圧縮してキャッシュし、注意計算の直前に復元します。長文・多タイルでも KV が肥大せず<b>高スループット</b>です。</figcaption></figure>

**負荷分散（load balancing）** も MoE の重要点です。専門家ごとに利用が偏ると一部だけが学習・計算され、容量が無駄になります。フルモデル（DeepSeek-VL2）は各専門家に **グローバルバイアス項（expert correction bias）** を導入し、補助損失（auxiliary loss）に頼らず**低コストで負荷を均す**設計を採ります（ルーティング関数も Tiny/Small の Softmax からフルでは Sigmoid に変更）。MoE 言語モデルの要点（PDF 2 節・Table 1）をまとめます。

- **疎活性**：Top-K=6 のルーティング専門家＋常時活性の共有専門家のみ計算 → 活性 ≪ 総
- **MLA**：KV を潜在ベクトル（rank=512）に圧縮 → 推論メモリ減・高スループット（Tiny は MHA）
- **負荷分散**：フルモデルはグローバルバイアスで補助損失なしに均す（Sigmoid ルーティング）
- **規模**：総 3B/16B/27B に対し、活性（LLM）は 0.57B/2.4B/4.1B

## 効率と性能

DeepSeek-VL2 の設計思想は「**効率と高性能の両立**」です。効率は主に 2 つの機構から来ます。

- **MoE の疎活性**：1 トークンあたりの計算量（FLOPs）を、活性パラメータ相当まで減らす。
- **MLA の KV 圧縮**：推論時のキャッシュ（メモリ・帯域）を、潜在ベクトル相当まで減らす。

MoE 言語モデルが 1 トークンで実際に計算する割合は、フルモデルで $4.1\text{B}/27\text{B}\approx 0.15$、Small で $2.4\text{B}/16\text{B}=0.15$、Tiny で $0.57\text{B}/3\text{B}\approx 0.19$ ―― つまり**総パラメータの約 2 割以下**しか毎回は通りません。

仮に同じ 27B を**密（dense）モデル**で組むと、どのトークンでも 27B 全部を計算することになります。MoE はそれを約 4.1B に抑えつつ、容量（知識の貯蔵）は 27B ぶん保てる ―― これが「総容量で蓄え、活性コストで動かす」という MoE の旨味です。さらに MLA の KV 圧縮が重なり、動的タイリングで視覚トークンが増えても高スループットを保ちます。

性能面では、動的タイリングが効く**高解像・細部依存タスク**で改善が顕著です。前章 DeepSeek-VL の固定解像度では難しかった、密な OCR・文書/表/図表理解・InfographicVQA・視覚グラウンディング（座標出力）まで守備範囲が広がりました。グラウンディングや GUI 知覚といった新しい能力は、改善された学習データ（座標を特殊トークンで表す grounding 形式や、視覚プロンプト付き QA など）にも支えられています。

学習は 3 段のパイプラインです（PDF 4 節）。

1. **VL アライメント**：視覚エンコーダと VL アダプタ（MLP）を学習、言語モデルは凍結（ShareGPT4V 約 1.2M で warmup）
2. **VL 事前学習**：視覚エンコーダ・アダプタ・MoE LLM の**全パラメータを解凍**して同時学習（約 800B 画像–テキストトークン、VL : テキスト ≒ 7 : 3）
3. **教師ありファインチューニング（SFT）**：高品質な社内 VL データ＋純テキスト対話で指示追従・対話を仕上げる

いずれの段でも**損失は次トークン予測をテキストトークンにのみ**課し、視覚トークンは予測対象ではなく条件付けの文脈として働きます（前章・InternVL 系と同じ思想）。

学習データも前章から拡張されています（PDF 3 節）。各段で使うデータの性格が異なる点を押さえておきます。

- **アライメント**：ShareGPT4V 由来の約 1.2M のキャプション・対話で、MLP コネクタを温める。
- **VL 事前学習**：交互（interleaved）画像–テキスト、キャプション、OCR（LaTeX OCR・RenderedText・社内 OCR）、表/図表/文書理解、Web→コード・プロット→Python、視覚プロンプト付き QA、そして**視覚グラウンディング**を混合。VL : テキスト ≒ 7 : 3 を保ち、テキスト能力も維持する。
- **SFT**：社内の高品質 VQA・OCR・文書・表/図表・推論・教科書・グラウンディングのデータに、純テキスト対話を加えて指示追従を仕上げる。

グラウンディングは座標を特殊トークンで表す形式（参照表現と検出枠を `<|ref|>` / `<|det|>` のようなトークンで囲み、座標を 0〜999 に正規化）で学習され、「画像内の特定オブジェクトを枠で指す」能力や、複数画像にまたがる同カテゴリ物体の対応付けといった新タスクを獲得しました。これは前章 DeepSeek-VL には無かった能力で、動的タイリングによる高解像と相性が良い拡張です。

配備の観点では、効く場所の違いを意識すると見通しが良くなります。MLA は「キャッシュ＝メモリ／帯域」に、MoE の疎活性は「FLOPs＝計算量」に効きます。両者を組み合わせることで、動的タイリングで視覚トークンが増える高解像入力でも、メモリと計算の両面で実用的な推論速度を保てます。学習側でも HAI-LLM 上でパイプライン／テンソル／エキスパート並列を併用し、タイル数のばらつきによる負荷不均衡を画像タイルのロードバランシングで緩和しています。

結果として DeepSeek-VL2 は、論文の主張する通り「**同等以下の活性パラメータ**で、既存のオープンソースの密モデル・MoE モデルに対して競争的〜最先端」の性能域に入ります。Tiny は全体活性わずか 1.0B でありながら同規模帯で戦え、Small・フルは活性を 2.8B / 4.5B に抑えたまま大型密モデルに匹敵する、という「活性パラメータ対性能」の効率が DeepSeek-VL2 の核心です。

## まとめと、読解後に答えたい問い

- DeepSeek-VL2 は LLaVA 系の 3 部品（視覚エンコーダ・VL アダプタ・言語モデル）を踏襲しつつ、**視覚側＝動的タイリング**・**言語側＝DeepSeekMoE＋MLA**の 2 軸で前章 DeepSeek-VL を更新した。
- **動的タイリング**は、余白最小の候補解像度 $C_R$ を選び、画像を**ローカルタイル＋全体サムネイル**に分割。単一の SigLIP-SO400M-384 で任意比・高解像を扱い、$2\times 2$ pixel shuffle で 1 タイル 196 トークンに圧縮する。
- **MoE 言語モデル**は、Router が Top-K=6 のルーティング専門家＋共有専門家だけを起動する疎活性で、**活性パラメータが総パラメータより遥かに小さい**（フルで $\approx 4.1\text{B}/27\text{B}$）。**MLA** が KV を rank=512 の潜在ベクトルに圧縮して高スループット化する（Tiny のみ MHA）。
- 規模は Tiny / Small / フル（全体活性 1.0B / 2.8B / 4.5B）。狙いは一貫して「**少ない活性パラメータで、高解像・高性能**」。

読解後に答えたい問い:

1. 動的タイリングの候補解像度を $mn\le 9$ で打ち切るのはなぜか。タイル数を増やすほど高解像に強くなる一方、視覚トークン列 $210 + 1 + m_i\cdot 14\cdot(n_i\cdot 14 + 1)$ はどう伸び、文脈長・計算コストとどこで釣り合うか。
2. なぜ Tiny だけ MHA で、Small・フルが MLA なのか。MLA の KV 圧縮（rank=512）が効くのはどんな入力（長文・多タイル）で、小容量モデルでは利得が出にくいのはなぜか。
3. フルモデルだけ「Sigmoid ルーティング＋専門家バイアス補正」を採るのはなぜか。専門家数が 64→72 に増えるとき、補助損失なしの負荷分散はどの失敗（専門家の偏り・崩壊）を防いでいるか。
4. 「活性パラメータ対性能」で密モデルに勝てるとして、MoE の総パラメータ（メモリ常駐量）は学習・配備コストにどう跳ね返るか。Tiny/Small/フルの選択は、どんな制約（VRAM・スループット・精度）で決めるべきか。
