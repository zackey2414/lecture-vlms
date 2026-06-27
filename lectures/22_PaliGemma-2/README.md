# PaliGemma 2 — A Family of Versatile VLMs for Transfer

PaliGemma 2 は、前章で学んだ **PaliGemma**（SigLIP 視覚エンコーダ＋線形射影＋デコーダ型言語モデル、3段階学習）の正統な後継である。アーキテクチャの骨格はそのままに、言語モデルを **Gemma 2 ファミリー（2B/9B/27B）** へ載せ替え、さらに **複数のモデルサイズ × 複数の入力解像度（224/448/896）** という「ファミリー」として公開された点が最大の特徴だ。本章のねらいは、単に新モデルの性能を眺めることではなく、**サイズと解像度という2つの軸を動かしたとき、転移（微調整）性能がどう変わるのか**という体系的な傾向を読み取ることにある。PaliGemma の構成（接頭辞への全結合注意と自己回帰生成、3段階学習）は既習として進める。

## PaliGemma からの更新（全体像）

PaliGemma 2 は「作り替え」ではなく「載せ替え」である。理解すべき更新点は次のとおり。

- **視覚エンコーダは据え置き**: PaliGemma と同じ **SigLIP-So400m**（パッチサイズ 14）を再利用する。画像は SigLIP で系列化され、**線形射影**で言語モデルの入力空間へ写像される。ここまでは前章と同一だ。
- **言語モデルを Gemma 2 へ**: 唯一の本質的な置き換えがここ。旧来の Gemma（2B 1種）に対し、Gemma 2 の **2B / 9B / 27B** を丸ごと採用し、3つの規模を選べるようにした。
- **ファミリー化**: 上記の3サイズ × 3解像度を体系的に学習・公開する。これにより「サイズを上げる」「解像度を上げる」という操作を**統制された条件**で比較できる。
- **学習レシピは同じ3段階**: PaliGemma と同じ段階構成を踏襲する（詳細は次節）。Gemma 2 由来の logits ソフトキャッピングは段階1・2で適用し、段階3（タスク微調整）では外す、という細かな調整が入る。
- **転移学習率のスケーリング**: 段階1・2の学習率は、PaliGemma で用いた基準値 $2\times10^{-5}$ に対し、3B では $\times0.5$、10B/28B では $\times0.25$ を掛ける。大きなモデルほど学習率を下げる、という運用が明示された。

命名にも注意したい。**「3B/10B/28B」は視覚エンコーダ（約4億パラメータ）を含めた総パラメータ規模**であり、中身の言語モデルはそれぞれ Gemma 2 の 2B/9B/27B にあたる。視覚エンコーダ自体は言語モデルに比べて小さく、**実際の計算量は言語モデル側に流れ込む画像トークンが支配する**。この「画像トークンが計算量を決める」という感覚が、次節のサイズ対解像度の話を理解する鍵になる。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="PaliGemma 2 のアーキテクチャ概要。入力画像を SigLIP 視覚エンコーダで画像トークンへ変換し、線形射影を経てテキストとともに Gemma 2 言語モデルへ入力し、出力テキストを自己回帰生成する流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="360" y="30" text-anchor="middle" font-size="17" font-weight="700" fill="#18181b">PaliGemma 2 の構成（視覚側は据え置き、言語モデルを Gemma 2 へ）</text>

  <rect x="24" y="150" width="92" height="120" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="70" y="172" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">入力画像</text>
  <rect x="36" y="196" width="68" height="58" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <rect x="48" y="210" width="44" height="38" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5"/>
  <rect x="58" y="220" width="24" height="22" fill="#6366f1"/>
  <text x="70" y="266" text-anchor="middle" font-size="11" fill="#4338ca">224 / 448 / 896</text>

  <line x1="118" y1="210" x2="154" y2="210" stroke="#71717a" stroke-width="2"/>
  <polygon points="144,204 156,210 144,216" fill="#71717a"/>

  <rect x="160" y="165" width="80" height="90" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="200" y="202" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">SigLIP</text>
  <text x="200" y="220" text-anchor="middle" font-size="11" fill="#0e7490">So400m / 14</text>

  <line x1="242" y1="210" x2="280" y2="210" stroke="#71717a" stroke-width="2"/>
  <polygon points="270,204 282,210 270,216" fill="#71717a"/>

  <rect x="284" y="183" width="60" height="54" rx="6" fill="#ecfeff" stroke="#06b6d4" stroke-width="2"/>
  <text x="314" y="207" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">線形</text>
  <text x="314" y="222" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">射影</text>

  <line x1="346" y1="210" x2="384" y2="210" stroke="#71717a" stroke-width="2"/>
  <polygon points="374,204 386,210 374,216" fill="#71717a"/>

  <rect x="388" y="150" width="200" height="120" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="488" y="178" text-anchor="middle" font-size="15" font-weight="700" fill="#16a34a">Gemma 2（言語モデル）</text>
  <rect x="400" y="200" width="52" height="42" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="1.5"/>
  <text x="426" y="226" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">2B</text>
  <rect x="466" y="200" width="52" height="42" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="1.5"/>
  <text x="492" y="226" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">9B</text>
  <rect x="532" y="200" width="52" height="42" rx="6" fill="#ffffff" stroke="#16a34a" stroke-width="1.5"/>
  <text x="558" y="226" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">27B</text>

  <rect x="388" y="92" width="200" height="34" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/>
  <text x="488" y="114" text-anchor="middle" font-size="11" fill="#18181b">テキストプロンプト（接頭辞）</text>
  <line x1="488" y1="126" x2="488" y2="148" stroke="#71717a" stroke-width="2"/>
  <polygon points="482,138 488,150 494,138" fill="#71717a"/>

  <line x1="590" y1="210" x2="628" y2="210" stroke="#71717a" stroke-width="2"/>
  <polygon points="618,204 630,210 618,216" fill="#71717a"/>

  <rect x="632" y="183" width="80" height="54" rx="6" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="672" y="214" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">出力トークン</text>

  <text x="300" y="300" text-anchor="middle" font-size="11" fill="#0e7490">画像トークン: 256 / 1024 / 4096</text>
</svg><figcaption>視覚エンコーダ SigLIP-So400m と線形射影は PaliGemma から不変で、言語モデルだけが Gemma 2 の 2B/9B/27B に置き換わる。<b>要点</b>: 接頭辞（画像＋テキスト）に全結合注意をかけ、出力を自己回帰生成する設計はそのまま受け継ぐ。</figcaption></figure>

## サイズ×解像度のファミリー

PaliGemma 2 を「1つのモデル」ではなく「**ファミリー**」として捉えるのが本章の核心だ。動かせる軸は2つある。

**軸1: 言語モデルの規模.** Gemma 2 の 2B/9B/27B を選べる。総パラメータでは次のようになる。

| モデル | 言語モデル | 総パラメータ（概数） | 視覚エンコーダ |
| --- | --- | --- | --- |
| PaliGemma 2 3B | Gemma 2 2B | 約 3.0B | SigLIP-So400m |
| PaliGemma 2 10B | Gemma 2 9B | 約 9.7B | 同上 |
| PaliGemma 2 28B | Gemma 2 27B | 約 27.7B | 同上 |

**軸2: 入力解像度.** 224 / 448 / 896 の3段階。パッチサイズが 14 なので、画像トークン数は解像度の関数として決まる。

$$
L_{\text{img}} = \left(\frac{R}{14}\right)^2,\qquad R \in \{224,\,448,\,896\} \;\Longrightarrow\; L_{\text{img}} \in \{256,\,1024,\,4096\}
$$

| 解像度 | 画像トークン数 |
| --- | --- |
| 224px | 256 |
| 448px | 1024 |
| 896px | 4096 |

ここで効いてくるのが、前節の「計算量は画像トークンが支配する」という観察だ。Transformer の自己注意は系列長 $L$ に対しておよそ $O(L^2)$ で増える。解像度を 224 から 448 へ上げるとトークン数は4倍になり、注意のコストはさらに大きく膨らむ。論文の計算コスト表が示すのも同じことで、**解像度を1段上げる負担は、言語モデルを1段大きくする負担と同程度**になる。サイズと解像度は「どちらも計算予算を食う、対等な2つのつまみ」だと考えるとよい。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="モデルサイズと解像度のマトリクス図。縦に 3B・10B・28B の言語モデル規模、横に 224・448・896 の解像度と画像トークン数を並べ、各セルに相対計算コストを示す。右下に向かうほどコストが大きくなる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="360" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">サイズ × 解像度のマトリクス（数値は相対計算コスト）</text>

  <text x="420" y="58" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">解像度（px）と画像トークン数 →</text>
  <text x="240" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">224 / 256 tok</text>
  <text x="420" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">448 / 1024 tok</text>
  <text x="600" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">896 / 4096 tok</text>

  <text x="78" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">3B</text>
  <text x="78" y="142" text-anchor="middle" font-size="10" fill="#71717a">Gemma2 2B</text>
  <text x="78" y="196" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">10B</text>
  <text x="78" y="212" text-anchor="middle" font-size="10" fill="#71717a">Gemma2 9B</text>
  <text x="78" y="266" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">28B</text>
  <text x="78" y="282" text-anchor="middle" font-size="10" fill="#71717a">Gemma2 27B</text>

  <rect x="156" y="96" width="168" height="60" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="240" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">≈ 1.0×</text>
  <rect x="336" y="96" width="168" height="60" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="420" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">≈ 4.6×</text>
  <rect x="516" y="96" width="168" height="60" rx="6" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
  <text x="600" y="132" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">≈ 23.5×</text>

  <rect x="156" y="166" width="168" height="60" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="240" y="202" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">≈ 3.7×</text>
  <rect x="336" y="166" width="168" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="420" y="202" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">≈ 18.3×</text>
  <rect x="516" y="166" width="168" height="60" rx="6" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
  <text x="600" y="202" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">≈ 67.7×</text>

  <rect x="156" y="236" width="168" height="60" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="240" y="272" text-anchor="middle" font-size="15" font-weight="700" fill="#4338ca">≈ 18.9×</text>
  <rect x="336" y="236" width="168" height="60" rx="6" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
  <text x="420" y="272" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">≈ 63.5×</text>
  <rect x="516" y="236" width="168" height="60" rx="6" fill="#4338ca" stroke="#4338ca" stroke-width="2"/>
  <text x="600" y="272" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">≈ 155.6×</text>

  <text x="360" y="328" text-anchor="middle" font-size="11" fill="#71717a">淡色ほど低コスト、濃色ほど高コスト（左上 → 右下でコスト増）</text>
</svg><figcaption>縦は言語モデルの規模、横は入力解像度と画像トークン数。各セルの数値は事前学習の相対計算コストで、右下ほど重い。<b>要点</b>: 解像度を上げることは、言語モデルを大きくするのと同程度に計算予算を押し上げる対等な「つまみ」である。</figcaption></figure>

学習は PaliGemma と同じ **3段階**で進む。段階を追うごとに解像度を引き上げていくのがポイントだ。

- **段階1（224px）**: SigLIP-So400m と Gemma 2 の学習済みチェックポイントを結合し、約10億例の多モーダル混合タスクで**共同学習**する。この段階では**どのパラメータも凍結しない**。
- **段階2（448px → 896px）**: まず 448px で5000万例、続いて 896px で1000万例を学習する。高解像度が効くタスク（文書・OCR 系）の比重を上げ、出力系列長も伸ばして、長い視覚テキストの読み取りを促す。
- **段階3（タスク微調整）**: 段階1または2のチェックポイントを起点に、目的の下流タスクへ微調整する。logits ソフトキャッピングはここで外す。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="PaliGemma 2 の3段階学習パイプライン図。段階0で各部品を単独事前学習し、段階1で224pxの共同学習、段階2で448pxから896pxへ解像度を上げた学習、段階3で下流タスクへの微調整を行う流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="360" y="28" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">3段階の学習（解像度を段階的に引き上げる）</text>

  <rect x="16" y="110" width="150" height="170" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="91" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">段階0</text>
  <text x="91" y="170" text-anchor="middle" font-size="11" fill="#71717a">各部品を単独で</text>
  <text x="91" y="188" text-anchor="middle" font-size="11" fill="#71717a">事前学習</text>
  <text x="91" y="216" text-anchor="middle" font-size="11" fill="#71717a">SigLIP-So400m</text>
  <text x="91" y="234" text-anchor="middle" font-size="11" fill="#71717a">Gemma 2</text>

  <line x1="168" y1="195" x2="186" y2="195" stroke="#71717a" stroke-width="2"/>
  <polygon points="178,189 190,195 178,201" fill="#71717a"/>

  <rect x="192" y="110" width="150" height="170" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="267" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">段階1</text>
  <text x="267" y="168" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">224px</text>
  <text x="267" y="196" text-anchor="middle" font-size="11" fill="#0e7490">約10億例で</text>
  <text x="267" y="214" text-anchor="middle" font-size="11" fill="#0e7490">結合・共同学習</text>
  <text x="267" y="242" text-anchor="middle" font-size="11" fill="#0e7490">凍結なし</text>

  <line x1="344" y1="195" x2="362" y2="195" stroke="#71717a" stroke-width="2"/>
  <polygon points="354,189 366,195 354,201" fill="#71717a"/>

  <rect x="368" y="110" width="160" height="170" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="448" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#4338ca">段階2</text>
  <text x="448" y="168" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">448px → 896px</text>
  <text x="448" y="196" text-anchor="middle" font-size="11" fill="#4338ca">高解像度タスクを</text>
  <text x="448" y="214" text-anchor="middle" font-size="11" fill="#4338ca">重み付けして強化</text>
  <text x="448" y="242" text-anchor="middle" font-size="11" fill="#4338ca">出力長を拡張（OCR）</text>

  <line x1="530" y1="195" x2="548" y2="195" stroke="#71717a" stroke-width="2"/>
  <polygon points="540,189 552,195 540,201" fill="#71717a"/>

  <rect x="554" y="110" width="160" height="170" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="634" y="138" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">段階3</text>
  <text x="634" y="170" text-anchor="middle" font-size="11" fill="#16a34a">各下流タスクへ</text>
  <text x="634" y="188" text-anchor="middle" font-size="11" fill="#16a34a">微調整</text>
  <text x="634" y="216" text-anchor="middle" font-size="11" fill="#16a34a">段階1/2 の重みから</text>
  <text x="634" y="234" text-anchor="middle" font-size="11" fill="#16a34a">開始</text>
</svg><figcaption>PaliGemma と同じ3段階。段階1で 224px の共同学習、段階2で解像度を 448 から 896 へ上げて文書・OCR 系を強化し、段階3で目的タスクへ微調整する。<b>要点</b>: 解像度は一気にではなく段階を追って引き上げる。</figcaption></figure>

## 転移性能の傾向

ファミリー化の本当の価値は、ここで明らかになる。3サイズ × 2解像度（224/448）を 30 以上の学術ベンチマークで微調整して比べると、**どのタスクがどの軸で伸びるか**が見えてくる。

**大原則: 多くのタスクは両方の軸で伸びる.** 解像度を上げても、言語モデルを大きくしても FLOPs は増え、ほとんどのタスクはどちらの増強からも恩恵を受ける。その上で、**どちらがより効くか**はタスクの性質で分かれる。

- **解像度が効くグループ**: 文字・文書・画面・チャートの理解（OCR 系）。これらのベンチマーク画像は元の解像度が 224px より明らかに大きいことが多く、細部を読むには高解像度が要る。
- **言語モデルのサイズが効くグループ**: 多言語データ（XM3600）や、高度な視覚推論（AI2D、CountBenchQA、NLVR2）。世界知識や言語理解、推論が鍵になるタスクだ。

**サイズ増の逓減.** 3B から 10B への引き上げは多くのタスクで明確に効くが、**10B から 28B への引き上げは効果が小さい、あるいは見られない**ことが多い。最大モデルは「計算・遅延の制約がなく、最高性能を狙いたい」場合に意味を持つ。逓減の一因として論文は、Gemma 2 の 27B が**ゼロから学習**されたのに対し、2B/9B は**蒸留**で作られている点を指摘している。

**学習率の傾向.** 大きなモデルほど**最適な転移学習率は低くなる**。モデルを大きくするときは学習率を下げて探索するのがよい。

**Gemma 2 と Gemma 1 の比較.** 同じ 3B・同じ解像度で比べると、PaliGemma 2 は PaliGemma を**わずかに上回る**（30 以上のベンチマーク平均で 1 点に満たない小幅な改善）。劇的な差ではないが、言語モデルの世代更新がそのまま VLM の底上げにつながることを示す。

PaliGemma の範囲を超えた**新しいタスク**でも、同じ「サイズ対解像度」の構図が繰り返される。OCR・表・楽譜のように細部の読み取りが本質のタスクは解像度で伸び、空間推論のように言語理解が鍵のタスクはサイズで伸びる。

| 新タスク | 主に効いた軸 | 補足 |
| --- | --- | --- |
| 文字検出・認識（OCR） | 解像度（896 が最良） | サイズ増の効果は薄い |
| 表構造認識 | 解像度 | サイズ増は効果薄、低解像度で軽い劣化 |
| 分子構造認識 | 解像度（448 で十分） | それ以上は頭打ち |
| 楽譜認識 | 解像度（896 が最良） | 3B → 10B で誤りは減らず |
| 長文の詳細キャプション | 解像度＋サイズの両方 | 事実整合性が向上 |
| 空間推論（VSR） | サイズ | 224 を超える解像度は効果薄 |
| 放射線レポート生成 | 解像度＋サイズの両方（小幅） | 医療ドメインでも同傾向 |

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="タスク感度の2次元マップ。横軸は解像度への感度、縦軸は言語モデルサイズへの感度。OCR や文書系は右下、多言語や視覚推論は左上、一般キャプションやVQAは右上、飽和タスクは左下に位置する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="360" y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">どの軸で伸びるか（タスク感度マップ）</text>

  <line x1="96" y1="312" x2="690" y2="312" stroke="#71717a" stroke-width="2"/>
  <polygon points="680,306 692,312 680,318" fill="#71717a"/>
  <line x1="96" y1="312" x2="96" y2="52" stroke="#71717a" stroke-width="2"/>
  <polygon points="90,62 96,50 102,62" fill="#71717a"/>
  <text x="396" y="340" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">解像度への感度 →</text>
  <text x="110" y="64" text-anchor="start" font-size="12" font-weight="700" fill="#71717a">言語モデルサイズへの感度 ↑</text>

  <ellipse cx="250" cy="135" rx="142" ry="50" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="250" y="128" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">言語モデルで伸びる</text>
  <text x="250" y="148" text-anchor="middle" font-size="11" fill="#4338ca">多言語・AI2D・計数・NLVR2・空間推論</text>

  <ellipse cx="560" cy="120" rx="130" ry="48" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="560" y="113" text-anchor="middle" font-size="13" font-weight="700" fill="#16a34a">両方で伸びる</text>
  <text x="560" y="133" text-anchor="middle" font-size="11" fill="#16a34a">一般キャプション・VQA・レポート生成</text>

  <ellipse cx="545" cy="248" rx="138" ry="50" fill="#ecfeff" stroke="#06b6d4" stroke-width="2"/>
  <text x="545" y="241" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">解像度で伸びる</text>
  <text x="545" y="261" text-anchor="middle" font-size="11" fill="#0e7490">OCR・文書・表構造・楽譜・チャート・画面</text>

  <circle cx="175" cy="268" r="10" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="195" y="272" text-anchor="start" font-size="11" fill="#71717a">飽和・容易なタスク（どちらでも伸びにくい）</text>
</svg><figcaption>横軸は解像度、縦軸は言語モデルサイズへの感度。OCR・文書・表・楽譜・画面理解は右下（解像度）、多言語・図表推論・計数・空間推論は左上（サイズ）に寄る。<b>要点</b>: タスクの性質で効く軸が違うので、サイズと解像度はタスクに合わせて選ぶ。</figcaption></figure>

## まとめと、読解後に答えたい問い

PaliGemma 2 が伝えるメッセージは明快だ。

- **PaliGemma の設計を継承し、言語モデルだけを Gemma 2 へ**載せ替えた。SigLIP＋線形射影＋デコーダ型 LM という骨格と3段階学習はそのまま。
- **3サイズ × 3解像度のファミリー**として提供することで、サイズと解像度を統制された条件で比較できるようにした。
- **サイズと解像度は対等な計算予算のつまみ**であり、解像度を1段上げる負担は言語モデルを1段大きくする負担と同程度。
- **効く軸はタスク次第**: OCR・文書・表・楽譜は解像度、多言語・視覚推論・空間推論はサイズ。多くのタスクは両方で伸びる。
- **サイズ増は逓減**しがちで、3B → 10B は効くが 10B → 28B は限定的。最大モデルは制約がない場合の最高性能狙い。

読解後、次の問いに自分の言葉で答えられるか確認してほしい。

1. PaliGemma から PaliGemma 2 への本質的な変更は何か。視覚エンコーダと言語モデルのどちらが変わり、どちらが据え置かれたか。
2. 解像度 224/448/896 に対し、画像トークン数が 256/1024/4096 になるのはなぜか。パッチサイズと自己注意の計算量から説明できるか。
3. 「解像度を上げる」ことと「言語モデルを大きくする」ことが、計算コストの観点でなぜ同程度の負担になるのか。
4. あるタスクが「解像度で伸びる」のか「サイズで伸びる」のかを、タスクの性質からどう見分けるか。OCR と空間推論を例に説明できるか。
5. 10B から 28B への伸びが鈍る一因として、論文は学習の作り方の違いをどう挙げているか。
6. 自分が新しい下流タスクに PaliGemma 2 を転移させるなら、サイズ・解像度・学習率をどう選び始めるか。
