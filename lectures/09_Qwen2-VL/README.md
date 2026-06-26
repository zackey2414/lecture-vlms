# Qwen2-VL — Perception at Any Resolution（動的解像度・M-RoPE・動画対応）

Qwen-VL までの VLM は、入力画像を 224×224 のような固定解像度へリサイズしてから視覚トークン列に変換していました。Qwen2-VL（*Enhancing Vision-Language Model's Perception of the World at Any Resolution*, arXiv:2409.12191）は、この「あらかじめ決め打ちした解像度」という前提を捨て、**任意解像度の画像を可変個の視覚トークンへ動的に変換する Naive Dynamic Resolution** と、**位置を時間・高さ・幅の3成分へ分解する M-RoPE** の2点で知覚を拡張します。本ページでは、この2つの仕組みがなぜ自然に「画像／複数画像／動画を1モデルで」扱う基盤になるのかを腹落ちさせます。

## 全体像（まず一枚で）

Qwen2-VL は Qwen-VL の枠組み（視覚エンコーダ → コネクタ → LLM）を踏襲しつつ、各部品を「解像度に縛られない」よう作り替えています。視覚エンコーダは約 675M パラメータの ViT で、画像でも動画でも共通に使われ、その後段で隣接 2×2 トークンを MLP で 1 トークンへ圧縮してから Qwen2 LLM（1.5B / 7.6B / 72B）へ渡します。ViT の規模を LLM の大小によらず一定に保つことで、LLM をスケールさせても視覚側の計算負荷が膨らまない設計です。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="Qwen2-VLの全体像。任意解像度の画像をViTで動的に符号化し、隣接2x2トークンをMLPで圧縮してからQwen2 LLMへ統一トークン列として入力する流れを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="74" y="36" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">任意解像度の入力</text>
  <rect x="48" y="55" width="44" height="34" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="42" y="130" width="62" height="48" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="34" y="215" width="78" height="104" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="74" y="345" text-anchor="middle" font-size="11" fill="#0e7490">ネイティブ解像度・縦横比そのまま</text>
  <line x1="116" y1="185" x2="140" y2="185" stroke="#71717a" stroke-width="2"/>
  <polygon points="150,185 140,180 140,190" fill="#71717a"/>
  <rect x="150" y="120" width="130" height="130" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="215" y="165" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3">ViT ≈675M</text>
  <text x="215" y="190" text-anchor="middle" font-size="12" fill="#3730a3">2D-RoPE</text>
  <text x="215" y="212" text-anchor="middle" font-size="11" fill="#3730a3">絶対位置埋め込みを撤廃</text>
  <text x="215" y="234" text-anchor="middle" font-size="11" fill="#3730a3">可変個の視覚トークン</text>
  <line x1="282" y1="185" x2="322" y2="185" stroke="#71717a" stroke-width="2"/>
  <polygon points="332,185 322,180 322,190" fill="#71717a"/>
  <rect x="332" y="130" width="120" height="110" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <text x="392" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">MLP merge</text>
  <text x="392" y="194" text-anchor="middle" font-size="12" fill="#3730a3">2×2 → 1</text>
  <text x="392" y="216" text-anchor="middle" font-size="11" fill="#3730a3">トークン圧縮</text>
  <line x1="454" y1="185" x2="494" y2="185" stroke="#71717a" stroke-width="2"/>
  <polygon points="504,185 494,180 494,190" fill="#71717a"/>
  <text x="603" y="106" text-anchor="middle" font-size="11" font-weight="700" fill="#18181b">統一トークン列</text>
  <rect x="510" y="116" width="15" height="26" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="527" y="116" width="15" height="26" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="544" y="116" width="15" height="26" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="561" y="116" width="15" height="26" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="578" y="116" width="15" height="26" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="595" y="116" width="15" height="26" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="612" y="116" width="15" height="26" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <rect x="629" y="116" width="15" height="26" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <rect x="646" y="116" width="15" height="26" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <rect x="663" y="116" width="15" height="26" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="603" y="159" text-anchor="middle" font-size="10" fill="#71717a">&lt;|vision_start|&gt; … &lt;|vision_end|&gt; …テキスト</text>
  <line x1="603" y1="146" x2="603" y2="168" stroke="#71717a" stroke-width="2"/>
  <polygon points="603,176 598,166 608,166" fill="#71717a"/>
  <rect x="505" y="176" width="196" height="86" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="603" y="212" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">Qwen2 LLM Decoder</text>
  <text x="603" y="236" text-anchor="middle" font-size="12" fill="#16a34a">1.5B / 7.6B / 72B</text>
</svg><figcaption>全体像。<b>共通の ViT（≈675M）が任意解像度を可変長の視覚トークンへ符号化</b>し、隣接 2×2 を MLP で 1 トークンへ圧縮、<code>&lt;|vision_start|&gt;</code>／<code>&lt;|vision_end|&gt;</code> で囲んでテキストと一本のトークン列にまとめて Qwen2 LLM に入力します。</figcaption></figure>

モデル構成は次の通りです。視覚エンコーダは全サイズ共通です。モデル名（2B/7B/72B）は概ね LLM 規模に由来する通称で、論文が報告する総パラメータ数は 2B/8B/72B（中位モデルは名前が 7B・総パラメータ約 8B・LLM 単体 7.6B）と一致しない点に注意してください。

| モデル | Vision Encoder | LLM | 位置づけ |
|---|---|---|---|
| Qwen2-VL-2B | 675M ViT | 1.5B | 最効率・オンデバイス向け |
| Qwen2-VL-7B | 675M ViT | 7.6B | コスト最適・OCR / 動画に強い |
| Qwen2-VL-72B | 675M ViT | 72B | 最高性能・推論／意思決定／エージェント |

## Naive Dynamic Resolution

従来 VLM の「one-size-fits-all」戦略、すなわちすべての画像を同じ解像度へリサイズしてから処理する方式は、実装は単純ですがスケールに応じた情報を捨ててしまい、特に高解像度画像の細部を大きく損ないます。Qwen2-VL の **Naive Dynamic Resolution** は、画像をネイティブ解像度のまま受け取り、その解像度に比例した**可変個の視覚トークン**へ動的に変換します。人間が小さな絵には少しの注意を、大きく細かい絵には多くの注意を払うのと同じ発想です。

これを可能にするため、ViT から**元の絶対位置埋め込みを撤廃し、2次元 RoPE（2D-RoPE）を導入**しました。絶対位置埋め込みは「位置 0〜N まで」という固定長を前提にしますが、2D-RoPE は patch の (行, 列) を回転として相対的に表現するため、何 patch 並んでもよく、任意解像度・任意縦横比を自然に扱えます。推論時には、解像度の異なる複数画像を1本のシーケンスへパックし、パック長の上限で GPU メモリ使用量を抑えます。

さらに ViT 出力をそのまま LLM に流すとトークン数が膨大になるため、**隣接する 2×2 トークンを単純な MLP で 1 トークンへ圧縮**してから LLM に渡します。圧縮後の視覚トークン列は前後を `<|vision_start|>` / `<|vision_end|>` で囲みます。具体例として、patch_size=14 の ViT で 224×224 画像を符号化すると（16×16=256 patch → 2×2 圧縮で 64）、特殊トークンを含めて **66 トークン**になってから LLM に入ります。トークン数の下限・上限は **min/max ピクセル**として指定でき、これが実質的な「視覚トークンの予算」になります。

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="解像度に応じて視覚トークン数が動的に変わる様子。小さい画像は少トークン、大きい画像は多トークンになり、隣接2x2をMLPで1トークンに圧縮し、min maxピクセルでトークン予算を制御することを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="40" y="50" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="63" y="104" text-anchor="middle" font-size="11" fill="#0e7490">小さい画像</text>
  <line x1="92" y1="68" x2="142" y2="68" stroke="#71717a" stroke-width="2"/>
  <polygon points="150,68 140,63 140,73" fill="#71717a"/>
  <rect x="158" y="58" width="15" height="15" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="176" y="58" width="15" height="15" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="194" y="58" width="15" height="15" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="320" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">少ない視覚トークン</text>
  <rect x="34" y="150" width="84" height="104" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="76" y="274" text-anchor="middle" font-size="11" fill="#0e7490">大きい画像</text>
  <line x1="124" y1="202" x2="142" y2="202" stroke="#71717a" stroke-width="2"/>
  <polygon points="150,202 140,197 140,207" fill="#71717a"/>
  <rect x="158" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="174" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="190" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="206" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="222" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="238" y="150" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="158" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="174" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="190" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="206" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="222" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="238" y="166" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="158" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="174" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="190" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="206" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="222" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="238" y="182" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="158" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="174" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="190" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="206" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="222" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="238" y="198" width="13" height="13" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="335" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#0e7490">多くの視覚トークン</text>
  <rect x="440" y="45" width="260" height="110" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="570" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">隣接 2×2 を MLP で圧縮</text>
  <rect x="470" y="90" width="22" height="22" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="494" y="90" width="22" height="22" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="470" y="114" width="22" height="22" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="494" y="114" width="22" height="22" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <line x1="522" y1="113" x2="552" y2="113" stroke="#71717a" stroke-width="2"/>
  <polygon points="560,113 550,108 550,118" fill="#71717a"/>
  <rect x="578" y="100" width="28" height="28" fill="#6366f1" stroke="#4338ca" stroke-width="2"/>
  <text x="650" y="118" text-anchor="middle" font-size="11" fill="#3730a3">1 トークン</text>
  <rect x="440" y="175" width="260" height="110" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="570" y="210" text-anchor="middle" font-size="14" font-weight="700" fill="#0e7490">min / max ピクセル</text>
  <text x="570" y="234" text-anchor="middle" font-size="12" fill="#0e7490">= 視覚トークンの予算</text>
  <text x="570" y="258" text-anchor="middle" font-size="11" fill="#0e7490">GPU メモリに合わせ動的調整</text>
</svg><figcaption>解像度に応じて<b>視覚トークン数が動的に変わる</b>（固定解像度の撤廃）。隣接 2×2 トークンを MLP で 1 トークンへ圧縮し、<b>min/max ピクセルでトークン予算</b>を制御します。</figcaption></figure>

## M-RoPE（Multimodal RoPE）

通常の LLM が使う 1D-RoPE は「系列上の何番目か」という1次元の位置しか符号化できません。しかし現実は3次元（空間2軸＋時間）であり、画像や動画の位置を1次元に潰すと空間構造や時間構造が失われます。Qwen2-VL の **M-RoPE（Multimodal Rotary Position Embedding）** は、回転位置符号を **時間（temporal）・高さ（height）・幅（width）の3成分へ分解**し、各モダリティに合った位置 ID の割り当て方を使い分けます。

- **テキスト**：3軸すべてに同一の位置 ID を与えます。3成分が同値なので、M-RoPE は実質的に従来の 1D-RoPE と等価に退化し、既存 LLM の挙動と整合します。
- **画像**：各視覚トークンの**時間 ID は固定**し、トークンの2次元位置に応じて**高さ・幅 ID を変化**させます。これで画像内の空間構造が位置に反映されます。
- **動画**：高さ・幅は画像と同じ割り当てのまま、**時間 ID をフレームごとに増加**させます。動画＝フレームの系列、という性質がそのまま位置に乗ります。

複数モダリティが混在する場合、あるモダリティの位置 ID は、直前モダリティの最大位置 ID に 1 を足した値から開始します。M-RoPE は画像・動画で使う位置 ID の値域を相対的に小さく抑えるため、推論時により長い系列への外挿（extrapolation）にも有利になります。

<figure class="lec-fig"><svg viewBox="0 0 720 380" role="img" aria-label="M-RoPEの3軸位置分解。テキストは時間 高さ 幅の3軸が同値で1D RoPEに退化、画像は時間を固定して高さと幅が空間で変化、動画はフレームごとに時間が増加することを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="206" y="14" width="16" height="14" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="228" y="26" font-size="12" font-weight="700" fill="#3730a3">時間 t</text>
  <rect x="306" y="14" width="16" height="14" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="328" y="26" font-size="12" font-weight="700" fill="#0e7490">高さ h</text>
  <rect x="406" y="14" width="16" height="14" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="428" y="26" font-size="12" font-weight="700" fill="#16a34a">幅 w</text>
  <text x="60" y="86" font-size="13" font-weight="700" fill="#18181b">テキスト</text>
  <rect x="130" y="60" width="80" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="170" y="85" text-anchor="middle" font-size="13" fill="#18181b">(3,3,3)</text>
  <rect x="220" y="60" width="80" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="260" y="85" text-anchor="middle" font-size="13" fill="#18181b">(4,4,4)</text>
  <rect x="310" y="60" width="80" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="350" y="85" text-anchor="middle" font-size="13" fill="#18181b">(5,5,5)</text>
  <text x="410" y="86" font-size="13" font-weight="700" fill="#3730a3">3軸が同値 → 1D RoPE に退化</text>
  <text x="60" y="190" font-size="13" font-weight="700" fill="#18181b">画像</text>
  <rect x="130" y="135" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="165" y="157" text-anchor="middle" font-size="12" fill="#0e7490">(4,0,0)</text>
  <rect x="208" y="135" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="243" y="157" text-anchor="middle" font-size="12" fill="#0e7490">(4,0,1)</text>
  <rect x="286" y="135" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="321" y="157" text-anchor="middle" font-size="12" fill="#0e7490">(4,0,2)</text>
  <rect x="130" y="177" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="165" y="199" text-anchor="middle" font-size="12" fill="#0e7490">(4,1,0)</text>
  <rect x="208" y="177" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="243" y="199" text-anchor="middle" font-size="12" fill="#0e7490">(4,1,1)</text>
  <rect x="286" y="177" width="70" height="34" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="321" y="199" text-anchor="middle" font-size="12" fill="#0e7490">(4,1,2)</text>
  <text x="380" y="184" font-size="12" font-weight="700" fill="#0e7490">時間 t を固定、高さ h・幅 w が空間位置で変化</text>
  <text x="60" y="318" font-size="13" font-weight="700" fill="#18181b">動画</text>
  <rect x="148" y="277" width="64" height="46" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="140" y="285" width="64" height="46" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="172" y="312" text-anchor="middle" font-size="11" fill="#3730a3">(t=4)</text>
  <rect x="298" y="277" width="64" height="46" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="290" y="285" width="64" height="46" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="322" y="312" text-anchor="middle" font-size="11" fill="#3730a3">(t=5)</text>
  <rect x="448" y="277" width="64" height="46" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <rect x="440" y="285" width="64" height="46" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="472" y="312" text-anchor="middle" font-size="11" fill="#3730a3">(t=6)</text>
  <line x1="212" y1="305" x2="286" y2="305" stroke="#71717a" stroke-width="2"/>
  <polygon points="294,305 284,300 284,310" fill="#71717a"/>
  <line x1="362" y1="305" x2="436" y2="305" stroke="#71717a" stroke-width="2"/>
  <polygon points="444,305 434,300 434,310" fill="#71717a"/>
  <text x="540" y="312" font-size="12" font-weight="700" fill="#4338ca">フレームごとに時間 t が +1</text>
</svg><figcaption>M-RoPE は位置を<b>時間・高さ・幅の3軸に分解</b>。テキストは3軸同値で 1D RoPE に退化、画像は時間固定で高さ・幅が空間に対応、<b>動画は時間軸がフレーム列に対応</b>します。</figcaption></figure>

## 動画への拡張

M-RoPE の時間軸は、動画を「フレームの系列」として画像と地続きに扱うための鍵です。Qwen2-VL は学習時に各動画を **2 FPS でサンプリング**し、さらに **depth=2 の 3D 畳み込み**を導入して隣接2フレームを1つの 3D tube として処理します。これにより系列長を増やさずにより多くのフレームを取り込めます（整合性のため、静止画は同一フレームを2枚並べた動画とみなして扱います）。各フレームは Naive Dynamic Resolution で解像度を動的に調整され、長尺動画と学習効率のバランスを取るために**1動画あたりのトークン総数に上限**（学習時は 16384）を設けます。

<figure class="lec-fig"><svg viewBox="0 0 720 280" role="img" aria-label="動画を時間軸に並んだフレーム列として扱う図。2FPSでサンプリングしたフレームをdepth2の3D畳み込みで2フレームずつ統合し、M-RoPEの時間軸tが進む様子と1動画あたりのトークン上限を示す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="40" y="28" font-size="12" font-weight="700" fill="#0e7490">2 FPS でサンプリング</text>
  <text x="690" y="28" text-anchor="end" font-size="12" font-weight="700" fill="#dc2626">≤ 16384 トークン / 動画</text>
  <rect x="60" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="160" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="260" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="360" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="460" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <rect x="560" y="45" width="46" height="36" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <line x1="83" y1="81" x2="130" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="135,122 124,120 130,111" fill="#71717a"/>
  <line x1="183" y1="81" x2="160" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="155,122 160,111 166,120" fill="#71717a"/>
  <line x1="283" y1="81" x2="330" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="335,122 324,120 330,111" fill="#71717a"/>
  <line x1="383" y1="81" x2="360" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="355,122 360,111 366,120" fill="#71717a"/>
  <line x1="483" y1="81" x2="530" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="535,122 524,120 530,111" fill="#71717a"/>
  <line x1="583" y1="81" x2="560" y2="118" stroke="#71717a" stroke-width="2"/>
  <polygon points="555,122 560,111 566,120" fill="#71717a"/>
  <rect x="110" y="120" width="70" height="46" rx="4" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="145" y="148" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">t=0</text>
  <rect x="310" y="120" width="70" height="46" rx="4" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="345" y="148" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">t=1</text>
  <rect x="510" y="120" width="70" height="46" rx="4" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="545" y="148" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">t=2</text>
  <text x="360" y="195" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">3D Conv (depth 2) で隣接2フレームを統合 → M-RoPE の時間軸 t が進む</text>
  <line x1="40" y1="225" x2="682" y2="225" stroke="#71717a" stroke-width="2"/>
  <polygon points="690,225 680,220 680,230" fill="#71717a"/>
  <text x="688" y="248" text-anchor="end" font-size="12" fill="#71717a">時間 →</text>
</svg><figcaption>動画は<b>時間軸に並んだフレーム列</b>。2 FPS のフレームを depth=2 の 3D 畳み込みで2枚ずつ統合し、<b>M-RoPE の時間軸 t</b> がフレームの進行を表します。画像・複数画像・動画を1モデルで統一的に扱えます。</figcaption></figure>

学習は Qwen-VL を踏襲した3段階です。(1) ViT のみを画像-テキストペアで学習し LLM 内の意味理解と整合させる、(2) 全パラメータを解凍して広範なデータで事前学習する、(3) ViT を固定し指示チューニング用データで LLM のみを微調整する、という流れです。これにより、任意解像度知覚・文書/OCR・多言語・エージェント（UI操作やロボット制御を逐次意思決定として扱う）といった幅広い能力を獲得します。

## 実装注記

Hugging Face Transformers では `Qwen2VLForConditionalGeneration` と、視覚入力の前処理を担うユーティリティ `qwen_vl_utils.process_vision_info` を組み合わせます。Naive Dynamic Resolution のトークン予算は、プロセッサの `min_pixels` / `max_pixels`（あるいは画像ごとの指定）で制御できます。

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
# min/max ピクセルで視覚トークン予算を制御（解像度に比例してトークン数が動的に変わる）
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

messages = [{"role": "user", "content": [
    {"type": "image", "image": "demo.jpg"},
    {"type": "text",  "text": "この画像を説明してください。"},
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)  # 画像/動画を分離・前処理
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```

動画を渡す場合は `content` に `{"type": "video", "video": ...}` を加えるだけで、同じ `process_vision_info` がフレーム列を抽出し、M-RoPE の時間軸でモデルが統一的に処理します。

## まとめと、読解後に答えたい問い

Qwen2-VL の核心は、**Naive Dynamic Resolution（任意解像度を可変長トークンへ）** と **M-RoPE（位置を時間・高さ・幅へ分解）** の2点に集約されます。前者は固定解像度の前提を撤廃して人間に近いスケール感の知覚を、後者は画像と動画を地続きに扱うための位置表現を与え、両者が組み合わさることで「画像／複数画像／動画を1モデルで」という統一が成立します。読解後に、次の問いへ自分の言葉で答えられるか確認してください。

- ViT から絶対位置埋め込みを撤廃し 2D-RoPE に置き換えると、なぜ任意解像度・任意縦横比を扱えるのか。
- min/max ピクセルによるトークン予算は、精度と計算コストのどのトレードオフを調整しているのか。
- M-RoPE がテキスト入力で 1D-RoPE に退化するのはなぜ重要で、既存 LLM 資産とどう整合するのか。
- 動画で時間 ID をフレームごとに増やすことと、3D 畳み込み（depth=2）でフレームを統合することは、それぞれ何を解決しているのか。
