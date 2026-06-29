# エッジ向け小型・高速MLLM — トークン削減・量子化・特化（Jetson実時間）

エッジ（Jetson 級 GPU やモバイル SoC）で MLLM を実時間動作させるという課題は、単一の銀の弾丸ではなく、**(A) 小型MLLM × (B) 視覚トークン削減 × (C) 量子化＋エッジランタイム** という 3 層の技術を組み合わせることで現実的に達成される。重要な前提は、MLLM の推論コストの主因が **視覚トークン数** にあるという点である。self-attention の計算量は系列長 $n$ に対して $\mathcal{O}(n^2 d)$（$d$=隠れ次元）で増え、画像・動画から生じる視覚トークンが系列長を支配するため、視覚トークンを $n \to n'$ と削れば計算量はおよそ $(n'/n)^2$ に縮む。本ページでは、この 3 層スタックを軸に、確度の高い研究と実機実測を arXiv ID 付きで整理する。各数値が「**エッジ実機実測**」か「**データセンタ GPU / ViT 単体での実測（＝エッジへの外挿）**」かを必ず区別し、誇張せず保守的に述べる。

## 全体像（3層スタック）

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="エッジで実時間動作させるための3層スタック。小型MLLM・視覚トークン削減・量子化とエッジランタイムを組み合わせてJetson Orinやモバイルで実時間動作させる図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="ee1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="20" y="30" font-size="15" font-weight="700" fill="#18181b">Jetson 級で実時間動作させる 3 層スタック</text>
  <rect x="40" y="64" width="380" height="72" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="230" y="94" text-anchor="middle" font-size="14" font-weight="700" fill="#166534">C. 量子化 ＋ エッジランタイム</text>
  <text x="230" y="116" text-anchor="middle" font-size="11.5" fill="#16a34a">Q4_K_M 4bit ／ llama.cpp など</text>
  <rect x="40" y="148" width="380" height="72" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="230" y="178" text-anchor="middle" font-size="14" font-weight="700" fill="#155e75">B. 視覚トークン削減・圧縮</text>
  <text x="230" y="200" text-anchor="middle" font-size="11.5" fill="#0e7490">コネクタ段 ／ 学習不要法 ／ 動画</text>
  <rect x="40" y="232" width="380" height="72" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="230" y="262" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3">A. 小型MLLM</text>
  <text x="230" y="284" text-anchor="middle" font-size="11.5" fill="#4338ca">SmolVLM ／ MiniCPM-V ／ Qwen2-VL-2B ／ MobileVLM V2</text>
  <rect x="500" y="150" width="190" height="100" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="595" y="192" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b"><tspan x="595">Jetson Orin /</tspan><tspan x="595" dy="20">モバイル SoC</tspan></text>
  <text x="595" y="236" text-anchor="middle" font-size="12" fill="#52525b">で実時間</text>
  <line x1="420" y1="184" x2="498" y2="194" stroke="#71717a" stroke-width="2" marker-end="url(#ee1)"/>
  <line x1="420" y1="200" x2="498" y2="206" stroke="#71717a" stroke-width="2" marker-end="url(#ee1)"/>
  <text x="40" y="338" font-size="12" fill="#52525b">推論コストの主因 ＝ 視覚トークン数 n（self-attention は二次）。3 層は相補的で、掛け合わせて効く。</text>
</svg>
<figcaption>3 層は独立ではなく相補的に働く。A で土台を小さくし、B で系列長 \(n\) を削り、C で重みを 4bit に落としてエッジランタイムに載せる。<b>要点</b>＝どれか一つではなく、3 層の組合せで初めて Jetson 級の実時間が現実的になる。</figcaption>
</figure>

「小さくする」には方向が複数ある。**A**＝パラメータ数を 2〜3B 級まで下げる。**B**＝視覚トークン数 $n$ を下げる（コスト主因なので効果が大きい）。**C**＝重みのビット幅を下げ（量子化）、エッジ向けランタイムで実行する。以下、確証度（高／中）を付しながら層ごとに見ていく。

## A. 小型MLLMの代表

エッジ候補となる小型 MLLM はおおむね「軽量な視覚エンコーダ（SigLIP / CLIP）＋ 小型 LLM ＋ 射影器（コネクタ）」という LLaVA 系の構成を踏襲し、サイズと視覚トークン圧縮で差を付けている。

- **SmolVLM（確度=高）**: 256M / 500M / 2.2B の 3 サイズ。視覚エンコーダは SigLIP（93M の B/16 または 400M の SO400M）、言語側は SmolLM2 系を **MLP 射影**で接続する。画像は**サブ画像分割**、動画は**フレームサンプリング**後に **pixel-shuffle** で視覚トークンを圧縮する。最小の 256M は**推論時 GPU メモリ 1GB 未満**で動き、約 300 倍大きい Idefics-80B を上回ったと報告される。arXiv:2504.05299。
- **MiniCPM-Llama3-V 2.5（確度=高）**: 約 8B 級（Llama3-8B-Instruct ＋ SigLIP SoViT-400m/14）。OpenCompass（11 ベンチ平均）**65.1** で、GPT-4V-1106（63.5）・Gemini Pro（62.9）を上回ったと報告される。arXiv:2408.01800。※この比較は**ベンダ自己申告ベンチ**であり、対照の GPT-4V は **Nov-2023 版**である点に留意。8B 級だが、後述のとおり 4bit 量子化でモバイル実機に載る。
- **Qwen2-VL（確度=高）**: 2B / 8B / 72B。このうち **2B がエッジ候補**。**Naive Dynamic Resolution**（解像度を動的に扱い視覚トークン予算を可変にする）を採用。arXiv:2409.12191。※表記揺れに注意——**公開名は 7B** だが論文中の「8B」は ViT を含めた丸めである。
- **MobileVLM V2（確度=高）**: 1.7B / 3B / 7B。1.7B / 3B は **CLIP ViT-L/14(336) ＋ MobileLLaMA-1.4B / 2.7B** で構成し、**LDPv2**（軽量ダウンサンプル射影器）で視覚トークンを **576 → 144** に圧縮する。arXiv:2402.03766。

## B. 視覚トークン削減・圧縮

### なぜ視覚トークンを削るのか

推論コストは self-attention の二次計算量 $\mathcal{O}(n^2 d)$ に支配され、その系列長 $n$ の大半が**視覚トークン**である。さらに CLIP / SigLIP の特徴は**冗長**で、隣接パッチが似た情報を持つため、性能を保ったまま大きく削れる余地がある。視覚トークンを $n \to n'$ と削れば、attention の計算量は概ね $(n'/n)^2$ に縮む。arXiv:2412.04467, arXiv:2505.18227。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="推論コストが視覚トークン数の二乗で増える様子と、トークンを576から144へ削ったときにコストが大きく下がることを示す曲線グラフ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="16" y="28" font-size="14" font-weight="700" fill="#18181b">コストは視覚トークン数 n の二乗で増える（コスト ∝ O(n²d)）</text>
  <line x1="70" y1="60" x2="70" y2="300" stroke="#71717a" stroke-width="2"/>
  <line x1="70" y1="300" x2="690" y2="300" stroke="#71717a" stroke-width="2"/>
  <text x="60" y="66" text-anchor="end" font-size="11" fill="#52525b">コスト</text>
  <text x="685" y="320" text-anchor="end" font-size="11" fill="#52525b">視覚トークン数 n</text>
  <polyline points="90,300 188,293 287,273 385,240 483,193 582,133 680,60" fill="none" stroke="#4338ca" stroke-width="3"/>
  <line x1="232" y1="300" x2="232" y2="286" stroke="#16a34a" stroke-width="2" stroke-dasharray="4 3"/>
  <circle cx="232" cy="286" r="4" fill="#16a34a"/>
  <text x="232" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">n'≈144 (削減後)</text>
  <line x1="656" y1="300" x2="656" y2="79" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 3"/>
  <circle cx="656" cy="79" r="4" fill="#dc2626"/>
  <text x="656" y="71" text-anchor="middle" font-size="11" font-weight="700" fill="#b91c1c">n=576 (削減前)</text>
  <text x="90" y="338" font-size="12" fill="#52525b">576 → 144（1/4）でコストは概ね 1/16 に。削減は二乗で効くため、まず n を削るのが筋。</text>
</svg>
<figcaption>視覚トークンを \(n \to n'\) と削ると、self-attention のコストは \(\mathcal{O}(n^2 d)\) ゆえ概ね \((n'/n)^2\) に縮む。<b>要点</b>＝コスト主因は視覚トークン数で、削減の効果は二乗で効く。</figcaption>
</figure>

### コネクタ段での圧縮（学習込み・構造側）

視覚トークンを減らす最初の場所は、ViT と LLM をつなぐ**コネクタ段**である。ここは学習で最適化される構造側の圧縮で、エッジ向け小型 MLLM の多くが採用する。

- **MiniCPM の perceiver resampler**: 1 層の cross-attention で**学習可能クエリ**に情報を集約する（64 クエリ。Llama3-V 2.5 では 96 トークン）。
- **SmolVLM の pixel-shuffle**: 空間を畳んでチャネルに移し替えてトークン数を $r^2$ 倍削減する（**小型では $r=4$ が有利**＝$1/16$）。
- **Qwen2-VL の動的解像度**: `min_pixels` / `max_pixels` で**トークン予算**を指定し、隣接特徴の **2x2 マージ**で 4 倍（$1/4$）削減する。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="画像からViTで多数の視覚トークンを得たのち、コネクタ段で64から144トークンに圧縮し、小型LLMへ渡す流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="ee2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="16" y="28" font-size="14" font-weight="700" fill="#18181b">ViT の冗長な視覚トークンを、コネクタで予算内に圧縮 → 小型LLM</text>
  <rect x="16" y="150" width="92" height="72" rx="9" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="62" y="190" text-anchor="middle" font-size="12.5" font-weight="700" fill="#155e75"><tspan x="62">画像 /</tspan><tspan x="62" dy="16">動画</tspan></text>
  <rect x="140" y="150" width="120" height="72" rx="9" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="200" y="184" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="200">ViT</tspan><tspan x="200" dy="16">CLIP / SigLIP</tspan></text>
  <text x="200" y="234" text-anchor="middle" font-size="10.5" fill="#4338ca">特徴は冗長</text>
  <rect x="300" y="116" width="184" height="140" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="392" y="142" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">コネクタ段</text>
  <text x="392" y="166" text-anchor="middle" font-size="11" fill="#166534">perceiver 64〜96 クエリ</text>
  <text x="392" y="188" text-anchor="middle" font-size="11" fill="#166534">pixel-shuffle r=4 (1/16)</text>
  <text x="392" y="210" text-anchor="middle" font-size="11" fill="#166534">2x2 マージ (1/4)</text>
  <text x="392" y="236" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">→ n' = 64〜144</text>
  <rect x="524" y="150" width="180" height="72" rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="614" y="184" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="614">小型LLM</tspan><tspan x="614" dy="16">+ FastV (layer2後)</tspan></text>
  <line x1="108" y1="186" x2="138" y2="186" stroke="#71717a" stroke-width="2" marker-end="url(#ee2)"/>
  <line x1="260" y1="186" x2="298" y2="186" stroke="#71717a" stroke-width="2" marker-end="url(#ee2)"/>
  <text x="279" y="176" text-anchor="middle" font-size="10.5" fill="#52525b">N≈576</text>
  <line x1="484" y1="186" x2="522" y2="186" stroke="#71717a" stroke-width="2" marker-end="url(#ee2)"/>
  <text x="503" y="176" text-anchor="middle" font-size="10.5" fill="#52525b">n'</text>
  <text x="16" y="320" font-size="12" fill="#52525b">構造側（コネクタ）で n を削った上に、学習不要法（FastV 等）を LLM 内で上乗せできる。</text>
</svg>
<figcaption>コネクタ段（perceiver / pixel-shuffle / 動的解像度）で視覚トークンを予算内に圧縮する。pixel-shuffle や 2x2 マージは \(r^2\) 倍削減（\(r=4 \Rightarrow 1/16\)）。<b>要点</b>＝まず構造側で \(n\) を減らし、学習不要法を上乗せする二段構え。</figcaption>
</figure>

### 学習不要のトークン削減（後付け・推論時）

既存モデルを**再学習せず**に推論時だけ視覚トークンを削る手法群。導入が容易な反面、**高速化の実測の多くはデータセンタ GPU / ViT 単体**であり、Jetson 実測ではない点に注意する。

- **FastV（確度=高）**: LLM の**第 2 層の後**で視覚トークンを約 50% 削減する。LLaVA-1.5-13B で**約 45% の FLOPs 削減**、**A40 実機で約 36% のレイテンシ短縮**を報告。arXiv:2403.06764。
- **VisionZip（確度=高）**: 重要トークンを残して**トークン 10% で約 95% の性能**を維持し、**prefilling を約 8 倍**高速化したと報告。※**データセンタ GPU 実測**。arXiv:2412.04467。
- **LLaVA-PruMerge（確度=高）**: 重要トークンの剪定と類似トークンの併合で**平均約 14 倍圧縮**（576 → 約 40）。arXiv:2403.15388。※後述のとおり「14 倍でも性能維持」は誇張で、圧縮率と精度はトレードオフ。
- **ToMe（Token Merging, 確度=高）**: **再学習不要**で ViT 内の類似トークンを併合し、**スループット約 2 倍**を**精度低下 0.2〜0.3%**で達成。arXiv:2210.09461。※**ViT 単体**の結果。
- **HoliTom（動画・確度=高）**: 学習不要で**視覚トークンを 90% 超削減**。LLaVA-OneVision-7B で **FLOPs を 6.9% まで圧縮しつつ性能 99.1% を維持**、**TTFT 2.28 倍・decode 1.32 倍**の高速化を報告。arXiv:2505.21334。※**データセンタ GPU 実測**。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="学習不要の視覚トークン削減手法FastV・VisionZip・PruMerge・ToMe・HoliTomの削減率と効果、および実測がエッジ実機かデータセンタかを比較した一覧" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="16" y="26" font-size="14" font-weight="700" fill="#18181b">学習不要トークン削減の比較（緑=実機GPU実測 / 灰=DC・ViT外挿 / シアン=動画）</text>
  <rect x="16" y="40" width="150" height="50" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="91" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">FastV</text>
  <text x="180" y="60" font-size="11.5" fill="#27272a">視覚トークン約50%削減（layer2後）。LLaVA-1.5-13B で FLOPs 約45%減・</text>
  <text x="180" y="78" font-size="11.5" fill="#27272a">A40 で約36% レイテンシ短縮。【実機GPU実測】</text>
  <rect x="16" y="100" width="150" height="50" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="91" y="130" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">VisionZip</text>
  <text x="180" y="120" font-size="11.5" fill="#27272a">トークン10%保持で性能 約95%・prefilling 約8倍。</text>
  <text x="180" y="138" font-size="11.5" fill="#52525b">※データセンタGPU実測（エッジ外挿）</text>
  <rect x="16" y="160" width="150" height="50" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="91" y="190" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">PruMerge</text>
  <text x="180" y="180" font-size="11.5" fill="#27272a">平均約14倍圧縮（576 → 約40）。</text>
  <text x="180" y="198" font-size="11.5" fill="#b91c1c">※「14倍でも性能維持」は誇張（精度トレードオフ）</text>
  <rect x="16" y="220" width="150" height="50" rx="8" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="91" y="250" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">ToMe</text>
  <text x="180" y="240" font-size="11.5" fill="#27272a">ViT で約2倍スループット・精度低下 0.2〜0.3%。</text>
  <text x="180" y="258" font-size="11.5" fill="#52525b">※ViT単体（再学習不要）</text>
  <rect x="16" y="280" width="150" height="50" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="91" y="310" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">HoliTom（動画）</text>
  <text x="180" y="300" font-size="11.5" fill="#27272a">視覚トークン90%超削減。LLaVA-OV-7B で FLOPs 6.9%/性能99.1%・</text>
  <text x="180" y="318" font-size="11.5" fill="#27272a">TTFT 2.28倍・decode 1.32倍。<tspan fill="#52525b">※データセンタGPU実測</tspan></text>
</svg>
<figcaption>学習不要法は導入が容易だが、速度の数値の多くは <b>データセンタ GPU / ViT 単体</b> の実測でエッジ実測ではない。<b>要点</b>＝FastV のみ実機 GPU（A40）の遅延短縮を報告。他はエッジへの外挿として扱う。</figcaption>
</figure>

## C. 量子化・蒸留・エッジランタイム

最後の層は、重みのビット幅を下げる**量子化**、容量を下げる**蒸留**、そしてエッジで動かす**ランタイム**である。

- **知識蒸留（確度=中）**: **EfficientVLM** は教師 X-VLM の **44.3%（約 93M）** で性能 **98.4% 維持・2.2 倍高速**を報告した。arXiv:2210.07795。※ただし **2022 年のエンコーダ型 VLM** で、近年の生成型 MLLM とは構成が異なる古い結果（確度=中）。
- **量子化・ランタイム（確度=中〜高）**: 本調査で**エッジ実機まで実証されているのは、MiniCPM の Q4_K_M（4bit）＋ llama.cpp が中心**である（下記 D を参照）。重みを 4bit へ落とすと容量が大きく下がり（後述のとおり約 5GB）、モバイル / Jetson のメモリに収まる。
- **未確認（留保）**: AWQ / GPTQ / SmoothQuant といった**VLM 特有の量子化**、TensorRT-LLM / MLC-LLM / vLLM などの**llama.cpp 以外のランタイム**の Jetson 体系的実測は、本調査では**未確認**である。これらは「やれば効くはず」という外挿であって、確証集合には含めていない。

## D. エッジ実機の実測

ここが本ページの核——**実機で測られた tokens/s** である。データセンタ（A100）の値と**ハードウェアを明示して**並べ、外挿と区別する。

- **MobileVLM（1.4B / 2.7B, 確度=高）**: **Jetson Orin GPU で 65.3 tokens/s**（同モデルを Snapdragon 888 CPU で動かすと 21.5）。arXiv:2312.16886。
- **MobileVLM V2（確度=高）**: **Jetson Orin の llama.cpp で約 15〜50 tokens/s**（精度・速度で Pareto front を構成）。A100（bs=1, 256 生成）では 1B / 3B が **37.37 / 28.97 tokens/s**。arXiv:2402.03766。
- **MiniCPM-Llama3-V 2.5（確度=高）**: **Snapdragon 8 Gen 3** で **Q4_K_M 4bit 量子化（約 5GB）＋ NPU 視覚エンコード**により、**decode 8.2 tokens/s・画像エンコード 1.3s** を達成（**最適化前は 1.3 tokens/s・64.2s**）。arXiv:2408.01800。最適化前後で **6 倍超**の改善であり、「量子化＋ランタイム＋デバイス機能」の合わせ技が効くことを示す好例。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="エッジ実機とA100でのMLLM推論速度の棒グラフ。MobileVLMがJetson Orin GPUで65.3トークン毎秒、MiniCPMがSnapdragon 8 Gen 3で最適化後8.2・最適化前1.3トークン毎秒など" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="16" y="16" font-size="14" font-weight="700" fill="#18181b">エッジ実機 vs データセンタの推論速度（tokens/s, 単一バッチ）</text>
  <rect x="80" y="38" width="12" height="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="98" y="48" font-size="11" fill="#27272a">Jetson(実機)</text>
  <rect x="200" y="38" width="12" height="12" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="218" y="48" font-size="11" fill="#27272a">モバイルSoC(実機)</text>
  <rect x="360" y="38" width="12" height="12" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="378" y="48" font-size="11" fill="#27272a">A100(データセンタ)</text>
  <rect x="510" y="38" width="12" height="12" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="528" y="48" font-size="11" fill="#27272a">最適化前</text>
  <line x1="70" y1="70" x2="70" y2="300" stroke="#71717a" stroke-width="1.5"/>
  <line x1="70" y1="300" x2="690" y2="300" stroke="#71717a" stroke-width="1.5"/>
  <text x="62" y="74" text-anchor="end" font-size="10" fill="#52525b">tok/s</text>
  <rect x="104" y="70" width="56" height="230" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="132" y="64" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">65.3</text>
  <rect x="204" y="168" width="56" height="132" rx="3" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="232" y="162" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">37.4</text>
  <rect x="304" y="198" width="56" height="102" rx="3" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
  <text x="332" y="192" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">29.0</text>
  <rect x="404" y="224" width="56" height="76" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="432" y="218" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">21.5</text>
  <rect x="504" y="271" width="56" height="29" rx="3" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="532" y="265" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">8.2</text>
  <rect x="604" y="295" width="56" height="5" rx="2" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="632" y="289" text-anchor="middle" font-size="12" font-weight="700" fill="#b91c1c">1.3</text>
  <text x="132" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MobileVLM 2.7B</text>
  <text x="132" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">Orin GPU</text>
  <text x="232" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MobileVLM V2 1B</text>
  <text x="232" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">A100</text>
  <text x="332" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MobileVLM V2 3B</text>
  <text x="332" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">A100</text>
  <text x="432" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MobileVLM 2.7B</text>
  <text x="432" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">SD888 CPU</text>
  <text x="532" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MiniCPM-V 2.5</text>
  <text x="532" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">SD8Gen3 後</text>
  <text x="632" y="314" text-anchor="middle" font-size="9.5" fill="#27272a">MiniCPM-V 2.5</text>
  <text x="632" y="326" text-anchor="middle" font-size="9.5" fill="#52525b">SD8Gen3 前</text>
  <text x="16" y="348" font-size="11" fill="#52525b">注: モデル・サイズ・ハードウェア・量子化が各々異なるため、棒は厳密な同条件比較ではない。</text>
</svg>
<figcaption>実機エッジ（Jetson Orin / Snapdragon）とデータセンタ（A100）を分けて並べた。<b>要点</b>＝Jetson Orin で 65.3 tok/s、モバイルでも 4bit ＋ NPU で 8.2 tok/s と実時間域に届くが、最適化前は 1.3 tok/s であり最適化が必須。条件が異なる棒なので順位の単純比較は不可。</figcaption>
</figure>

## 現実的なレシピ

以上を**合成した手順**（確度=中＝個々の部品は確証済みだが、全部を 1 本のパイプラインで通した端から端までの実測は本調査の確証集合外）。

1. **モデル選定**: 2〜3B 級を選ぶ（SmolVLM-2.2B / Qwen2-VL-2B / MobileVLM V2-3B / MiniCPM-V 系）。
2. **視覚トークン削減**: コネクタで予算を圧縮（perceiver 64〜96 / pixel-shuffle $r=4$ / 動的解像度）し、画像は FastV・VisionZip・PruMerge、動画は HoliTom を上乗せ。
3. **4bit 量子化**: Q4_K_M（約 5GB）へ。
4. **エッジランタイム**: llama.cpp 等で実行。

**存在証明**: MobileVLM が **Jetson Orin で 65.3 tok/s**、MiniCPM が **Snapdragon 8 Gen 3 で 8.2 tok/s** を実機で示している。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="モデル選定・視覚トークン削減・4bit量子化・エッジランタイムの4ステップからなる現実的なレシピのパイプライン図と存在証明" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="ee3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="16" y="32" font-size="15" font-weight="700" fill="#18181b">現実的なレシピ（4 ステップ）</text>
  <rect x="12" y="118" width="160" height="96" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="92" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">(1) モデル選定</text>
  <text x="92" y="174" text-anchor="middle" font-size="10.5" fill="#4338ca"><tspan x="92">2〜3B 級</tspan><tspan x="92" dy="15">SmolVLM / Qwen2-VL</tspan><tspan x="92" dy="15">MobileVLM V2 / MiniCPM</tspan></text>
  <rect x="190" y="118" width="160" height="96" rx="10" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="270" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">(2) トークン削減</text>
  <text x="270" y="174" text-anchor="middle" font-size="10.5" fill="#0e7490"><tspan x="270">コネクタ予算 +</tspan><tspan x="270" dy="15">画像 FastV/VisionZip</tspan><tspan x="270" dy="15">動画 HoliTom</tspan></text>
  <rect x="368" y="118" width="150" height="96" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="443" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">(3) 4bit 量子化</text>
  <text x="443" y="178" text-anchor="middle" font-size="10.5" fill="#16a34a"><tspan x="443">Q4_K_M</tspan><tspan x="443" dy="15">約 5GB</tspan></text>
  <rect x="536" y="118" width="170" height="96" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="621" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">(4) ランタイム</text>
  <text x="621" y="178" text-anchor="middle" font-size="10.5" fill="#4338ca"><tspan x="621">llama.cpp 等</tspan><tspan x="621" dy="15">エッジ実行</tspan></text>
  <line x1="172" y1="166" x2="188" y2="166" stroke="#71717a" stroke-width="2" marker-end="url(#ee3)"/>
  <line x1="350" y1="166" x2="366" y2="166" stroke="#71717a" stroke-width="2" marker-end="url(#ee3)"/>
  <line x1="518" y1="166" x2="534" y2="166" stroke="#71717a" stroke-width="2" marker-end="url(#ee3)"/>
  <rect x="12" y="252" width="694" height="78" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="359" y="282" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">存在証明（実機実測）</text>
  <text x="359" y="308" text-anchor="middle" font-size="12" fill="#3f3f46">MobileVLM: Jetson Orin 65.3 tok/s ／ MiniCPM-V 2.5: Snapdragon 8 Gen 3 で 8.2 tok/s（4bit + NPU）</text>
</svg>
<figcaption>個々の部品は確証済みだが、4 ステップを 1 本に通した端から端までの実測は本調査の確証集合外（確度=中）。<b>要点</b>＝この順で組めば実時間域に届く、という「合成された経験則」であり、両端の存在証明が後ろ盾。</figcaption>
</figure>

## 反証・留保（注意）

- **反証**: 「PruMerge は約 14 倍圧縮でも性能維持」という主張は**反証する**。圧縮率を上げれば精度は基本的に落ちるため、**圧縮率と精度のトレードオフは保守的に**見積もるべきで、特定ベンチでの好結果を一般化しない。
- **自己申告**: 本ページの数値の多くは**ベンダ／著者の自己申告**である。特に MiniCPM の「OpenCompass で GPT-4V 超え」は**自己申告ベンチ・GPT-4V は Nov-2023 版**という条件付きで読む。
- **エッジ実測 vs 外挿**: トークン削減の高速化数値（VisionZip 約 8 倍 / ToMe 約 2 倍 / HoliTom など）は **データセンタ GPU または ViT 単体**の実測であり、**Jetson 実測ではない（＝外挿）**。FastV の A40 遅延短縮は実機 GPU だが Jetson ではない。
- **未確認・要追加調査**: VLM 特有の量子化（AWQ / GPTQ / SmoothQuant）、llama.cpp 以外のランタイム（TensorRT-LLM / MLC-LLM / vLLM）の Jetson 体系的実測、**物体中心 / grounding 能力を保ったままの小型化**、そして **InternVL2/2.5・Phi-3.5-vision・PaliGemma・TinyLLaVA** は本調査の確証集合に未含で、要追加調査。
- **表記揺れ**: パラメータ数の表記に注意（例: Qwen2-VL の「8B」は実体 7B ＋ ViT 込みの丸め。MiniCPM-Llama3-V 2.5 は約 8B で「小型」の定義からはやや大きい）。

## まとめと、読解後に答えたい問い

エッジ実時間化は **(A) 小型MLLM × (B) 視覚トークン削減 × (C) 量子化＋ランタイム** の掛け算であり、コスト主因が視覚トークン数（$\mathcal{O}(n^2 d)$）であるがゆえに、**$n$ を削る B が効率改善の中心**になる。実機の存在証明は **MobileVLM（Jetson Orin 65.3 tok/s）** と **MiniCPM（Snapdragon 8.2 tok/s, 4bit＋NPU）** であり、後者は最適化前 1.3 tok/s からの 6 倍超で「最適化の合わせ技」の価値を示す。一方、トークン削減手法の華々しい高速化数値の大半は**データセンタ実測の外挿**であり、Jetson 実測での裏取りはこれからの課題である。

**読解後に答えたい問い**:

1. 自分の用途（画像 QA か動画か grounding か）で、視覚トークン予算 $n'$ をどこまで削れば精度がどれだけ落ちるか。$(n'/n)^2$ の理論削減は実機レイテンシにどこまで反映されるか。
2. コネクタ段の構造的圧縮（perceiver / pixel-shuffle / 動的解像度）と、学習不要法（FastV 等）は**重ねて効く**のか、それとも互いに食い合うのか。
3. VisionZip / ToMe / HoliTom の高速化は、Jetson Orin で実測すると外挿どおり出るのか。出ないとしたらボトルネックはどこ（メモリ帯域 / カーネル最適化の有無）か。
4. 4bit 量子化（Q4_K_M）で精度はどの能力（OCR / grounding / 細粒度）から先に劣化するか。AWQ / GPTQ は llama.cpp 経路に対して Jetson で優位か。
5. 「特定能力に長けた小型 MLLM」を作るとき、汎用性能を犠牲にしてでも狙うべきタスクは何か。蒸留（C）とトークン削減（B）はその特化とどう組み合わせるべきか。

## 出典

**A. 小型MLLM**

- SmolVLM（256M/500M/2.2B、256M は GPU メモリ 1GB 未満）— [arXiv:2504.05299](https://arxiv.org/abs/2504.05299)
- MiniCPM-Llama3-V 2.5（約8B、OpenCompass 65.1、※自己申告・GPT-4V は Nov-2023 版）— [arXiv:2408.01800](https://arxiv.org/abs/2408.01800)
- Qwen2-VL（2B/8B/72B、Naive Dynamic Resolution、※公開名 7B）— [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- MobileVLM V2（1.7B/3B/7B、LDPv2 で 576→144）— [arXiv:2402.03766](https://arxiv.org/abs/2402.03766)

**B. 視覚トークン削減・圧縮**

- 視覚トークンが推論コストを支配・特徴の冗長性 — [arXiv:2412.04467](https://arxiv.org/abs/2412.04467)（VisionZip）, [arXiv:2505.18227](https://arxiv.org/abs/2505.18227)
- FastV（layer2後に約50%削減、A40 で約36% レイテンシ短縮）— [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
- VisionZip（トークン10%で約95%性能、prefilling 約8倍、※DC GPU実測）— [arXiv:2412.04467](https://arxiv.org/abs/2412.04467)
- LLaVA-PruMerge（平均約14倍圧縮 576→約40）— [arXiv:2403.15388](https://arxiv.org/abs/2403.15388)
- ToMe（再学習不要、ViT で約2倍スループット・精度低下0.2-0.3%）— [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- HoliTom（動画・90%超削減、LLaVA-OV-7B で FLOPs 6.9%/性能99.1%）— [arXiv:2505.21334](https://arxiv.org/abs/2505.21334)

**C. 量子化・蒸留・ランタイム**

- EfficientVLM（教師の44.3%＝約93Mで性能98.4%維持・2.2倍高速、※2022・エンコーダ型、確度=中）— [arXiv:2210.07795](https://arxiv.org/abs/2210.07795)

**D. エッジ実機実測**

- MobileVLM（Jetson Orin GPU 65.3 tok/s、Snapdragon 888 CPU 21.5）— [arXiv:2312.16886](https://arxiv.org/abs/2312.16886)
- MobileVLM V2（Jetson Orin llama.cpp 約15-50 tok/s、A100 で 1B/3B=37.37/28.97）— [arXiv:2402.03766](https://arxiv.org/abs/2402.03766)
- MiniCPM-Llama3-V 2.5（Snapdragon 8 Gen 3、Q4_K_M 4bit＋NPU で decode 8.2 tok/s・最適化前 1.3）— [arXiv:2408.01800](https://arxiv.org/abs/2408.01800)
