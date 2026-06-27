# CLIP — Learning Transferable Visual Models from Natural Language

CLIP（Contrastive Language-Image Pre-training）は、画像とテキストを別々のエンコーダで符号化し、「どのキャプションがどの画像に対応するか」をバッチ内で当てる**対照学習**だけで、約 4 億の画像-テキストペアから汎用的な視覚表現を獲得する手法です。ラベルの代わりに自然言語そのものを教師に使うため、固定されたクラス集合に縛られず、学習後はクラス名を文へ変換するだけで多数のタスクへ **ゼロショット転移** できます。CLIP 自身は言語を生成しませんが、ここで得られる「言語と整合した視覚エンコーダ」は、LLaVA や BLIP-2 をはじめとする現行 MLLM の視覚入力の土台になっています。腹落ちさせたいのは「対照損失は結局なにを学ぶのか」と「なぜクラス名を文にするだけで分類器になるのか」の 2 点です。

## 全体像（まず一枚で）

構成は **二重エンコーダ（dual encoder）** です。**画像エンコーダ**（ResNet 系 または Vision Transformer）と **テキストエンコーダ**（Transformer）がそれぞれ独立に入力を符号化し、各出力を **線形射影** で同一次元の **共有埋め込み空間** へ写したうえで **L2 正規化** します。正規化済みなので、2 つの埋め込みの内積がそのまま **コサイン類似度** になります。学習は WIT（WebImageText、約 4 億の画像-テキストペア）上での **対照事前学習のみ**。設計はあえて簡素で、表現と埋め込み空間の間は非線形ではなく **線形射影だけ**、データ拡張も **ランダム正方クロップだけ** です。

埋め込みたい直感は「画像とその説明文が、空間の中で **同じ向き** を指すように両エンコーダを引き寄せる」こと。対応する画像と文のコサイン類似度 $\cos(\mathbf{I}_i,\mathbf{T}_i)$ を上げ、対応しない組 $\cos(\mathbf{I}_i,\mathbf{T}_j)\ (i\neq j)$ を下げます。

<figure class="lec-fig"><svg viewBox="0 0 720 380" role="img" aria-label="画像エンコーダとテキストエンコーダの二重構造。画像とキャプションをそれぞれ符号化し線形射影とL2正規化を経て共有埋め込み空間へ写し、対応ペアのコサイン類似度を最大化する全体像の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="18" y="70" width="92" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="64" y="95" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力画像</text>
  <text x="64" y="113" text-anchor="middle" font-size="10" fill="#3f3f46">1枚</text>
  <line x1="110" y1="99" x2="130" y2="99" stroke="#71717a" stroke-width="2"/>
  <polygon points="138,99 128,94 128,104" fill="#71717a"/>
  <rect x="140" y="64" width="146" height="70" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="213" y="94" text-anchor="middle" font-size="14" font-weight="700" fill="#155e75">画像エンコーダ</text>
  <text x="213" y="114" text-anchor="middle" font-size="11" fill="#155e75">ResNet 系 / ViT</text>
  <line x1="286" y1="99" x2="306" y2="99" stroke="#71717a" stroke-width="2"/>
  <polygon points="314,99 304,94 304,104" fill="#71717a"/>
  <rect x="316" y="68" width="118" height="62" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/>
  <text x="375" y="93" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">線形射影</text>
  <text x="375" y="111" text-anchor="middle" font-size="10" fill="#155e75">+ L2正規化</text>
  <rect x="18" y="254" width="92" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="64" y="279" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">入力テキスト</text>
  <text x="64" y="297" text-anchor="middle" font-size="10" fill="#3f3f46">キャプション</text>
  <line x1="110" y1="283" x2="130" y2="283" stroke="#71717a" stroke-width="2"/>
  <polygon points="138,283 128,278 128,288" fill="#71717a"/>
  <rect x="140" y="248" width="146" height="70" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="213" y="278" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3">テキストエンコーダ</text>
  <text x="213" y="298" text-anchor="middle" font-size="11" fill="#3730a3">Transformer</text>
  <line x1="286" y1="283" x2="306" y2="283" stroke="#71717a" stroke-width="2"/>
  <polygon points="314,283 304,278 304,288" fill="#71717a"/>
  <rect x="316" y="252" width="118" height="62" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="375" y="277" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">線形射影</text>
  <text x="375" y="295" text-anchor="middle" font-size="10" fill="#3730a3">+ L2正規化</text>
  <line x1="434" y1="99" x2="486" y2="150" stroke="#71717a" stroke-width="2"/>
  <polygon points="492,156 480,153 485,143" fill="#71717a"/>
  <line x1="434" y1="283" x2="486" y2="232" stroke="#71717a" stroke-width="2"/>
  <polygon points="492,226 485,239 480,229" fill="#71717a"/>
  <rect x="470" y="92" width="232" height="198" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="586" y="118" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">共有埋め込み空間</text>
  <line x1="540" y1="220" x2="628" y2="150" stroke="#0e7490" stroke-width="2.5"/>
  <polygon points="634,145 622,148 627,158" fill="#0e7490"/>
  <text x="648" y="148" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">画像</text>
  <line x1="540" y1="220" x2="640" y2="206" stroke="#4338ca" stroke-width="2.5"/>
  <polygon points="648,205 636,201 636,212" fill="#4338ca"/>
  <text x="664" y="210" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">文</text>
  <text x="586" y="258" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">対応ペアの cos を最大化</text>
  <text x="586" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">非対応ペアの cos を最小化</text>
</svg><figcaption><b>二重エンコーダ</b>が画像と文を別々に符号化し、<b>線形射影と L2 正規化</b>で同じ空間へ。正規化済みなので内積がそのままコサイン類似度になり、<b>対応ペアは引き寄せ、非対応ペアは引き離す</b>よう両エンコーダを学習します。</figcaption></figure>

## 対照学習の損失

学習の単位は **$N$ 組の画像-テキストペアからなるバッチ** です。CLIP は「バッチ内で起こりうる $N\times N$ 通りの (画像, 文) の組のうち、実際に対応するのはどれか」を当てるよう訓練されます。具体的には、対応する $N$ 個の正例ペアのコサイン類似度を上げ、残り $N^2-N$ 個の非対応ペアの類似度を下げます。

まず、L2 正規化された埋め込み同士の内積がコサイン類似度です。

$$\cos(\mathbf{I}_i,\mathbf{T}_j)=\frac{\langle \mathbf{I}_i,\mathbf{T}_j\rangle}{\lVert \mathbf{I}_i\rVert\,\lVert \mathbf{T}_j\rVert}=\langle \mathbf{I}_i,\mathbf{T}_j\rangle \quad(\text{正規化済みなら }\lVert\mathbf{I}_i\rVert=\lVert\mathbf{T}_j\rVert=1)$$

この類似度を **温度** $\tau$ でスケールしてロジットにします。実装上は $\tau$ を直接持たず、対数パラメータ化したスカラ $t$ を学習し、ロジットを $\exp(t)=1/\tau$ 倍します（学習を安定させるため、ロジットのスケールが過大にならないよう上限でクリップされます）。CLIP は行（各画像から見た全文）と列（各文から見た全画像）の両方向に **クロスエントロピー** をかける、**対称な** 損失を最小化します。

$$\mathcal{L}=\tfrac12(\mathcal{L}_{\text{img}}+\mathcal{L}_{\text{txt}}),\quad \mathcal{L}_{\text{img}}=-\tfrac1N\sum_i \log\frac{\exp(\langle I_i,T_i\rangle/\tau)}{\sum_j \exp(\langle I_i,T_j\rangle/\tau)}$$

$$\mathcal{L}_{\text{txt}}=-\tfrac1N\sum_j \log\frac{\exp(\langle I_j,T_j\rangle/\tau)}{\sum_i \exp(\langle I_i,T_j\rangle/\tau)}$$

$\mathcal{L}_{\text{img}}$ は「各画像 $i$ にとって、正しい文 $T_i$ を $N$ 個の文の中から選ぶ」分類、$\mathcal{L}_{\text{txt}}$ はその転置（各文から正しい画像を選ぶ）です。この形は対比表現学習でいう InfoNCE / multi-class N-pair 損失に相当し、**バッチ内の他ペアがそのまま負例** になります。だからバッチサイズが大きいほど負例が増え、学習が効きます。

<figure class="lec-fig"><svg viewBox="0 0 720 380" role="img" aria-label="バッチ内のN個の画像埋め込みを行、N個のテキスト埋め込みを列とするN×N類似度行列。対角は対応ペアで正例、非対角は非対応ペアで負例となることを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <text x="232" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3">テキスト埋め込み（列）</text>
  <text x="120" y="78" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">T1</text>
  <text x="176" y="78" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">T2</text>
  <text x="232" y="78" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">T3</text>
  <text x="288" y="78" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">T4</text>
  <text x="338" y="78" text-anchor="middle" font-size="12" font-weight="700" fill="#71717a">…</text>
  <text x="64" y="210" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75" transform="rotate(-90 64 210)">画像埋め込み（行）</text>
  <text x="96" y="120" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">I1</text>
  <text x="96" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">I2</text>
  <text x="96" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">I3</text>
  <text x="96" y="288" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">I4</text>
  <rect x="92" y="92" width="56" height="56" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <rect x="148" y="92" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="204" y="92" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="260" y="92" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="92" y="148" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="148" y="148" width="56" height="56" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <rect x="204" y="148" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="260" y="148" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="92" y="204" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="148" y="204" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="204" y="204" width="56" height="56" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <rect x="260" y="204" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="92" y="260" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="148" y="260" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="204" y="260" width="56" height="56" fill="#fee2e2" stroke="#71717a" stroke-width="1"/>
  <rect x="260" y="260" width="56" height="56" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="120" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">+</text>
  <text x="176" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">+</text>
  <text x="232" y="236" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">+</text>
  <text x="288" y="292" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">+</text>
  <rect x="430" y="100" width="270" height="74" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="565" y="128" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">対角（緑）= 対応ペア</text>
  <text x="565" y="150" text-anchor="middle" font-size="11" fill="#166534">正例 N 個 → cos を上げる</text>
  <rect x="430" y="186" width="270" height="74" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="565" y="214" text-anchor="middle" font-size="12" font-weight="700" fill="#991b1b">非対角（赤）= 非対応ペア</text>
  <text x="565" y="236" text-anchor="middle" font-size="11" fill="#991b1b">負例 N²−N 個 → cos を下げる</text>
  <rect x="430" y="272" width="270" height="64" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="565" y="298" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">行方向と列方向の両方で</text>
  <text x="565" y="318" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">クロスエントロピー（対称）</text>
</svg><figcaption>バッチで作る <b>N×N 類似度行列</b>。<b>対角＝対応ペア（正例 N 個）</b>を上げ、<b>非対角＝非対応ペア（負例 N²−N 個）</b>を下げます。行方向（画像→文）と列方向（文→画像）の両方にクロスエントロピーをかけるので損失は対称です。</figcaption></figure>

論文の中核実装は、数行の擬似コードに集約されています。「特徴抽出 → 線形射影と L2 正規化 → スケール済みコサイン類似度の行列 → 対称クロスエントロピー」という流れがそのまま読み取れます。

```python
# image_encoder : ResNet または Vision Transformer
# text_encoder  : CBOW または Text Transformer
# I[n, h, w, c] : 整列済み画像のミニバッチ
# T[n, l]       : 整列済みテキストのミニバッチ
# W_i, W_t      : 各モダリティ → 共有埋め込みへの学習済み射影
# t             : 学習される温度パラメータ（exp(t) = 1/τ）

I_f = image_encoder(I)                      # 画像特徴 [n, d_i]
T_f = text_encoder(T)                       # テキスト特徴 [n, d_t]

I_e = l2_normalize(np.dot(I_f, W_i), axis=1)  # 射影 + 正規化 [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  # 射影 + 正規化 [n, d_e]

logits = np.dot(I_e, T_e.T) * np.exp(t)     # スケール済みコサイン類似度 [n, n]

labels = np.arange(n)                       # 対角が正解
loss_i = cross_entropy_loss(logits, labels, axis=0)  # 画像 → 文
loss_t = cross_entropy_loss(logits, labels, axis=1)  # 文 → 画像
loss   = (loss_i + loss_t) / 2              # 対称損失
```

ここでのポイントは、教師信号が **正解インデックスの並び（対角）だけ** で済むこと。人手のクラスラベルは一切不要で、「同じ行に並んだ画像と文が対応する」という事実だけが教師になります。だから、Web から集めた緩いペアでもそのまま大規模に学習できます。

## ゼロショット転移

CLIP は「画像と文が対応するか」を学んだので、**分類器を作らずに分類** できます。手順はこうです。(1) データセットの各クラス名を「a photo of a {class}.」のような **文（プロンプト）** に変換し、テキストエンコーダで埋め込む。(2) これらのクラス文埋め込みが、そのまま線形分類器の **重みベクトル** になる。(3) 入力画像の埋め込みとの **コサイン類似度** を全クラスについて計算し、最大の文を予測ラベルとする。

$$\hat{y}=\arg\max_{k}\ \cos(\mathbf{I},\mathbf{T}_k),\qquad p(y=k\mid \mathbf{I})=\frac{\exp\!\big(\cos(\mathbf{I},\mathbf{T}_k)/\tau\big)}{\sum_{k'}\exp\!\big(\cos(\mathbf{I},\mathbf{T}_{k'})/\tau\big)}$$

ここでテキストエンコーダは、クラス名（自然言語）から分類器の重みを動的に生成する **ハイパーネットワーク** とみなせます。固定の softmax 分類器と違い、クラス集合を **言語で記述するだけ** で差し替えられるので、ImageNet 用に学習し直さずとも OCR・行動認識・細粒度分類など多様なデータセットへ転移します。クラス文埋め込みは一度計算すればキャッシュでき、以降の全画像で再利用できます。

ただし弱点もあります。クラス名が単語ひとつだと文脈が乏しく、**多義語**（例: crane が「鶴」か「クレーン」か）を取り違えることがあります。学習時のテキストは単語ではなく文が普通なので、素の単語より「a photo of a {class}.」のような文テンプレートのほうが分布のずれを埋められ、精度が上がります（論文では ImageNet で約 1.3 ポイントの改善）。複数のプロンプトを用意して埋め込みを平均する **プロンプトアンサンブル** も有効です。

<figure class="lec-fig"><svg viewBox="0 0 720 400" role="img" aria-label="クラス名をプロンプト文に変換してテキストエンコーダで埋め込み、入力画像の埋め込みとのコサイン類似度が最大の文を選ぶゼロショット分類の流れの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="16" y="64" width="92" height="124" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="62" y="86" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">クラス名</text>
  <text x="62" y="112" text-anchor="middle" font-size="11" fill="#3f3f46">plane</text>
  <text x="62" y="132" text-anchor="middle" font-size="11" fill="#3f3f46">car</text>
  <text x="62" y="152" text-anchor="middle" font-size="11" fill="#3f3f46">dog</text>
  <text x="62" y="172" text-anchor="middle" font-size="11" fill="#3f3f46">bird …</text>
  <line x1="108" y1="126" x2="126" y2="126" stroke="#71717a" stroke-width="2"/>
  <polygon points="134,126 124,121 124,131" fill="#71717a"/>
  <rect x="136" y="96" width="148" height="60" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="210" y="120" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">プロンプト化</text>
  <text x="210" y="140" text-anchor="middle" font-size="10" fill="#3730a3">a photo of a {class}.</text>
  <line x1="284" y1="126" x2="302" y2="126" stroke="#71717a" stroke-width="2"/>
  <polygon points="310,126 300,121 300,131" fill="#71717a"/>
  <rect x="312" y="92" width="120" height="68" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="372" y="120" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">テキスト</text>
  <text x="372" y="138" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">エンコーダ</text>
  <line x1="432" y1="126" x2="450" y2="126" stroke="#71717a" stroke-width="2"/>
  <polygon points="458,126 448,121 448,131" fill="#71717a"/>
  <rect x="460" y="86" width="60" height="22" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="490" y="101" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">T1</text>
  <rect x="460" y="114" width="60" height="22" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="490" y="129" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">T2</text>
  <rect x="460" y="142" width="60" height="22" rx="5" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="490" y="157" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">T3 …</text>
  <rect x="16" y="276" width="92" height="64" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="62" y="313" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">入力画像</text>
  <line x1="108" y1="308" x2="126" y2="308" stroke="#71717a" stroke-width="2"/>
  <polygon points="134,308 124,303 124,313" fill="#71717a"/>
  <rect x="136" y="274" width="148" height="68" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="210" y="302" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">画像</text>
  <text x="210" y="320" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">エンコーダ</text>
  <line x1="284" y1="308" x2="302" y2="308" stroke="#71717a" stroke-width="2"/>
  <polygon points="310,308 300,303 300,313" fill="#71717a"/>
  <rect x="312" y="286" width="60" height="44" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="342" y="313" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">I</text>
  <line x1="490" y1="164" x2="540" y2="210" stroke="#71717a" stroke-width="2"/>
  <polygon points="546,216 534,212 539,203" fill="#71717a"/>
  <line x1="372" y1="308" x2="538" y2="252" stroke="#71717a" stroke-width="2"/>
  <polygon points="546,249 534,250 538,260" fill="#71717a"/>
  <rect x="500" y="216" width="200" height="74" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
  <text x="600" y="244" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">コサイン類似度を計算</text>
  <text x="600" y="266" text-anchor="middle" font-size="11" fill="#3730a3">最大の文を選ぶ（argmax）</text>
  <line x1="600" y1="290" x2="600" y2="312" stroke="#71717a" stroke-width="2"/>
  <polygon points="600,320 595,310 605,310" fill="#71717a"/>
  <rect x="486" y="322" width="228" height="48" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="600" y="351" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">予測: a photo of a dog.</text>
</svg><figcaption>各クラス名を <b>「a photo of a {class}.」へ変換</b>してテキストエンコーダで埋め込むと、それが <b>分類器の重み</b>になります。入力画像の埋め込みと <b>コサイン類似度が最大の文</b>を選ぶだけで、再学習なしに分類できます。クラスを言語で差し替えられるのがゼロショットの肝です。</figcaption></figure>

## 視覚エンコーダとしての意義（現行 MLLM の土台）

ここまで見たとおり、CLIP は **言語を生成しません**。出力はあくまで埋め込み（とその類似度）です。それでも CLIP がその後の VLM 研究で決定的に重要なのは、対照学習を通じて画像エンコーダが **「言語と整合した視覚特徴」** を獲得するからです。CLIP-ViT の特徴空間では、近い意味の画像が近くに集まり、しかもその位置が言語側の意味付けと結びついています。

この性質は、画像を「LLM が解釈しやすい入力」へ変換するうえで理想的です。そのため LLaVA や BLIP-2、その後の多くの MLLM は、**事前学習済み CLIP-ViT を視覚エンコーダとして流用** し（しばしば凍結したまま）、その **パッチ特徴** を軽量な **コネクタ**（線形射影や MLP、Q-Former など）で LLM の埋め込み空間へ橋渡しします。CLIP が「画像→言語寄りの表現」までを担い、コネクタと LLM が「表現→自由な言語生成」を担う、という分業です。CLIP 単体ではできなかった対話的な説明や VQA は、この組み合わせで初めて成立します。

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="事前学習済みCLIP-ViTを凍結した視覚エンコーダとして使い、パッチ特徴をコネクタでLLMの埋め込み空間へ橋渡しし、LLMがテキストを生成する現行MLLMの構成図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <rect x="16" y="120" width="84" height="64" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="58" y="157" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">入力画像</text>
  <line x1="100" y1="152" x2="118" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="126,152 116,147 116,157" fill="#71717a"/>
  <rect x="128" y="108" width="128" height="88" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="192" y="142" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">CLIP-ViT</text>
  <text x="192" y="160" text-anchor="middle" font-size="10" fill="#155e75">言語整合した視覚特徴</text>
  <rect x="150" y="170" width="84" height="16" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="1.5"/>
  <text x="192" y="182" text-anchor="middle" font-size="10" font-weight="700" fill="#0e7490">多くは凍結</text>
  <line x1="256" y1="152" x2="274" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="282,152 272,147 272,157" fill="#71717a"/>
  <text x="298" y="140" text-anchor="middle" font-size="9" fill="#71717a">パッチ特徴</text>
  <rect x="300" y="108" width="120" height="88" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="360" y="142" text-anchor="middle" font-size="12" font-weight="700" fill="#3730a3">コネクタ</text>
  <text x="360" y="160" text-anchor="middle" font-size="10" fill="#3730a3">射影 / MLP / Q-Former</text>
  <rect x="318" y="170" width="84" height="16" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="1.5"/>
  <text x="360" y="182" text-anchor="middle" font-size="10" font-weight="700" fill="#4338ca">新規に学習</text>
  <line x1="420" y1="152" x2="438" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="446,152 436,147 436,157" fill="#71717a"/>
  <rect x="448" y="108" width="120" height="88" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="508" y="150" text-anchor="middle" font-size="13" font-weight="700" fill="#155e75">LLM</text>
  <text x="508" y="170" text-anchor="middle" font-size="10" fill="#155e75">言語生成を担当</text>
  <line x1="568" y1="152" x2="586" y2="152" stroke="#71717a" stroke-width="2"/>
  <polygon points="594,152 584,147 584,157" fill="#71717a"/>
  <rect x="596" y="120" width="108" height="64" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="650" y="150" text-anchor="middle" font-size="12" font-weight="700" fill="#166534">テキスト生成</text>
  <text x="650" y="168" text-anchor="middle" font-size="10" fill="#166534">説明 / 回答</text>
  <rect x="118" y="40" width="148" height="40" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="192" y="64" text-anchor="middle" font-size="11" font-weight="700" fill="#155e75">CLIP が事前学習</text>
  <line x1="192" y1="80" x2="192" y2="104" stroke="#0e7490" stroke-width="1.5" stroke-dasharray="4 3"/>
  <rect x="372" y="40" width="232" height="40" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="488" y="64" text-anchor="middle" font-size="11" font-weight="700" fill="#3730a3">MLLM 側で追加: コネクタ + LLM</text>
</svg><figcaption>CLIP が学習した <b>言語整合の視覚特徴</b>を、現行 MLLM は <b>CLIP-ViT を流用</b>して入力に使います。<b>コネクタ</b>がパッチ特徴を LLM 空間へ橋渡しし、<b>LLM が生成</b>を担当します。CLIP 自身は生成しませんが、視覚側の土台を提供しています。</figcaption></figure>

## まとめと、読解後に答えたい問い

- CLIP は **二重エンコーダ**（画像: ResNet 系/ViT、テキスト: Transformer）を、約 4 億の画像-テキストペア（WIT）上で **対照学習** だけで事前学習する。線形射影と L2 正規化で **共有埋め込み空間** を作る。
- 損失は **バッチ内 N×N 類似度行列** に対する **対称クロスエントロピー**。対応ペア（対角・正例 N 個）を上げ、非対応ペア（非対角・負例 N²−N 個）を下げる。温度 $\tau$（学習される log スカラ、$\exp(t)=1/\tau$）でロジットをスケールする。教師は「対角が正解」という並びだけで、人手ラベルは不要。
- **ゼロショット転移**: クラス名を「a photo of a {class}.」の文に変換し、その埋め込みを分類器の重みとして使う。画像埋め込みとのコサイン類似度が最大の文を選ぶだけで、再学習なしに多数タスクへ転移する。テキストエンコーダはクラス名から重みを生む **ハイパーネットワーク** とみなせる。多義語には弱く、プロンプト設計とアンサンブルが効く。
- **生成はしない**。だが言語と整合した視覚エンコーダ（CLIP-ViT）は、LLaVA・BLIP-2 など現行 MLLM の **視覚入力の土台** になっている。

読解後に自分の言葉で答えたい問い:

1. なぜ「キャプションを予測する（生成する）」のではなく「どのキャプションが対応するか当てる（対照）」ほうが、効率よく強い表現を学べるのか。生成タスクと対照タスクの難しさの違いから説明できるか。
2. 損失が画像→文と文→画像の **両方向で対称** なのはなぜか。片方向だけにすると何が崩れるか。
3. 温度 $\tau$ は類似度の分布の鋭さを制御する。$\tau$ が小さい（ロジットのスケールが大きい）とき・大きいとき、学習はどう変わるか。なぜ上限クリップが必要なのか。
4. ゼロショットで「クラス名→文埋め込み＝分類器の重み」と言えるのはなぜか。固定 softmax 分類器との本質的な違いは何か。
5. CLIP の特徴が「言語と整合している」とは具体的にどういう状態か。その性質が、後段の LLM に画像を渡すコネクタ設計をなぜ容易にするのか。
