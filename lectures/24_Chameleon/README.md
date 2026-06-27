# Chameleon — Mixed-Modal Early-Fusion Foundation Models

Chameleon（Meta FAIR, arXiv:2405.09818）は、画像とテキストを **最初から同じ土俵に乗せる** タイプのマルチモーダル基盤モデルである。これまで本サーベイで見てきた LLaVA 系のモデルは、画像を専用のビジョンエンコーダで処理し、その特徴量を projector（コネクタ）で言語モデルの埋め込み空間へ橋渡しする「late-fusion（後期融合）」の設計だった。Chameleon はこの発想を捨て、**画像もテキストも等しく離散トークンへ変換し、ただ1つのトランスフォーマで次トークン予測を行う「early-fusion（早期融合）」** をとる。

その結果、入力も出力もテキストと画像を任意の順序で混在させられる。たとえば「画像つきの質問に文章で答える（理解）」ことも、「文章のプロンプトから画像を生成する（生成）」ことも、さらには「文章と画像が交互に並ぶ文書まるごと」を生成することも、すべて同じ仕組みで扱える。本ページでは、この early-fusion がどういう設計で、なぜ素直に学習させると不安定になり、どう安定化したのかを順に読み解く。

なお Chameleon は LLaMa-2 のアーキテクチャ（RMSNorm・SwiGLU・RoPE）をベースにし、7B と 34B の2サイズが公開された。本ページでは性能評価には踏み込まず、**設計思想と安定化の工夫** に焦点を当てる。

## 全体像（early-fusion とは）

early-fusion の要点は「**コネクタが存在しない**」ことに尽きる。late-fusion ではビジョンエンコーダの出力をテキスト空間へ写すための projector という別ブランチが必要だったが、Chameleon では画像を最初に離散トークン列へ変換してしまうので、テキストのトークンと完全に同じ「語彙の一員」になる。あとは両者を1本の系列に並べ、テキスト言語モデルと同じ自己回帰トランスフォーマに流すだけでよい。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="テキストと画像をともに離散トークンへ変換し、単一の系列として1つのトランスフォーマに入力し次トークン予測を行う early-fusion の全体像を示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
<text x="360" y="28" font-size="16" font-weight="700" text-anchor="middle" fill="#18181b">early-fusion：すべてを離散トークンにして単一トランスフォーマへ</text>
<rect x="16" y="86" width="118" height="46" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="75" y="114" font-size="13" font-weight="700" text-anchor="middle" fill="#166534">テキスト入力</text>
<rect x="16" y="228" width="118" height="54" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="75" y="260" font-size="13" font-weight="700" text-anchor="middle" fill="#155e75">画像入力</text>
<line x1="134" y1="109" x2="166" y2="109" stroke="#71717a" stroke-width="2"/>
<polygon points="166,109 158,105 158,113" fill="#71717a"/>
<line x1="134" y1="255" x2="166" y2="255" stroke="#71717a" stroke-width="2"/>
<polygon points="166,255 158,251 158,259" fill="#71717a"/>
<rect x="168" y="86" width="112" height="46" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<text x="224" y="106" font-size="12" font-weight="700" text-anchor="middle" fill="#3730a3">BPE</text>
<text x="224" y="123" font-size="11" font-weight="700" text-anchor="middle" fill="#3730a3">トークナイザ</text>
<rect x="168" y="226" width="112" height="58" rx="6" fill="#ecfeff" stroke="#06b6d4" stroke-width="2"/>
<text x="224" y="250" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">VQ</text>
<text x="224" y="267" font-size="11" font-weight="700" text-anchor="middle" fill="#155e75">トークナイザ</text>
<line x1="280" y1="109" x2="300" y2="150" stroke="#71717a" stroke-width="2"/>
<polygon points="300,150 291,147 296,141" fill="#71717a"/>
<line x1="280" y1="255" x2="300" y2="220" stroke="#71717a" stroke-width="2"/>
<polygon points="300,220 296,229 291,223" fill="#71717a"/>
<text x="362" y="74" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">単一の系列（混在トークン）</text>
<rect x="300" y="84" width="124" height="206" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<rect x="312" y="96" width="100" height="22" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<rect x="312" y="124" width="100" height="22" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<rect x="312" y="152" width="100" height="22" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="312" y="180" width="100" height="22" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="312" y="208" width="100" height="22" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="312" y="236" width="100" height="22" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<line x1="424" y1="187" x2="450" y2="187" stroke="#71717a" stroke-width="2"/>
<polygon points="450,187 442,183 442,191" fill="#71717a"/>
<rect x="452" y="120" width="134" height="138" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="519" y="182" font-size="13" font-weight="700" text-anchor="middle" fill="#3730a3">単一</text>
<text x="519" y="200" font-size="13" font-weight="700" text-anchor="middle" fill="#3730a3">トランスフォーマ</text>
<text x="519" y="222" font-size="11" font-weight="700" text-anchor="middle" fill="#4338ca">次トークン予測</text>
<line x1="586" y1="187" x2="610" y2="187" stroke="#71717a" stroke-width="2"/>
<polygon points="610,187 602,183 602,191" fill="#71717a"/>
<rect x="612" y="132" width="96" height="110" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<rect x="624" y="144" width="72" height="20" rx="3" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<rect x="624" y="170" width="72" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="624" y="196" width="72" height="20" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="624" y="222" width="72" height="14" rx="3" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="660" y="262" font-size="11" font-weight="700" text-anchor="middle" fill="#18181b">出力トークン</text>
<text x="360" y="332" font-size="12" font-weight="700" text-anchor="middle" fill="#dc2626">projector も別ブランチも無い（最初から融合）</text>
</svg><figcaption>緑はテキストトークン、シアンは画像トークン。両者は同じ語彙の中で1本の系列にまとめられ、<b>要点</b>はコネクタを介さずに最初から1つのトランスフォーマで次トークンを予測する点である。理解も生成もこの同一機構で行う。</figcaption></figure>

この設計が効くのは、「全モダリティが最初から共有された表現空間にある」ためである。late-fusion では画像とテキストの相互作用が projector 以降の浅い層に限られがちだが、early-fusion では入力の最下層からテキストと画像のトークンが同じ自己注意で混ざり合う。その代わり、後述するように **混在学習が不安定になりやすい** という代償を払うことになる。

## 画像の離散トークン化（VQ）

early-fusion の生命線は「画像をどうやって離散トークンにするか」である。Chameleon は VQ（Vector Quantization, ベクトル量子化）方式の画像トークナイザを採用する。これは Gafni らの手法（Make-A-Scene 系）をベースに新規に学習したもので、連続値である画像を有限個の「コード」の並びへ写す。

仕組みは次の通り。まず CNN エンコーダが画像を空間的な潜在ベクトルの格子へ落とし込む。次に各潜在ベクトルを、あらかじめ学習しておいた **コードブック（約8192個の代表ベクトルの辞書）** の中で最も近いものに置き換える（最近傍量子化）。最後に、選ばれたコードの番号（インデックス）の並びが画像のトークン列となる。Chameleon では **512×512 の画像が1024個の離散トークン** に変換される（コードブックサイズは約8192）。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="512かける512の画像をCNNエンコーダで潜在格子に変換し、約8192個のコードブックで最近傍量子化して1024個の離散トークンに写すVQトークナイザの流れを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
<text x="360" y="28" font-size="16" font-weight="700" text-anchor="middle" fill="#18181b">VQ トークナイザ：画像を有限個のコードの並びへ</text>
<rect x="20" y="140" width="104" height="104" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="72" y="186" font-size="13" font-weight="700" text-anchor="middle" fill="#155e75">512×512</text>
<text x="72" y="206" font-size="13" font-weight="700" text-anchor="middle" fill="#155e75">画像</text>
<line x1="124" y1="192" x2="156" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="156,192 148,188 148,196" fill="#71717a"/>
<rect x="158" y="150" width="120" height="84" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="218" y="188" font-size="13" font-weight="700" text-anchor="middle" fill="#3730a3">CNN</text>
<text x="218" y="206" font-size="12" font-weight="700" text-anchor="middle" fill="#3730a3">エンコーダ</text>
<line x1="278" y1="192" x2="310" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="310,192 302,188 302,196" fill="#71717a"/>
<text x="356" y="138" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">潜在ベクトル格子</text>
<rect x="312" y="150" width="88" height="88" rx="4" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
<line x1="334" y1="150" x2="334" y2="238" stroke="#6366f1" stroke-width="1"/>
<line x1="356" y1="150" x2="356" y2="238" stroke="#6366f1" stroke-width="1"/>
<line x1="378" y1="150" x2="378" y2="238" stroke="#6366f1" stroke-width="1"/>
<line x1="312" y1="172" x2="400" y2="172" stroke="#6366f1" stroke-width="1"/>
<line x1="312" y1="194" x2="400" y2="194" stroke="#6366f1" stroke-width="1"/>
<line x1="312" y1="216" x2="400" y2="216" stroke="#6366f1" stroke-width="1"/>
<line x1="400" y1="192" x2="432" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="432,192 424,188 424,196" fill="#71717a"/>
<text x="490" y="120" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">コードブック 約8192</text>
<rect x="434" y="130" width="112" height="124" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<rect x="446" y="142" width="88" height="18" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="446" y="166" width="88" height="18" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<rect x="446" y="190" width="88" height="18" rx="3" fill="#06b6d4" stroke="#0e7490" stroke-width="2"/>
<rect x="446" y="214" width="88" height="18" rx="3" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="490" y="270" font-size="11" font-weight="700" text-anchor="middle" fill="#155e75">最近傍を選ぶ</text>
<line x1="546" y1="192" x2="578" y2="192" stroke="#71717a" stroke-width="2"/>
<polygon points="578,192 570,188 570,196" fill="#71717a"/>
<rect x="580" y="150" width="124" height="84" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="642" y="184" font-size="13" font-weight="700" text-anchor="middle" fill="#166534">1024 個の</text>
<text x="642" y="204" font-size="13" font-weight="700" text-anchor="middle" fill="#166534">離散トークン</text>
<text x="360" y="332" font-size="12" font-weight="700" text-anchor="middle" fill="#71717a">弱点：文字が多い画像（OCR用途）の復元は苦手</text>
</svg><figcaption>連続値の画像を、固定辞書（コードブック）のインデックス列へ落とし込むのが VQ である。<b>要点</b>は1枚の画像が固定長（1024個）のトークン列になり、テキストと同じ「離散シンボルの並び」として扱える点。なお文字の多い画像の復元は不得手で、これがモデルの上限を縛る。</figcaption></figure>

ここで重要なのは、VQ により画像が **固定長のトークンブロック** になることだ。テキストが可変長なのに対し、1枚の画像はつねに1024トークンに対応する。この性質は後述の生成時の扱い（画像トークンのブロック生成）にも効いてくる。一方で論文は、コードブックの表現力には限界があり、**大量の文字を含む画像の復元が苦手** であることを明記している。これは Chameleon が OCR 的なタスクで頭打ちになりうる原因として正直に述べられている。

## 単一語彙・単一トランスフォーマ

画像トークンが用意できたら、あとはテキストと混ぜて1本の系列にする。Chameleon は BPE トークナイザを新規学習し、その **語彙サイズを 65,536 とした。このうち 8192 個が画像コードブックのトークン** であり、残りがテキスト用のサブワードである。つまりテキストの単語片も画像のコードも、まったく区別なく「1つの語彙」に同居する。

学習目標は標準的な言語モデルと同じ自己回帰の次トークン予測である。系列 $x=(x_1,\dots,x_n)$ に対し、

$$p(x)=\prod_i p(x_i\mid x_{<i})$$

を最大化するだけでよい。トークンがテキストか画像かを問わず、ただ「次に来るトークン」を当て続ける。この単純さこそが early-fusion の強みで、**理解と生成が同じ目的関数の中で統一される**。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="テキストと画像のトークンが単一語彙の中で1本の系列に並び、画像開始と画像終了の特別トークンに挟まれた画像トークンも含めて左から右へ次トークン予測される様子を示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
<text x="360" y="28" font-size="16" font-weight="700" text-anchor="middle" fill="#18181b">単一語彙・単一系列で次トークン予測</text>
<rect x="60" y="56" width="600" height="40" rx="6" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<rect x="60" y="56" width="430" height="40" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<rect x="490" y="56" width="170" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="275" y="81" font-size="13" font-weight="700" text-anchor="middle" fill="#166534">テキスト BPE 語彙</text>
<text x="575" y="81" font-size="13" font-weight="700" text-anchor="middle" fill="#155e75">画像コード 8192</text>
<text x="360" y="118" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">単一語彙（合計 65,536）</text>
<g>
<rect x="24" y="186" width="76" height="40" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="62" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#166534">テキスト</text>
<rect x="108" y="186" width="76" height="40" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="146" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#166534">テキスト</text>
<rect x="192" y="186" width="76" height="40" rx="5" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="230" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">画像開始</text>
<rect x="276" y="186" width="76" height="40" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="314" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">画像</text>
<rect x="360" y="186" width="76" height="40" rx="5" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="398" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">画像</text>
<rect x="444" y="186" width="76" height="40" rx="5" fill="#e4e4e7" stroke="#71717a" stroke-width="2"/>
<text x="482" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">画像終了</text>
<rect x="528" y="186" width="76" height="40" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="566" y="211" font-size="12" font-weight="700" text-anchor="middle" fill="#166534">テキスト</text>
<rect x="612" y="186" width="76" height="40" rx="5" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="650" y="211" font-size="14" font-weight="700" text-anchor="middle" fill="#18181b">…</text>
</g>
<g stroke="#4338ca" stroke-width="2" fill="none">
<path d="M62 182 C 90 158, 118 158, 146 182"/>
<path d="M146 182 C 174 158, 202 158, 230 182"/>
<path d="M230 182 C 258 158, 286 158, 314 182"/>
<path d="M314 182 C 342 158, 370 158, 398 182"/>
<path d="M398 182 C 426 158, 454 158, 482 182"/>
<path d="M482 182 C 510 158, 538 158, 566 182"/>
</g>
<polygon points="146,182 139,176 148,173" fill="#4338ca"/>
<polygon points="230,182 223,176 232,173" fill="#4338ca"/>
<polygon points="314,182 307,176 316,173" fill="#4338ca"/>
<polygon points="398,182 391,176 400,173" fill="#4338ca"/>
<polygon points="482,182 475,176 484,173" fill="#4338ca"/>
<polygon points="566,182 559,176 568,173" fill="#4338ca"/>
<text x="356" y="262" font-size="12" font-weight="700" text-anchor="middle" fill="#4338ca">各位置で「次のトークン」を予測（曲線矢印）</text>
<text x="360" y="300" font-size="13" font-weight="700" text-anchor="middle" fill="#16a34a">理解：画像→テキスト</text>
<text x="360" y="324" font-size="13" font-weight="700" text-anchor="middle" fill="#0e7490">生成：テキスト→画像トークン→画像へ復元</text>
</svg><figcaption>テキスト片と画像コードが同じ語彙に同居し、特別トークンで画像ブロックの境界を示す。<b>要点</b>は同一の自己回帰目的だけで、画像つき質問への回答（理解）と文章からの画像生成（生成）を区別なく扱える点。画像を出すときは画像トークンを並べ、最後に VQ デコーダで画素へ戻す。</figcaption></figure>

「理解」では、画像トークンを文脈に含めたままテキストトークンを生成すればよい（画像を見て答える）。「生成」では、`画像開始` の特別トークンに続けて画像コードのトークンを並べ、`画像終了` で閉じる。生成された画像トークン列を VQ デコーダに通すと画素へ戻る。系列の途中で何度でもこの切り替えが起きるため、文章と画像が交互に並ぶ文書をそのまま生成できる。なお生成時は、いま画像とテキストどちらのモードかでデコードを切り替える必要があり、画像は固定長ブロックとして扱われる（推論実装上の工夫が必要になる点も論文は述べている）。

## 混在学習の安定化

ここが Chameleon の技術的な核心である。テキストと画像という **エントロピーの大きく異なるモダリティで重みを共有** すると、学習は素直には進まない。論文は、8B パラメータ・1T トークンを超えるスケールで、学習の中盤から終盤にかけて損失が発散する現象に悩まされたと報告している。

原因はソフトマックスの **平行移動不変性**（$\mathrm{softmax}(z)=\mathrm{softmax}(z+c)$）にある。重みを共有する各モダリティが、互いに「競合」してわずかずつノルムを押し上げていき、最終的に bf16 の表現可能範囲を外れて発散する。論文はこれを **logit drift（ロジットのドリフト）問題** と呼ぶ。実際、画像生成を外したアブレーションでは発散しなかったことから、混在こそが不安定性の引き金だと突き止めている。出力ノルムの暴走は、将来の損失発散を予告する強いシグナルになっていた。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="ソフトマックス入力のノルム増大を抑えるQK-Normと、SwiGLUを含むFFNのノルム増大を抑える正規化の再配置という二つの安定化策を、出力ノルムの発散と抑制の曲線とともに示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
<rect x="0" y="0" width="720" height="360" fill="#ffffff"/>
<text x="360" y="28" font-size="16" font-weight="700" text-anchor="middle" fill="#18181b">混在学習の安定化：ノルム増大を二箇所で抑える</text>
<rect x="24" y="58" width="320" height="120" rx="8" fill="#eef2ff" stroke="#4338ca" stroke-width="2"/>
<text x="184" y="82" font-size="14" font-weight="700" text-anchor="middle" fill="#3730a3">QK-Norm</text>
<text x="184" y="106" font-size="12" font-weight="700" text-anchor="middle" fill="#3730a3">注意機構の Q・K に LayerNorm</text>
<text x="184" y="126" font-size="12" font-weight="700" text-anchor="middle" fill="#3730a3">→ ソフトマックス入力の</text>
<text x="184" y="144" font-size="12" font-weight="700" text-anchor="middle" fill="#3730a3">ノルム増大を直接抑制</text>
<text x="184" y="166" font-size="11" font-weight="700" text-anchor="middle" fill="#4338ca">7B では dropout と z-loss も併用</text>
<rect x="24" y="196" width="320" height="120" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
<text x="184" y="220" font-size="14" font-weight="700" text-anchor="middle" fill="#155e75">正規化の再配置（post-norm 寄り）</text>
<text x="184" y="244" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">残差の外側で正規化（Swin 流）</text>
<text x="184" y="264" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">→ SwiGLU を含む FFN の</text>
<text x="184" y="282" font-size="12" font-weight="700" text-anchor="middle" fill="#155e75">ノルム増大を抑える</text>
<text x="184" y="304" font-size="11" font-weight="700" text-anchor="middle" fill="#0e7490">34B の安定化に必須だった</text>
<rect x="380" y="58" width="316" height="258" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
<text x="538" y="82" font-size="13" font-weight="700" text-anchor="middle" fill="#18181b">出力ノルムの推移（概念図）</text>
<line x1="420" y1="280" x2="668" y2="280" stroke="#18181b" stroke-width="2"/>
<line x1="420" y1="280" x2="420" y2="104" stroke="#18181b" stroke-width="2"/>
<text x="544" y="304" font-size="12" font-weight="700" text-anchor="middle" fill="#18181b">学習ステップ</text>
<path d="M420 268 C 520 262, 600 230, 660 116" fill="none" stroke="#dc2626" stroke-width="3"/>
<path d="M420 268 C 520 262, 600 258, 668 252" fill="none" stroke="#16a34a" stroke-width="3"/>
<rect x="446" y="118" width="14" height="14" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="466" y="130" font-size="12" font-weight="700" text-anchor="start" fill="#dc2626">対策なし：発散</text>
<rect x="446" y="142" width="14" height="14" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="466" y="154" font-size="12" font-weight="700" text-anchor="start" fill="#16a34a">対策あり：安定</text>
</svg><figcaption>ソフトマックスは入力に定数を足しても不変なため、共有重み下では各モダリティがノルムを押し上げ合い、やがて bf16 の範囲を超えて発散する。<b>要点</b>は注意（QK-Norm）と FFN（正規化の再配置）の二箇所でノルム増大を抑えることで、出力ノルムの暴走を防ぎ学習を安定させた点。</figcaption></figure>

具体的な処方箋は次の通りである。

- **QK-Norm（query-key normalization）**: ソフトマックスは注意機構と最終ロジットの2か所に現れる。Chameleon はまず注意の内部に手を入れ、クエリ $Q$ とキー $K$ に LayerNorm をかけてからスコアを計算する。これにより、ソフトマックスへ入る値のノルム増大を直接抑え込む。7B ではこれが安定化に必須だった。
- **正規化の再配置**: QK-Norm だけでは 34B を安定化できなかった。そこで Swin Transformer 流の正規化配置（残差の外側で正規化する post-norm 寄りの形）を採り、特に乗算的な SwiGLU を含む FFN ブロックのノルム増大を抑えた。LLaMa-2 と Chameleon-34B の正規化配置を並べると違いが分かる。

$$
\begin{aligned}
\text{Chameleon-34B:}\quad & h = x + \mathrm{attn\_norm}\big(\mathrm{attention}(x)\big)\\
& \mathrm{out} = h + \mathrm{ffn\_norm}\big(\mathrm{feed\_forward}(h)\big)\\[4pt]
\text{LLaMa-2:}\quad & h = x + \mathrm{attention}\big(\mathrm{attn\_norm}(x)\big)\\
& \mathrm{out} = h + \mathrm{feed\_forward}\big(\mathrm{ffn\_norm}(h)\big)
\end{aligned}
$$

- **dropout と z-loss**: 7B では QK-Norm に加えて、注意と FFN の後段に dropout を入れることと、最終ソフトマックスの分配関数 $Z=\sum_i e^{x_i}$ を正則化する z-loss（損失に $10^{-5}\,\log^2 Z$ を加える）を併用して安定させた。一方 34B では正規化の再配置と dropout の相性が悪く、dropout を使わず正規化再配置と z-loss で安定化している。

要するに、early-fusion で「重みを完全共有する」という強い前提を貫いた代償として、Chameleon は **ノルム増大を注意と FFN の二箇所で押さえ込む** という地道な工夫を積み重ねた。これらは派手な新規モジュールではないが、混在モダリティを大規模に学習するための実務的な勘所であり、本モデルの最も重要な貢献の一つである。

## まとめと、読解後に答えたい問い

本ページの要点を整理する。

- Chameleon は **early-fusion・トークンベース** の基盤モデルで、画像もテキストも離散トークンへ変換し、コネクタを介さず1つのトランスフォーマで次トークン予測する。
- 画像は **VQ トークナイザ** で離散化する（512×512 → 1024トークン、コードブック約8192）。文字の多い画像の復元は苦手という限界がある。
- テキストと画像は **単一語彙（65,536、うち8192が画像コード）・単一系列** にまとめられ、$p(x)=\prod_i p(x_i\mid x_{<i})$ という同一目的のもとで理解も生成も統一される。
- 混在学習は **logit drift** で不安定になりやすく、**QK-Norm・正規化の再配置・dropout・z-loss** によってノルム増大を抑え込み安定化した。

読解後、次の問いに自分の言葉で答えられるかを確認してほしい。

1. late-fusion（LLaVA 系の projector 方式）と early-fusion（Chameleon）は、画像とテキストの相互作用の「深さ」という観点でどう違うか。それぞれの長所と短所は何か。
2. なぜ画像を「離散トークン」にする必要があったのか。連続特徴量のまま系列に混ぜる方式と比べて、何が得られ、何を失うか。
3. logit drift 問題はなぜ「単一モダリティでは顕在化しにくく、混在で顕在化する」のか。ソフトマックスの平行移動不変性と重み共有の関係から説明できるか。
4. QK-Norm と「正規化の再配置」は、それぞれトランスフォーマのどの部分のノルム増大を抑えているか。なぜ2か所への対策が必要だったか。
5. VQ トークナイザのコードブックサイズや1画像あたりのトークン数を変えると、画像の表現力・系列長・学習安定性にどんなトレードオフが生じると考えられるか。
