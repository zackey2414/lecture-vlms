# InternVL3 — Native Multimodal Pretraining と Test-Time Recipe

InternVL3（*Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models*, arXiv:2504.10479, OpenGVLab 2025）は、InternVL シリーズの最新世代です。最大の主張は **native multimodal pretraining** ―― 「テキスト専用 LLM を後から視覚に適応させる」従来の段階的パイプラインをやめ、**最初から視覚と言語を 1 段で同時に**事前学習するという設計の転換にあります。さらに、可変解像度に対応する位置符号化 **V2PE**、後処理アライメントの **MPO（Mixed Preference Optimization）**、推論時の **test-time scaling** を組み合わせ、InternVL3-78B は MMMU で 72.2 という、当時のオープンソース MLLM の新水準を達成しました。本ページは、InternVL/LLaVA/Qwen 系を既習の読者が、この「同時学習」と先進レシピを腹落ちさせることを狙います。

## 全体像（まず一枚で）

アーキテクチャ自体は InternVL 系の **ViT–MLP–LLM** を踏襲します。画像を 448×448 のタイルに分割し、**InternViT**（300M または 6B）で符号化、**2 層 MLP** で言語空間へ写像し、**LLM**（Qwen2.5 系または InternLM3-8B）が自己回帰で生成します。pixel unshuffle により視覚トークンを 1/4 に圧縮し、448×448 タイル 1 枚あたり 256 トークンに抑えます。新しさは「箱の並び」ではなく、**それらの箱をどう学習させるか**にあります。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="InternVL3 の ViT-MLP-LLM 構成。入力画像をタイル分割し InternViT で符号化、2層MLPで言語空間へ写像、LLMが生成する。ViT・MLP・LLM の全パラメータをネイティブ事前学習で同時最適化し、V2PEで視覚トークンの位置を緩やかに進める" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="360" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ViT–MLP–LLM 構成（InternVL 系を踏襲）</text><rect x="24" y="66" width="92" height="62" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="70" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">入力画像</text><text x="70" y="111" text-anchor="middle" font-size="10" fill="#0e7490">448²タイル分割</text><line x1="118" y1="97" x2="158" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="166,97 155,92 155,102" fill="#71717a"/><rect x="168" y="66" width="118" height="62" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="227" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">InternViT</text><text x="227" y="111" text-anchor="middle" font-size="10" fill="#4338ca">300M / 6B</text><line x1="288" y1="97" x2="320" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="328,97 317,92 317,102" fill="#71717a"/><rect x="330" y="66" width="96" height="62" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="378" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">2層MLP</text><text x="378" y="111" text-anchor="middle" font-size="10" fill="#4338ca">視覚→言語</text><line x1="428" y1="97" x2="460" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="468,97 457,92 457,102" fill="#71717a"/><rect x="470" y="66" width="140" height="62" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="540" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">LLM</text><text x="540" y="111" text-anchor="middle" font-size="10" fill="#4338ca">Qwen2.5 / InternLM3-8B</text><line x1="612" y1="97" x2="640" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="648,97 637,92 637,102" fill="#71717a"/><rect x="650" y="66" width="58" height="62" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="679" y="101" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">テキスト</text><line x1="227" y1="128" x2="227" y2="168" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/><line x1="378" y1="128" x2="378" y2="168" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/><line x1="540" y1="128" x2="540" y2="168" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/><rect x="168" y="168" width="442" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="389" y="188" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">native multimodal pretraining</text><text x="389" y="206" text-anchor="middle" font-size="11" fill="#15803d">ViT・MLP・LLM の全パラメータを 1 段で同時最適化</text><rect x="168" y="240" width="442" height="46" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="389" y="260" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">V2PE：視覚トークンの位置を δ&lt;1 で緩やかに進める</text><text x="389" y="278" text-anchor="middle" font-size="11" fill="#4338ca">長い多モーダル文脈を位置窓を浪費せず収める</text><text x="360" y="318" text-anchor="middle" font-size="11" fill="#71717a">後処理：SFT → MPO ／ 推論時：Best-of-N（VisualPRM）</text></svg><figcaption>構成は ViT–MLP–LLM のまま。差分は <b>学習のさせ方</b>にあり、全パラメータの同時学習・V2PE・後処理レシピが効きます。</figcaption></figure>

InternVL2.5 を土台に、各スケール（1B〜78B）で性能を底上げしています。前世代との設計差を整理します。

| 観点 | InternVL / InternVL2.5（従来） | InternVL3 |
|---|---|---|
| 事前学習の段階 | テキスト専用 LLM 事前学習 → 視覚整列 → 指示チューニングの多段 | **1 段の native multimodal pretraining** |
| パラメータ更新 | 整列段では MLP 中心、LLM は凍結/部分微調整が一般的 | **ViT・MLP・LLM を同時に更新** |
| 視覚位置符号化 | 通常の $+1$ 刻み（V2PE で $\delta = 1$ に相当） | **V2PE（$\delta < 1$）** で可変解像度・長文脈に対応 |
| 後処理 | SFT 中心 | **SFT → MPO（選好最適化）** |
| 推論時 | 単純デコード | **test-time scaling（Best-of-$N$＋VisualPRM）** |
| 公開範囲 | 重み中心 | **学習データと重みの双方を公開** |

> LLM 部分は **指示チューニング済みではなく事前学習済みのベースモデル**（Qwen2.5 系・InternLM3-8B）で初期化される点に注意。指示能力は後段の SFT/MPO で付与します。

スケール展開は 1B から 78B まで 7 種で、視覚エンコーダは小〜中規模が InternViT-300M、大規模（38B/78B）が InternViT-6B、言語側は主に Qwen2.5 系（9B のみ InternLM3-8B）という対応です（PDF Table 1）。

| モデル | 総パラメータ | 視覚エンコーダ | 言語モデル |
|---|---|---|---|
| InternVL3-1B | 0.9B | InternViT-300M | Qwen2.5-0.5B |
| InternVL3-2B | 1.9B | InternViT-300M | Qwen2.5-1.5B |
| InternVL3-8B | 8.1B | InternViT-300M | Qwen2.5-7B |
| InternVL3-9B | 9.2B | InternViT-300M | InternLM3-8B |
| InternVL3-14B | 15.1B | InternViT-300M | Qwen2.5-14B |
| InternVL3-38B | 38.4B | InternViT-6B | Qwen2.5-32B |
| InternVL3-78B | 78.4B | InternViT-6B | Qwen2.5-72B |

## Native Multimodal Pretraining

従来の MLLM は「テキスト専用 LLM をまず作り、後から視覚モダリティを足す」ため、適応時にアライメントの綻びや、言語能力を壊さないための凍結・段階スケジュールといった工夫が必要でした。InternVL3 はこの順序を捨て、**言語事前学習とマルチモーダル整列を 1 つの事前学習段に統合**します。テキストコーパスと、画像–テキスト・動画–テキスト・交互（interleaved）列のマルチモーダルデータを混ぜて流し込み、言語能力と視覚言語能力を**同時に**獲得させます。

<figure class="lec-fig"><svg viewBox="0 0 720 360" role="img" aria-label="従来は①テキスト専用LLM事前学習②視覚整列③指示チューニングと段階的に後付け適応するのに対し、InternVL3は単一の事前学習段でテキスト・画像・動画を交互に入力し全パラメータを同時最適化するnative multimodal pretrainingを行う対比図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="30" y="40" font-size="13" font-weight="700" fill="#dc2626">従来：後付け適応（multi-stage）</text><rect x="30" y="56" width="172" height="60" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="116" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">① テキスト専用</text><text x="116" y="98" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">LLM 事前学習</text><line x1="202" y1="86" x2="232" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="240,86 229,81 229,91" fill="#71717a"/><rect x="242" y="56" width="172" height="60" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/><text x="328" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">② 視覚へ整列</text><text x="328" y="98" text-anchor="middle" font-size="11" fill="#dc2626">MLP warmup・凍結</text><line x1="414" y1="86" x2="444" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="452,86 441,81 441,91" fill="#71717a"/><rect x="454" y="56" width="172" height="60" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="540" y="80" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">③ 指示チューニング</text><text x="540" y="98" text-anchor="middle" font-size="11" fill="#18181b">アライメント課題が残存</text><line x1="360" y1="150" x2="360" y2="180" stroke="#71717a" stroke-width="2.5"/><polygon points="360,190 353,178 367,178" fill="#71717a"/><text x="392" y="174" font-size="12" font-weight="700" fill="#16a34a">InternVL3 は 1 段に統合</text><text x="30" y="222" font-size="13" font-weight="700" fill="#15803d">InternVL3：native multimodal pretraining（単一段）</text><rect x="30" y="234" width="596" height="96" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><rect x="48" y="252" width="156" height="38" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="126" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">テキストコーパス</text><rect x="216" y="252" width="156" height="38" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="294" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">画像–テキスト</text><rect x="384" y="252" width="224" height="38" rx="6" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="496" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">動画–テキスト / 交互列</text><text x="328" y="314" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">全パラメータを同時更新・損失はテキストトークンのみ（視覚は文脈）</text></svg><figcaption>従来の <b>後付け適応</b> に対し、InternVL3 は <b>単一の事前学習段</b> でテキスト・画像・動画を交互に学びます。言語:多モーダル ≈ 1:3、計およそ 200B トークン（言語 50B＋多モーダル 150B）。</figcaption></figure>

学習の定式化は素直な左から右への自己回帰ですが、**損失計算をテキストトークンに限定**するのが要点です。視覚トークンは予測対象ではなく「条件付けの文脈」として働き、勾配はテキスト予測に有用な形でマルチモーダル情報を埋め込ませます。トークン長による勾配バイアスを抑えるため、重み付けには **square averaging**（$w_i \propto 1/\sqrt{l}$）を採用します。

さらに **Joint Parameter Optimization** として、整列段で一部を凍結する従来流儀と異なり、**ViT・MLP・LLM の全層を同時に最適化**します。これにより、テキスト表現と視覚特徴が「同じ釜」で協調して形成され、別途のブリッジモジュールや後段のモデル間整列を要しません。データ混合は経験的に **言語:多モーダル $\approx 1:3$** が最良で、総計はおよそ 200B トークン（言語 50B＋多モーダル 150B）でした。

ネイティブ事前学習の核心（PDF 2.2 節）

- 目的関数：標準の自己回帰（左→右）
- 損失対象：テキストトークンのみ（視覚トークンは条件付け文脈）
- 重み付け：square averaging で長短応答の勾配バイアスを緩和
- 更新範囲：ViT + MLP + LLM の全パラメータを同時に
- データ比：言語 : 多モーダル $\approx 1:3$（計 約200B token）

なお実装上は、ゼロから訓練する代わりに ViT と LLM を**事前学習済み重みで初期化**して計算コストを抑えています。アブレーション（PDF 3.14 節, Fig.3）では、InternVL2-8B の MLP warmup 段をネイティブ事前学習に置き換えるだけで、多段学習版に匹敵する多モーダル性能が得られ、さらに指示チューニングを足すと上振れすることが示されています。

## 可変解像度と位置符号化

ネイティブ/可変解像度の多モーダル入力では、高解像画像や長尺動画で視覚トークン数が膨れ上がり、通常の「全トークン +1 刻み」では位置インデックスが急速に伸びて、LLM の有効文脈窓を食い潰します。InternVL3 は **V2PE（Variable Visual Position Encoding）** を採用し、視覚トークンに対しては **1 より小さい増分 $\delta$** で位置を進めます。

位置インデックスは型に応じて再帰的に決まります。

$$p_i = p_{i-1} + 1 \quad (x_i\ \text{がテキストトークン})$$

$$p_i = p_{i-1} + \delta \quad (x_i\ \text{が視覚トークン},\ \delta < 1)$$

$$\delta \in \left\{ 1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{16}, \tfrac{1}{32}, \tfrac{1}{64}, \tfrac{1}{128}, \tfrac{1}{256} \right\}$$

- 学習時：画像ごとに $\delta$ をランダム選択（画像内では一定に保つ）
- 推論時：入力系列長に応じて $\delta$ を柔軟に選び、位置を有効文脈内に収める
- $\delta = 1$ のとき InternVL2.5 と同じ通常の位置符号化に戻る

<figure class="lec-fig"><svg viewBox="0 0 720 320" role="img" aria-label="同じトークン列（テキスト2個・視覚6個・テキスト2個）に対し、従来のδ=1では位置が0から9まで等間隔に伸びるのに対し、V2PEのδ=1/4では視覚トークンの位置が1.25から2.5へと緩やかにしか進まず、視覚ブロックが位置窓を浪費しないことを示す対比図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="360" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">同じトークン列に与える位置インデックスの違い</text><rect x="40" y="48" width="48" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="64" y="73" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">文</text><rect x="92" y="48" width="48" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="116" y="73" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">文</text><rect x="144" y="48" width="256" height="40" rx="6" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="272" y="73" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">視覚トークン × 6</text><rect x="404" y="48" width="48" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="428" y="73" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">文</text><rect x="456" y="48" width="48" height="40" rx="6" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="480" y="73" text-anchor="middle" font-size="13" font-weight="700" fill="#4338ca">文</text><text x="40" y="150" font-size="12" font-weight="700" fill="#dc2626">従来 δ=1</text><line x1="40" y1="170" x2="600" y2="170" stroke="#71717a" stroke-width="2"/><polygon points="608,170 597,165 597,175" fill="#71717a"/><text x="64" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">0</text><text x="116" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">1</text><text x="170" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">2</text><text x="272" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">…</text><text x="374" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">7</text><text x="428" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">8</text><text x="480" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">9</text><text x="40" y="234" font-size="12" font-weight="700" fill="#15803d">V2PE δ=¼</text><line x1="40" y1="254" x2="600" y2="254" stroke="#71717a" stroke-width="2"/><polygon points="608,254 597,249 597,259" fill="#71717a"/><text x="64" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">0</text><text x="116" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">1</text><text x="170" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">1.25</text><text x="272" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">…</text><text x="374" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">2.5</text><text x="428" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">3.5</text><text x="480" y="274" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">4.5</text><text x="660" y="190" text-anchor="middle" font-size="11" fill="#71717a">位置を</text><text x="660" y="206" text-anchor="middle" font-size="11" fill="#71717a">大量消費</text><text x="660" y="274" text-anchor="middle" font-size="11" fill="#15803d">圧縮</text></svg><figcaption>視覚 6 トークンが従来は位置 2〜7 を占有するのに対し、<b>δ=¼ では 1.25〜2.5 に圧縮</b>。長文脈でも位置窓を浪費しません。</figcaption></figure>

アブレーション（PDF Table 12, InternVL3-8B）では、V2PE の導入が多くの指標で性能を押し上げ、しかも短文脈中心のタスクでも**中程度に小さい $\delta$（$1/4$ 付近）**が好成績でした。論文本文では公平比較のため Table 12 以外の結果は $\delta = 1$ に固定しています。視覚トークンの位置を「テキストと同格に 1 ずつ進める」必然性はなく、**モダリティごとに位置の進み方を変えてよい**という発想が V2PE の肝です。

## 学習・test-timeレシピ

ネイティブ事前学習の後、InternVL3 は 2 段の後処理 ―― **SFT → MPO** ―― を行い、さらに推論時に **test-time scaling** を載せます。

**SFT** では、ランダム JPEG 圧縮・square 損失再重み付け・マルチモーダルデータパッキング（InternVL2.5 由来）を踏襲しつつ、ツール使用・3D シーン・GUI 操作・科学図表・創作・多モーダル推論まで**より高品質で多様な**データへ拡張します（学習サンプルは 16.3M → 21.7M）。

**MPO（Mixed Preference Optimization）** は、学習時は正解トークンで条件付けされるのに推論時は自分の出力に条件付けられるという**分布シフト**が CoT 推論を劣化させる問題に対処します。正例だけでなく**負例**からも監督を与え、応答分布を正解側に寄せます。目的関数は 3 つの損失の混合です。

| 損失 | 役割 | 実体 |
|---|---|---|
| 選好損失 $L_p$ | 採用応答 vs 棄却応答の**相対**選好を学ぶ | DPO 損失 |
| 品質損失 $L_q$ | 各応答の**絶対**品質（良し悪し）を学ぶ | BCO 損失（採用 $L_q^+$＋棄却 $L_q^-$） |
| 生成損失 $L_g$ | 採用応答の**生成過程**を学ぶ | LM 損失 |

$$L = w_p L_p + w_q L_q + w_g L_g$$

MPO 用の選好ペアは MMPR v1.2 から構築し約 300K サンプル、SFT データの部分集合であるため、**改善はアルゴリズム由来**（データ増ではない）と論文は強調します。Table 13 では MPO により 78B/38B でそれぞれ **+4.1 / +4.5 ポイント**の推論性能向上が報告されています。

**test-time scaling** は、推論時に複数候補を生成して最良を選ぶ **Best-of-$N$** 戦略です。批評器（critic）には **VisualPRM-8B**（Visual Process Reward Model）を用い、解答を**ステップごとに採点**して全体スコアを集約、最良候補を選びます。MathVerse（Vision-Only）では Best-of-8 で 38B/78B にそれぞれ約 +6.0 / +3.2 ポイントの効果が示され、1B/2B のような小型でも恩恵が出ます。

<figure class="lec-fig"><svg viewBox="0 0 720 340" role="img" aria-label="学習レシピはネイティブ事前学習からSFT、さらにMPO（選好DPO・品質BCO・生成LMの3損失の混合）へ進む。推論時はtest-time scalingとして、質問と画像からN個の応答候補を生成し、VisualPRM-8Bが各ステップを採点してBest-of-Nで最終回答を選ぶ流れを示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="30" y="36" font-size="13" font-weight="700" fill="#4338ca">学習レシピ（後処理アライメント）</text><rect x="24" y="50" width="150" height="54" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="99" y="73" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">native</text><text x="99" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">pretraining</text><line x1="174" y1="77" x2="202" y2="77" stroke="#71717a" stroke-width="2"/><polygon points="210,77 199,72 199,82" fill="#71717a"/><rect x="212" y="50" width="120" height="54" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/><text x="272" y="73" text-anchor="middle" font-size="12" font-weight="700" fill="#4338ca">SFT</text><text x="272" y="90" text-anchor="middle" font-size="10" fill="#4338ca">高品質・多様データ</text><line x1="332" y1="77" x2="360" y2="77" stroke="#71717a" stroke-width="2"/><polygon points="368,77 357,72 357,82" fill="#71717a"/><rect x="370" y="50" width="140" height="54" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="440" y="73" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">MPO</text><text x="440" y="90" text-anchor="middle" font-size="10" fill="#15803d">正例＋負例で選好最適化</text><line x1="440" y1="104" x2="440" y2="124" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="4 3"/><rect x="212" y="124" width="476" height="40" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="450" y="149" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">L = w_p·選好(DPO) + w_q·品質(BCO) + w_g·生成(LM)</text><line x1="30" y1="190" x2="690" y2="190" stroke="#e4e4e7" stroke-width="2"/><text x="30" y="216" font-size="13" font-weight="700" fill="#0e7490">推論時（test-time scaling：Best-of-N）</text><rect x="24" y="232" width="120" height="58" rx="8" fill="#ecfeff" stroke="#0e7490" stroke-width="2"/><text x="84" y="265" text-anchor="middle" font-size="12" font-weight="700" fill="#0e7490">質問 ＋ 画像</text><line x1="144" y1="261" x2="172" y2="261" stroke="#71717a" stroke-width="2"/><polygon points="180,261 169,256 169,266" fill="#71717a"/><rect x="182" y="232" width="140" height="58" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/><text x="252" y="258" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">N 個の応答候補</text><text x="252" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#4338ca">を生成</text><line x1="322" y1="261" x2="350" y2="261" stroke="#71717a" stroke-width="2"/><polygon points="358,261 347,256 347,266" fill="#71717a"/><rect x="360" y="232" width="170" height="58" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/><text x="445" y="258" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">VisualPRM-8B が</text><text x="445" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#0e7490">各ステップを採点</text><line x1="530" y1="261" x2="558" y2="261" stroke="#71717a" stroke-width="2"/><polygon points="566,261 555,256 555,266" fill="#71717a"/><rect x="568" y="232" width="128" height="58" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/><text x="632" y="258" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">Best-of-N で</text><text x="632" y="276" text-anchor="middle" font-size="11" font-weight="700" fill="#15803d">最終回答を選択</text></svg><figcaption>学習側は <b>SFT→MPO（3 損失の混合）</b>、推論側は <b>VisualPRM による Best-of-N</b>。学習レシピとテスト時レシピを別軸で重ねます。</figcaption></figure>

インフラ面では、ZeRO 由来の **InternEVO** を拡張し、ViT/MLP/LLM を分離してシャーディング、最大 32K トークン系列に対応、InternVL2.5 比で 50%〜200% の学習高速化を報告しています。総じて InternVL3 は「同時学習で土台を作り、選好最適化で整え、推論時スケーリングで仕上げる」という多層レシピを束ねたモデルです。

## まとめと、読解後に答えたい問い

- InternVL3 の核心は **native multimodal pretraining**：テキスト専用 LLM を後付け適応するのではなく、**1 段で視覚＋言語を同時に**学び、ViT・MLP・LLM の全パラメータを同時最適化する（損失はテキストトークンのみ、言語:多モーダル $\approx 1:3$、約 200B トークン）。
- **V2PE** は視覚トークンの位置を $\delta < 1$ で緩やかに進め、可変解像度・長文脈を有効文脈窓に収める。$\delta = 1$ で従来法に一致する後方互換設計。
- 後処理は **SFT → MPO**（選好 DPO＋品質 BCO＋生成 LM の混合損失）、推論は **VisualPRM による Best-of-$N$** の test-time scaling。学習軸とテスト時軸を別々に重ねて底上げする。
- InternVL3-78B は MMMU 72.2 などでオープンソースの新水準に達し、学習データと重みの双方を公開。

読解後に答えたい問い:

1. 「損失をテキストトークンのみに限定する」設計は、視覚特徴の質にどう影響するか。視覚側にも損失を置く設計（対照学習やマスク再構成）と比べた利点・欠点は何か。
2. V2PE の $\delta$ をなぜ画像内では一定に保つのか。$\delta$ を可変にする/位置を非線形に進める拡張は、相対位置関係をどこまで壊さずに長文脈を稼げるか。
3. MPO の 3 損失（選好・品質・生成）は、それぞれどの失敗モード（分布シフト・自信過剰・崩れた生成）を打ち消しているか。1 つ抜くと何が起きるか。
4. test-time scaling の Best-of-$N$ は批評器 VisualPRM の質に依存する。批評器自体の幻覚やバイアスは、最終精度をどこで頭打ちにするか。
