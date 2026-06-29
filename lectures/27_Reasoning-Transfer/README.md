# 推論能力の抽出・移植 — CoT蒸留 / reasoning vector / RL(R1系)

大規模なMLLM（とそのベースとなるLLM）は高い「推論能力」をもつが、推論の良さと推論時のコストは表裏一体で、現場ではより小さく速いモデルを使いたい。そこで生まれるのが「**推論能力だけを能力として切り出し、別の小型/高速モデルへ移植する**」という問題設定である。本ページでは、この移植を狙う研究を **A. CoT蒸留 / B. 推論ベクトル・steering / C. model merging / D. RL(R1系)** の 4 系統に整理してサーベイする。最も実証が厚いのは **A. CoT（根拠）蒸留** であり、ベクトル系（B）は概念としては魅力的だが、定量証拠の多くがテキストLLMで、**VLMでの直接実証は限定的**である点を最初に強調しておく。各手法には arXiv ID を併記し、確証度（高／中）と「テキストLLM実証か VLM実証か」を区別して述べる。

## 全体像（4系統の地図）

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="大規模MLLMの推論能力をCoT蒸留・推論ベクトル・model merging・RLの4系統で小型モデルへ移植する全体像" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="rt1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <rect x="16" y="136" width="130" height="88" rx="10" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="81" y="170" text-anchor="middle" font-size="14" font-weight="700" fill="#3730a3"><tspan x="81">大規模</tspan><tspan x="81" dy="18">MLLM / LLM</tspan></text>
  <text x="81" y="212" text-anchor="middle" font-size="11" fill="#4338ca">高い推論能力</text>
  <rect x="266" y="20" width="220" height="56" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="376" y="42" text-anchor="middle" font-size="13.5" font-weight="700" fill="#166534">A. CoT／根拠(rationale)蒸留</text>
  <text x="376" y="62" text-anchor="middle" font-size="11" fill="#16a34a">確度=高・VLM直接実証あり</text>
  <rect x="266" y="90" width="220" height="56" rx="9" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="376" y="112" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3730a3">B. 推論ベクトル・steering</text>
  <text x="376" y="132" text-anchor="middle" font-size="11" fill="#4338ca">確度=高・主にテキストLLM</text>
  <rect x="266" y="160" width="220" height="56" rx="9" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="376" y="182" text-anchor="middle" font-size="13.5" font-weight="700" fill="#155e75">C. model merging / 整列</text>
  <text x="376" y="202" text-anchor="middle" font-size="11" fill="#0e7490">確度=中</text>
  <rect x="266" y="230" width="220" height="56" rx="9" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="376" y="252" text-anchor="middle" font-size="13.5" font-weight="700" fill="#b91c1c">D. RL（GRPO / R1系）</text>
  <text x="376" y="272" text-anchor="middle" font-size="11" fill="#dc2626">確度=高・教師づくりに有効</text>
  <rect x="566" y="132" width="138" height="96" rx="10" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="635" y="170" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b"><tspan x="635">小型 / 高速</tspan><tspan x="635" dy="18">モデル（生徒）</tspan></text>
  <text x="635" y="212" text-anchor="middle" font-size="11" fill="#52525b">推論能力を移植</text>
  <line x1="146" y1="170" x2="264" y2="48" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="146" y1="178" x2="264" y2="118" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="146" y1="186" x2="264" y2="188" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="146" y1="194" x2="264" y2="258" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="488" y1="48" x2="564" y2="160" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="488" y1="118" x2="564" y2="172" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="488" y1="188" x2="564" y2="184" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
  <line x1="488" y1="258" x2="564" y2="200" stroke="#71717a" stroke-width="2" marker-end="url(#rt1)"/>
</svg>
<figcaption>大モデルの推論能力を、出力（CoT文）・パラメータ差分（ベクトル）・活性（steering）・RL報酬の 4 つの経路で小型モデルに渡す。<b>要点</b>＝実証の厚みは A（蒸留）と D（RLで教師を作る）が中心で、B・C は概念的に強力だが VLM での直接証拠はまだ薄い。</figcaption>
</figure>

「推論を移植する」とは、具体的には次のどれを移すかで系統が分かれる。**A** は大モデルが書き出す**推論文（rationale / chain-of-thought）そのもの**を教師信号にする。**B** は学習で生じた**パラメータ差分や活性方向**を「推論ベクトル」として取り出して足し込む。**C** は複数モデルを**重み空間で合成（merge）**して推論を持ち込む。**D** は **RL（強化学習）で推論を引き出した大モデルを"教師"**にし、最終的には A の蒸留へ合流させる。以下、確証度の高い順に見ていく。

## A. CoT／根拠(rationale)蒸留 ― 最も実証済み

最も枯れていて再現性が高いのがこの系統である。アイデアは単純で、**大モデルに問題を解かせ、答えだけでなく「なぜそうなるか」の推論文（rationale）も出させ、それを生徒の追加教師信号にする**。生徒の学習目的は、入力 $x$ に対して根拠と答えの同時生成尤度を上げること、すなわち

$$\mathcal{L}=\mathbb{E}\big[-\log p_\theta(\text{rationale},\text{answer}\mid x)\big]$$

と書ける（実装では rationale 生成と回答を別タスク／別段階に分けることが多い）。

- **Distilling Step-by-Step（確度=高）**: 大モデルの CoT を「追加の教師信号」としてマルチタスク学習に組み込む。結果として **770M の T5 が、few-shot の 540B PaLM を上回る** ことを示した。約 700 倍の小型化で推論を移植できた象徴的な結果である。arXiv:2305.02301（ACL 2023 Findings）。
- **SCoTD（Symbolic CoT Distillation, 確度=高）**: 教師からサンプルした rationale で訓練すれば、**従来は 50B 超でしか出ないとされた CoT の恩恵が、125M〜1.3B の小モデルでも現れる**ことを示した。arXiv:2306.14050。
- **非単調性 ＝ 設計指針（確度=高）**: 「**強い教師＝良い生徒、ではない**」。小モデルでは、能力の高い生徒には細粒度（fine-grained）な推論が効き、能力の低い生徒には単純な CoT の方が効く。つまり teacher と student の**容量差を意識した設計**が重要になる。arXiv:2502.18001（ACL 2025 Findings）。

### マルチモーダルでの直接証拠（ここが VLM 実証の中心）

- **Multimodal-CoT（確度=高）**: VLM 向けに **(1) rationale 生成 → (2) 回答推論 の 2 段階を分離**する枠組み。先に画像と質問から根拠を作り、次にその根拠も入力に加えて答えを出す。arXiv:2302.00923。
- **LLaVA-Reasoner（確度=高）**: **GPT-4o から CoT を蒸留**し、**19.3 万件のマルチモーダル CoT で SFT（さらに DPO）**を行うと、**8 ベンチマーク平均で CoT 精度が +11.7 点（62.7 → 74.4）**、個別でも **ChartQA 71.2 → 83.0**、**DocVQA 67.0 → 81.8** と改善した。arXiv:2410.16198（ACL 2025）。これが「大教師 → マルチモーダル CoT 蒸留 → 小型 VLM」という現状の実証的ベストの代表例である。
- **体系化**: 初の包括的サーベイ **MCoT Survey**（arXiv:2503.12605）と、対応リポジトリ **Awesome-MCoT** が全体像の入口になる。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="教師モデルが生成したCoTデータで小型生徒をSFTする蒸留の流れと、Multimodal-CoTの2段階推論の図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="rt2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="16" y="28" font-size="13" font-weight="700" fill="#166534">蒸留の流れ（教師 → CoTデータ → 小型生徒）</text>
  <rect x="16" y="42" width="170" height="66" rx="9" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="101" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#3730a3"><tspan x="101">教師（大）</tspan><tspan x="101" dy="17">GPT-4o / R1系</tspan></text>
  <rect x="266" y="42" width="190" height="66" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="361" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#166534"><tspan x="361">CoTデータ</tspan><tspan x="361" dy="17">(rationale, answer)</tspan></text>
  <rect x="536" y="42" width="168" height="66" rx="9" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="620" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b"><tspan x="620">生徒（小）</tspan><tspan x="620" dy="17">SFT（+DPO）</tspan></text>
  <line x1="186" y1="75" x2="264" y2="75" stroke="#71717a" stroke-width="2" marker-end="url(#rt2)"/>
  <line x1="456" y1="75" x2="534" y2="75" stroke="#71717a" stroke-width="2" marker-end="url(#rt2)"/>
  <line x1="16" y1="138" x2="704" y2="138" stroke="#e4e4e7" stroke-width="2"/>
  <text x="16" y="166" font-size="13" font-weight="700" fill="#3730a3">Multimodal-CoT の2段階（根拠生成 → 回答推論）</text>
  <rect x="16" y="182" width="150" height="64" rx="9" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="91" y="210" text-anchor="middle" font-size="12.5" font-weight="700" fill="#155e75"><tspan x="91">画像 + 質問</tspan></text>
  <rect x="206" y="182" width="180" height="64" rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="296" y="210" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="296">段階1</tspan><tspan x="296" dy="17">rationale 生成</tspan></text>
  <rect x="426" y="182" width="180" height="64" rx="9" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="516" y="210" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="516">段階2</tspan><tspan x="516" dy="17">回答推論</tspan></text>
  <rect x="636" y="182" width="68" height="64" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="670" y="218" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534">答え</text>
  <line x1="166" y1="214" x2="204" y2="214" stroke="#71717a" stroke-width="2" marker-end="url(#rt2)"/>
  <line x1="386" y1="214" x2="424" y2="214" stroke="#71717a" stroke-width="2" marker-end="url(#rt2)"/>
  <line x1="606" y1="214" x2="634" y2="214" stroke="#71717a" stroke-width="2" marker-end="url(#rt2)"/>
  <text x="296" y="266" text-anchor="middle" font-size="11" fill="#52525b">段階2 の入力には「画像 + 質問 + 生成された rationale」を与える</text>
  <line x1="296" y1="246" x2="296" y2="300" stroke="#71717a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="296" y1="300" x2="514" y2="300" stroke="#71717a" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="514" y1="300" x2="514" y2="248" stroke="#71717a" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#rt2)"/>
</svg>
<figcaption>上＝大教師が生成した根拠付きデータで小型生徒を SFT する基本形（LLaVA-Reasoner はこの形で +11.7 点）。下＝Multimodal-CoT は根拠生成と回答推論を分け、段階2 で根拠を入力に戻す。<b>要点</b>＝VLM で確実に効くのは「出力（CoT 文）」を移すこの系統。</figcaption>
</figure>

## B. 推論ベクトル・task arithmetic・activation steering

出力を真似させる A に対し、B は**学習で生じた"差"を直接ベクトルとして取り出して足し引きする**。基本概念は **task vector**：あるタスクで微調整したモデルと事前学習モデルの**重みの差分**

$$\tau = \theta_{\text{ft}} - \theta_{\text{pre}}$$

である。これを推論に特化させたのが **reasoning vector**で、**RL 後と SFT 後の重みの差**

$$v_{\text{reason}} = \theta_{\text{GRPO}} - \theta_{\text{SFT}}$$

として定義し、互換な別モデルに $\theta' = \theta_{\text{base}} + \lambda\, v_{\text{reason}}$ と加算して推論能力を移植する。活性側で同じことをやるのが **activation steering** で、推論方向のベクトル $s$ を残差ストリームに $h' = h + \alpha\, s$ と注入する。

- **Reasoning Vectors（確度=高・テキストLLM）**: $v_{\text{reason}} = \theta_{\text{GRPO}} - \theta_{\text{SFT}}$ が RL 由来の推論能力を捉えており、互換な指示チューニング済みモデルへ加算すると（1.5B で）**GSM8K +4.9 / HumanEval +4.3 / BBH +12.3**。逆に**減算すると −11.8** と落ちることから、因果的に推論を担っていると示した（アブレーション）。※検証は **テキスト Qwen2.5**、査読前。arXiv:2509.01363。
- **Small Vectors, Big Effects（確度=高・テキストLLM）**: RL で得た推論を**残差ストリームに挿す軽量な steering vector** として抽出するだけで、フル RL 微調整による向上の**大部分を再現**できる。arXiv:2509.06608。
- **線形方向としての推論挙動（確度=高・テキストLLM）**: backtracking（やり直し）などの推論挙動は**活性空間の線形方向**で制御でき、同系統の別モデルへ転移する。arXiv:2506.18167。
- **【最重要 VLM 証拠】Task Vectors are Cross-Modal（確度=高・VLM実証）**: VLM は**タスクを、モダリティ（テキスト／画像）とフォーマットに不変な"共有 task vector"**として活性内に符号化し、それは**系列の末尾近傍に局在**する。あるモダリティで抽出した task vector を**別モダリティへパッチ（移植）**でき、**ベース LLM と派生 VLM の task vector は cosine ≥ 0.89** で、**モデル間のパッチでも +1〜5% の改善**が得られる。arXiv:2410.22330。B 系統で唯一、VLM での直接実証がある核心的な結果である。

<figure class="lec-fig">
<svg viewBox="0 0 720 360" role="img" aria-label="重み空間での推論ベクトルの差分と加算、およびモダリティ間で共有されるcross-modal task vectorのパッチ移植を示す図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="rt3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="16" y="26" font-size="13" font-weight="700" fill="#3730a3">重み空間：差分を取り、別モデルへ加算</text>
  <rect x="16" y="40" width="150" height="48" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="91" y="69" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">SFT後の重み</text>
  <rect x="16" y="104" width="150" height="48" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="91" y="133" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b91c1c">RL(GRPO)後の重み</text>
  <rect x="198" y="72" width="132" height="48" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="264" y="93" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="264">推論ベクトル</tspan><tspan x="264" dy="16">＝ 差分</tspan></text>
  <line x1="166" y1="64" x2="196" y2="88" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <line x1="166" y1="128" x2="196" y2="104" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <rect x="198" y="160" width="132" height="48" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="264" y="189" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">別の小型モデル</text>
  <rect x="198" y="244" width="132" height="48" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="264" y="265" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534"><tspan x="264">推論が向上</tspan><tspan x="264" dy="16">した小型モデル</tspan></text>
  <line x1="264" y1="120" x2="264" y2="158" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <line x1="264" y1="208" x2="264" y2="242" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <text x="338" y="228" font-size="11" fill="#52525b">＋ λ × ベクトル</text>
  <line x1="360" y1="20" x2="360" y2="340" stroke="#e4e4e7" stroke-width="2"/>
  <text x="380" y="26" font-size="13" font-weight="700" fill="#155e75">cross-modal task vector（VLM 実証）</text>
  <rect x="380" y="44" width="150" height="56" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="455" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="#155e75"><tspan x="455">テキスト例から</tspan><tspan x="455" dy="16">task vector を抽出</tspan></text>
  <rect x="556" y="44" width="148" height="56" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="630" y="68" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="630">画像クエリの</tspan><tspan x="630" dy="16">活性へパッチ</tspan></text>
  <line x1="530" y1="72" x2="554" y2="72" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <rect x="468" y="138" width="172" height="56" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="554" y="162" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534"><tspan x="554">別モダリティでも</tspan><tspan x="554" dy="16">同じタスクを実行</tspan></text>
  <line x1="630" y1="100" x2="600" y2="136" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <line x1="455" y1="100" x2="508" y2="136" stroke="#71717a" stroke-width="2" marker-end="url(#rt3)"/>
  <text x="554" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">ベースLLM ↔ 派生VLM の task vector は cosine ≥ 0.89</text>
  <text x="554" y="256" text-anchor="middle" font-size="11" fill="#52525b">モダリティ・フォーマットに不変、末尾近傍に局在</text>
</svg>
<figcaption>左＝RL後とSFT後の重み差分を「推論ベクトル」とし、別モデルに \(\theta' = \theta_{\text{base}} + \lambda\, v\) と加算（テキストLLMで実証）。右＝VLM はタスクをモダリティ不変の共有ベクトルとして符号化し、抽出して別モダリティへパッチできる。<b>要点</b>＝B の定量証拠は主にテキスト、VLM 直接証拠は cross-modal task vector が中心。</figcaption>
</figure>

**留保（重要）**: B 系統の華々しい数値（+12.3 など）の多くは**テキスト LLM（Qwen2.5 など）で得られたもので、査読前のものも含む**。VLM への直接実証は実質的に cross-modal task vector に限られる。「重みや活性を足し引きするだけで VLM の推論を移植できる」と一般化するのは時期尚早である。

## C. model merging とパラメータ空間アライメント

C は複数モデルを**重み空間で合成（merge）して推論を持ち込む**系統で、確証度は**中**。素朴な task arithmetic（差分ベクトルの足し算）は強力だが弱点がある。

- **負干渉の問題**: 標準的な task arithmetic は、合成元（source）と合成先（target）が**乖離していると負干渉（negative interference）を起こして劣化**する。
- **対称性での事前アライメント（確度=中）**: Transformer がもつ**置換・回転・スケールの対称性**を使って合成前にモデルを**整列（align）**してから合成すると、非推論モデルへ高度な推論を移植でき、標準法を上回る。arXiv:2511.10850。
- **干渉低減の基盤 ― TIES-Merging**: 符号の衝突や冗長なパラメータ更新を整理して干渉を減らす、merge の代表的基盤手法。arXiv:2306.01708。

<figure class="lec-fig">
<svg viewBox="0 0 720 320" role="img" aria-label="標準のtask arithmeticは負干渉で劣化するのに対し、対称性で事前整列してから合成すると推論移植に成功することを比較する図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="rt4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <text x="180" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">標準 task arithmetic</text>
  <rect x="40" y="44" width="120" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="100" y="72" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3">推論モデル</text>
  <rect x="200" y="44" width="120" height="46" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="260" y="72" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">非推論モデル</text>
  <rect x="100" y="128" width="160" height="50" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="180" y="152" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b91c1c"><tspan x="180">単純に合成</tspan><tspan x="180" dy="16">負干渉で劣化</tspan></text>
  <line x1="100" y1="90" x2="160" y2="126" stroke="#71717a" stroke-width="2" marker-end="url(#rt4)"/>
  <line x1="260" y1="90" x2="200" y2="126" stroke="#71717a" stroke-width="2" marker-end="url(#rt4)"/>
  <text x="180" y="214" text-anchor="middle" font-size="11" fill="#52525b">source と target が乖離すると不安定</text>
  <line x1="362" y1="16" x2="362" y2="300" stroke="#e4e4e7" stroke-width="2"/>
  <text x="540" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#166534">整列してから合成</text>
  <rect x="400" y="44" width="120" height="46" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
  <text x="460" y="72" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3">推論モデル</text>
  <rect x="560" y="44" width="120" height="46" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/>
  <text x="620" y="72" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">非推論モデル</text>
  <rect x="440" y="120" width="200" height="46" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="540" y="148" text-anchor="middle" font-size="12" font-weight="700" fill="#155e75">置換・回転・スケールで整列</text>
  <line x1="460" y1="90" x2="500" y2="118" stroke="#71717a" stroke-width="2" marker-end="url(#rt4)"/>
  <line x1="620" y1="90" x2="580" y2="118" stroke="#71717a" stroke-width="2" marker-end="url(#rt4)"/>
  <rect x="440" y="196" width="200" height="50" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="540" y="220" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534"><tspan x="540">合成 → 推論移植に成功</tspan><tspan x="540" dy="16">標準法を上回る</tspan></text>
  <line x1="540" y1="166" x2="540" y2="194" stroke="#71717a" stroke-width="2" marker-end="url(#rt4)"/>
</svg>
<figcaption>左＝乖離したモデルを素朴に合成すると負干渉で劣化する。右＝Transformer の対称性で事前に整列してから合成すると、非推論モデルへ高度な推論を移植でき標準法を上回る（arXiv:2511.10850）。<b>要点</b>＝merge は「整列」が鍵で、確証度は中。</figcaption>
</figure>

## D. RL（GRPO/R1系）で推論を引き出し、小型へ

D は厳密には「移植」そのものより、**移植元（教師）の推論能力を RL で引き出す**系統である。確証度は**高**。

- **DeepSeek-R1（確度=高）**: **GRPO**（Group Relative Policy Optimization）により、SFT を経ずに複雑な推論と、いわゆる "**aha-moment**"（自己検証・やり直しの創発）を引き出した。arXiv:2501.12948。
- **MM-Eureka（確度=高・VLM）**: **ルールベース報酬の RL** で、視覚版の "aha" を再現。arXiv:2503.07365。
- **R1-V / VLM-R1 / LMM-R1（GRPO ベースのマルチモーダル R1 派生）**: いずれも GRPO を VLM に適用した系列。特に **VLM-R1** は **grounding／open-vocabulary 検出**に適用され、**3B が 72B を凌駕したとの報告**がある。arXiv:2503.07536。
- **A への合流（実運用の本線）**: D 単独で小型化するのではなく、**R1 系教師の推論を小型 VLM へ CoT 蒸留する**流れ（D → A）が現実的なパイプラインになる。

<figure class="lec-fig">
<svg viewBox="0 0 720 240" role="img" aria-label="大モデルをGRPOで訓練して推論を創発させ、CoTトレースを生成して小型VLMへ蒸留するR1系パイプラインの図" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif">
  <defs>
    <marker id="rt5" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><polygon points="0,0 7,3 0,6" fill="#71717a"/></marker>
  </defs>
  <rect x="14" y="60" width="150" height="70" rx="9" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
  <text x="89" y="88" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b91c1c"><tspan x="89">大モデル + GRPO</tspan><tspan x="89" dy="16">ルールベース報酬</tspan></text>
  <rect x="194" y="60" width="150" height="70" rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>
  <text x="269" y="88" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3730a3"><tspan x="269">推論の創発</tspan><tspan x="269" dy="16">aha-moment</tspan></text>
  <rect x="374" y="60" width="150" height="70" rx="9" fill="#cffafe" stroke="#0e7490" stroke-width="2"/>
  <text x="449" y="88" text-anchor="middle" font-size="12.5" font-weight="700" fill="#155e75"><tspan x="449">CoTトレース</tspan><tspan x="449" dy="16">を生成（教師化）</tspan></text>
  <rect x="554" y="60" width="152" height="70" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="630" y="88" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534"><tspan x="630">小型VLMへ蒸留</tspan><tspan x="630" dy="16">SFT(+DPO/GRPO)</tspan></text>
  <line x1="164" y1="95" x2="192" y2="95" stroke="#71717a" stroke-width="2" marker-end="url(#rt5)"/>
  <line x1="344" y1="95" x2="372" y2="95" stroke="#71717a" stroke-width="2" marker-end="url(#rt5)"/>
  <line x1="524" y1="95" x2="552" y2="95" stroke="#71717a" stroke-width="2" marker-end="url(#rt5)"/>
  <text x="360" y="172" text-anchor="middle" font-size="12" fill="#52525b">D（RL）で教師の推論を引き出し → A（CoT蒸留）で小型へ渡す、が実運用の本線</text>
  <text x="360" y="196" text-anchor="middle" font-size="11" fill="#71717a">MM-Eureka＝視覚版 aha ／ VLM-R1＝grounding・検出（3B が 72B 凌駕の報告）</text>
</svg>
<figcaption>RL（GRPO）で大モデルに推論を創発させ、その CoT を生成して小型 VLM へ蒸留する。<b>要点</b>＝D は「良い教師を作る」工程で、最終的には A の蒸留に合流させるのが現状の実証的本線。</figcaption>
</figure>

## 反証された通説（注意）

透明性のため、ディープリサーチで**不支持・誇張と判定された主張**を明記する。

- 「**sub-1B の Multimodal-CoT が ScienceQA で SOTA**」→ **反証（誇張）**。小型 Multimodal-CoT の SOTA 主張は文献上裏づけられない。
- 「**self-consistency 的な多数 rationale サンプリングが CoT 蒸留の成功に必須**」→ **不支持**。多数サンプリングは有用なことはあっても、蒸留成功の必要条件ではない。
- 「**知識＝浅層／推論＝深層、という層局在**」→ **反証**。すなわち「**推論が宿る場所を特定して切り出す**」というアイデアは、現状**確証を欠く**。reasoning vector / cross-modal task vector が示すのは「方向（ベクトル）として捉えられる」ことであって、「特定の層に局在する」ことではない。

## まとめと、読解後に答えたい問い

**まとめ**:

- 推論移植は **A. CoT蒸留 / B. 推論ベクトル・steering / C. model merging / D. RL(R1系)** の 4 系統に整理できる。
- **確証度が最も高いのは A（CoT/根拠蒸留）**。Distilling Step-by-Step（770M T5 が 540B PaLM 超）や SCoTD（125M〜1.3B で CoT の恩恵）、VLM では Multimodal-CoT と LLaVA-Reasoner（8 ベンチ平均 +11.7 点）が直接証拠を与える。
- **B は概念的に強力**で、reasoning vector $v_{\text{reason}} = \theta_{\text{GRPO}} - \theta_{\text{SFT}}$ の加減算（+12.3／−11.8）など印象的だが、**定量証拠の多くはテキスト LLM・査読前**。VLM 直接実証は **cross-modal task vector**（cosine ≥ 0.89）に実質限られる。
- **C（merge）は確証度=中**。素朴な合成は負干渉で劣化し、**対称性での事前整列**が鍵。
- **D（RL/R1系）は確証度=高**だが、その役割は「良い教師を作る」こと。実運用では **D → A**（R1 系教師 → マルチモーダル CoT 蒸留）に合流する。
- **「推論が宿る場所を切り出す」**という層局在の発想は**反証**されており、現状の実証的ベストは「**GPT-4o / R1 系教師 → マルチモーダル CoT 蒸留 → 小型 VLM に SFT（+DPO/GRPO）**」である。

**読解後に答えたい問い**:

1. teacher と student の**容量差**は、蒸留の効き方をどう変えるか（細粒度の推論 vs 単純 CoT）。手元のモデル対でどちらを選ぶべきか（arXiv:2502.18001）。
2. reasoning vector の加減算がテキスト LLM で成り立つとして、**VLM でも同じく成り立つ**ことを確かめるには、どんな実験（どのモデル対・どの指標）が必要か。
3. cross-modal task vector の **cosine ≥ 0.89** は、どんな前提（同一ベース・同一トークナイザなど）で成立するのか。前提が崩れると移植はどこから失敗し始めるか。
4. model merging で「整列」が必要なのはなぜか。対称性（置換・回転・スケール）で何が揃い、負干渉のどの成分が消えるのか。
5. RL（D）で得た推論を **CoT 蒸留（A）に変換するとき**、何が失われ・何が保たれるか。RL でしか出ない挙動（backtracking 等）は蒸留で再現できるか。

## 出典

**A. CoT／根拠蒸留**

- Distilling Step-by-Step（770M T5 が 540B PaLM 超）— [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)（ACL 2023 Findings）
- SCoTD（小モデルでも CoT の恩恵）— [arXiv:2306.14050](https://arxiv.org/abs/2306.14050)
- 蒸留の非単調性（強い教師 ≠ 良い生徒）— [arXiv:2502.18001](https://arxiv.org/abs/2502.18001)（ACL 2025 Findings）
- Multimodal-CoT（2段階：根拠生成 → 回答推論）— [arXiv:2302.00923](https://arxiv.org/abs/2302.00923)
- LLaVA-Reasoner（GPT-4o から 19.3 万件 CoT 蒸留、+11.7 点）— [arXiv:2410.16198](https://arxiv.org/abs/2410.16198)（ACL 2025）
- MCoT Survey（包括サーベイ）— [arXiv:2503.12605](https://arxiv.org/abs/2503.12605) ／ リポジトリ Awesome-MCoT

**B. 推論ベクトル・task arithmetic・activation steering**

- Reasoning Vectors（GRPO−SFT の差分、+4.9/+4.3/+12.3、−11.8）※テキスト Qwen2.5・査読前 — [arXiv:2509.01363](https://arxiv.org/abs/2509.01363)
- Small Vectors, Big Effects（軽量 steering vector で RL の向上を再現）— [arXiv:2509.06608](https://arxiv.org/abs/2509.06608)
- 推論挙動の線形方向制御・転移 — [arXiv:2506.18167](https://arxiv.org/abs/2506.18167)
- 【最重要 VLM 証拠】Task Vectors are Cross-Modal（cosine ≥ 0.89、モデル間パッチ +1〜5%）— [arXiv:2410.22330](https://arxiv.org/abs/2410.22330)

**C. model merging とパラメータ空間アライメント**

- 対称性による事前整列 → 推論移植（標準 task arithmetic を上回る）— [arXiv:2511.10850](https://arxiv.org/abs/2511.10850)
- TIES-Merging（干渉低減の基盤）— [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)

**D. RL（GRPO/R1系）**

- DeepSeek-R1（GRPO、SFT なしで推論と aha-moment）— [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- MM-Eureka（ルールベース RL、視覚版 aha）— [arXiv:2503.07365](https://arxiv.org/abs/2503.07365)
- VLM-R1（grounding／open-vocabulary 検出、3B が 72B 凌駕の報告）— [arXiv:2503.07536](https://arxiv.org/abs/2503.07536) ／ 同系統に R1-V・LMM-R1
