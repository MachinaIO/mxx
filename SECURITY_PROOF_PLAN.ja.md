# Lean安全性証明フレームワーク実装計画（core部とBGV）

この計画は、既存のcorrectness証明に加えて、DSLで書いたプロトコルの安全性をLeanで機械検証できるようにするためのものです。最初の対象はir-coreの共通部分と、`crates/fhe`のBGVです。まず公開鍵型FHEの構文、correctness、securityと、許可された仮定の一覧をLeanで定義し、次に既存のBGV実装がその定義を満たすことを証明します。

秘密鍵型FHE（現在のTFHE実装）、PKE、WE、iOなどほかのプロトコルは、この基盤ができてから後続段階で扱います。

日本語はユーザーの明示指定に従います。実装時のrustdoc、READMEなどの通常のリポジトリ文書、コミット、PRは既存規約どおり英語にします。この文書は計画だけを記載します。ソース実装、ビルド、統合テストは含みません。

## 実装の現状と計画からの変更（2026-11-07）

以下の節は当初の計画です。実装では次のように決定・変更しました。

- **配置。** 実装非依存の公開鍵型FHEの定義（構文、correctness、IND-CPA、hybrid補題）は`crates/fhe/lean/spec`（ライブラリ`MxxFhe`）に置きます。Rust側の構文と役割による束縛は`crates/fhe/src/spec/`（`Fhe`、`Circuit`、`Depth`、`Statement`、`Reduction`）、TFHEとBGVに依存するものはそれぞれ`crates/fhe/src/tfhe/`と`crates/fhe/lean/tfhe/`、`crates/fhe/src/bgv/`と`crates/fhe/lean/bgv/`に置きます。`MxxSecurity`（`crates/ir-core/lean/security`）には、確率的意味論（サンプラ、`determined`、距離）と許可された仮定だけを置きます。
- **アルゴリズムと実装の対応。** 各アルゴリズムは役割名（`public_key`、`evaluation_key`、`secret_key`、`message`、`ciphertext`、`ciphertext_i`、`message_i`）を入出力に持つDSLグラフ1つに束縛します。`spec::lean::export`が役割名をグラフと照合し、`Scheme.lean`で`MxxFhe.Scheme`を組み立て、`CorrectnessClaim`と`SecurityClaim`を固定の定義から生成します。
- **既存のcorrectness生成はlegacy。** プロトコル単位のexport（`bgv_round_trip`、`crates/fhe/lean/bgv/generated`）はlegacyとして残し、新しい命題は上の対応と実装非依存の定義から生成します。
- **securityの形（方式B）。** コストモデルと`Hard`述語の代わりに、リダクションをDSLグラフ（setupとchallengeの2つの模擬器）として宣言し、固定のテンプレート`MxxFhe.reduce`が敵を1回だけ呼びます。`SecurityClaim := MxxFhe.Secure scheme uses 0`は「各リダクションの仮定に対する優位性が`ε_i`以下なら、IND-CPA優位性は`2·Σε_i`以下」です。リダクションのコストは敵のコストと有限回路2つ分なので、コストモデルは不要になりました。
- **段階1.3の健全性補題は作らない。** `comp`は根のサンプラを引いたあと関係の出力を`determined`で返す形にし、証明側が関係の出力が一意に存在することを示します。
- **進捗。** 段階0〜4は完了しています。段階0では、全certificate（PEのW128とW256の行を含む）が`v4.35.0`で`lake build`に通り、axiomが`propext`、`Classical.choice`、`Quot.sound`だけであることを確認しました。PEとKHEの行生成器は、修正後のLeanファイルをバイト単位で再生成します。BGVの`CorrectnessClaim`（回路`x₀ · (x₁ · x₂)`について、すべての実行で正しく復号）と、仮定`RlweKeySwitchCircular`と`DecisionRlwe`（2回）からの`SecurityClaim`が証明済みです（`crates/fhe/lean/bgv/scheme/Certificate.lean`、axiomは`propext`、`Classical.choice`、`Quot.sound`だけ）。correctnessは当面、回路ごとの命題です（`MxxFhe.Evaluation`：回路の平文上の値とその`Eval`。`WorstCaseCorrect`と`AverageCaseCorrect`）。深さ`L`以下のすべての回路についての命題は、ゲートの集合をspecで固定するか、スキームが宣言する形で将来扱います。Rust側の`spec::Fhe`は、各アルゴリズムをDSLの値の関数として与えます（`depth`、`key_generation`、`encrypt`、`plain_gate`、`eval_gate`、`decrypt`）。回路はスキームによらず共通の`spec::Circuit`（2入力ゲートの位相順の列）で、スキームはゲートだけを与え、`spec::evaluate`と`spec::plain_eval`がゲートごとにたどります。深さは`spec::Depth`で、有限の`Bounded(d)`（BGV）か`Unbounded`（TFHE）を指定し、`spec::graphs`と`export`はそれを超える回路を拒否します。暗号文の型はそれを返したゲートに依存してよく（BGVのレベルなど）、暗号文の値がそれを持つので、どのアルゴリズムも深さを引数に取りません。`spec::lean::export`はスキームと回路を1つ受け取り、その回路の`Eval`と`plain_eval`のグラフを書き出します。BGVのゲートはslotごとの乗算で、レベル`i`と`j`の引数に対しては高い方を`max(i, j)`まで下ろしてから、乗算、再線形化、`modswitch_steps`段のmodulus switchingを行います。全ゲートが最上位レベルの再線形化鍵1本を使います。`BgvParams::relinearize`は上位レベルの鍵を暗号文のレベルに縮約し、桁の分解前に値を`κ`倍することで、鍵のgadget係数のまま桁を再結合します。雑音はそのレベルで生成した鍵と同じ上界に収まります。そのため敵が見る評価鍵は深さによらず同じで、追加の仮定（2鍵のcircular security）は要りません。

## 基本方針

| 要素 | 誰が書くか | 信頼の扱い |
| --- | --- | --- |
| 公開鍵型FHEの構文、correctnessとsecurityの定義 | 人がLeanで最初から書く（`MxxSecurity`ライブラリ） | 信頼する。レビュー対象 |
| 許可された仮定の一覧（ID）と各仮定の定式化 | ir-coreが一覧を持ち、各定式化は人がLeanで書く | 信頼する。レビュー対象。DSL側からは追加できない |
| 各アルゴリズムの具体的な定義（KeyGen、Enc、Dec、評価演算） | ir-coreがDSLグラフから決定論的に生成する | 生成器を信頼する。手で編集しない |
| 使う仮定のIDとそのパラメータ | プロトコル宣言（DSL側）が指定する | 宣言の一部。ir-coreが一覧に照らして検査する |
| 具体的な命題 | 生成器が、宣言された仮定IDに対応するir-coreの仮定を前提に置き、固定された定義を生成スキームに適用する | 生成物。中身は上の要素で決まる |
| correctnessとsecurityの証明 | AIが書く | 信頼しない。Leanカーネルが検査する |
| 安全性証明の基盤 | 最新のVCVio（`OracleComp`/`ProbComp`、ゲーム、random oracle） | VCVioの定義は信頼基盤に含める |

レビューで読むのは、`MxxSecurity`の定義、仮定の一覧と定式化、生成器の対応規則、そして各certificateの数行と`#print axioms`の結果だけになります。これは既存のcorrectnessの方式（`crates/ir-core/LEAN-SPEC.md`）と同じです。

## 現状と、安全性へ進むうえでの差分

コードから確認した現状は次のとおりです。

- **意味論は関係である。** 各グラフは`Stage_x.generatedRoot params inputs outputs : Prop`という関係として出力されます。サンプラは「cutoff内の任意の値」、ハッシュは`MxxRuntime.HashModel`の全称量化です。確率は`MxxRuntime.tapeMeasure`（`SamplerLaw`に従う独立なサンプルのテープ）としてだけ現れます。安全性にはアルゴリズムを確率的プログラムとして扱う意味論が必要ですが、現状にはありません。
- **関係として書かれたプリミティブがある。** `concatRows`、`modulusReduceRuns`、`rnsModUpRuns`などは`*Runs`という関係で、witnessとして出力されます。確率的プログラムにするには関数版が必要です。
- **BGVの既存claimは固定パイプラインである。** 「keygen→2回の暗号化→乗算→再線形化→modulus switch→復号」が常に正しいことを示しています（`crates/fhe/src/utils/protocol.rs`の`bgv_round_trip`、`crates/fhe/lean/bgv`）。任意の回路の評価や安全性は扱っていません。
- **BGVの鍵と暗号化**（`crates/fhe/src/bgv.rs`）は次のとおりです。
  - `keygen`: `s ← sample_secret()`（`[-1,1]`または`[0,1]`の一様係数）、`a ← uniform_residue`、`pk = (a, s·a + t·e)`。`e`は切り捨て離散ガウスです。
  - `encrypt`: `u ← sample_secret()`、`ct = (a·u + t·e1, b·u + t·e2 + m)`。
  - `relinearization_key`: 拡大法`PQ`上で、`key_switch_key`が`(a_i, s·a_i + t·e_i + P·g_i·s^2)`を作ります（`a_i`は一様、`g_i`はハイブリッドRNSのガジェット）。rustdocにも「評価鍵は秘密依存の値を暗号化する。呼び出し側はこの仮定を考慮すること」と書かれています。
  - `a`はすべて`uniform_residue`で、ハッシュを使いません。
- **ツールチェインが異なる。** このリポジトリは`leanprover/lean4:v4.28.0`とMathlib `v4.28.0`です。VCVioのHEAD（2026-11-07時点、`1c7ab7e`）は`v4.35.0`とMathlib `v4.35.0`を要求します。Lakeは1つのビルドでツールチェインを混在できないため、ir-coreとすべてのcertificateパッケージの移行が前提になります。
- **VCVioで再利用できるもの。** `AsymmEncAlg`とIND-CPA（`AsymmEncAlg/INDCPA`）、`HardnessAssumptions/NoisyLearning`（判定LWE型ゲーム）、`LatticeCrypto/HardnessAssumptions/LearningWithErrors`、`LatticeCrypto/DiscreteGaussian`、`Asymptotics/ReductionCost`が使えます。評価演算を含むFHEの定義はVCVioにないので自前で定義します。

## BGVに必要な仮定

BGVのIND-CPA（敵が`pk`と再線形化鍵`rk`を受け取る）には、次の2つの仮定が必要です。

1. **判定Ring-LWE（RLWE）**: 2か所で使います。
   - `pk = (a, s·a + t·e)`を一様に置き換えるとき。秘密`s`、法`Q`、サンプル1つです。
   - 一様な`pk = (a, b)`の下で、暗号文`(a·u + t·e1, b·u + t·e2)`を一様に置き換えるとき。秘密`u`、法`Q`、公開値`a, b`のサンプル2つです。
   - 誤差が`t·e`の形ですが、`gcd(t, Q) = 1`なら`t^{-1}`倍で標準形と同値です。この同値はMxxSecurityの補題として証明し、仮定は標準形で書きます。
   - 秘密の分布は`sample_secret`と同じ一様三値（または一様binary）、誤差の分布はランタイムと同じ切り捨て離散ガウスです。
2. **再線形化鍵の循環安全性**: `rk`は`s^2`を`s`自身で暗号化したものなので、RLWEだけからは安全性を導けません。BGV、BFV、CKKSで再線形化鍵を公開する場合の標準的な仮定として、「法`Q`のRLWEサンプル（`pk`）が与えられても、法`PQ`のkey-switchingサンプル`(a_i, s·a_i + t·e_i + P·g_i·s^2)`は一様と区別できない」を定式化します。
   - この仮定を許可された仮定として採用することは決定済みです（ユーザー承認、2026-11-07）。最終的な安全性の命題は、`rk`を敵に渡すIND-CPAです。
   - `rk`を敵に渡さない安全性（評価鍵なしのIND-CPA）はRLWEだけで示せます。そこで、証明の最初のマイルストーンを「RLWEだけによる`rk`なしのIND-CPA」とし、次に「RLWEと循環安全性による`rk`ありのIND-CPA」へ進みます。

## 段階0: ツールチェイン移行とVCVio導入

1. 全Leanパッケージの`lean-toolchain`を`v4.35.0`へ、Mathlibを`v4.35.0`へ上げます。対象は`crates/ir-core/lean`、`crates/dsl/examples/rlwe`、`crates/fhe/lean/{bgv,tfhe}`、`crates/khe/lean/**`、`crates/exponent-lut/lean/**`、`crates/pe/lean/**`、`crates/we/lean`です。
2. 新しいLakeパッケージ`crates/ir-core/lean/security`（ライブラリ`MxxSecurity`）を作り、そこでだけVCVioを特定のcommit（`1c7ab7e40e8e7ba9f75c88cd495193b7e6bb4248`）に固定して`require`します。ir-core本体はVCVioに依存しないので、既存のcorrectnessパッケージはVCVioとその推移依存（PolyFun、QuantumFoundationsなど）を取り込みません。VCVioのAPIは頻繁に変わるため、更新は明示的な作業として行います。
3. 既存certificateをすべて再検査します。Mathlibの補題名の変更などで証明が壊れた箇所を修正します。巨大exportのパッケージ（PE W128など）は時間がかかるので個別に記録します。
4. CI（`.github/workflows/ci.yml`のLean build jobとclaim freshness check）を新ツールチェインに合わせます。

**完了条件:** 全certificateが新ツールチェインで`lake build`に通り、`#print axioms`が`propext`、`Classical.choice`、`Quot.sound`だけを示すこと。生成claimが移行前とバイト同一であること（生成器は変更しないため）。

この段階はcorrectness側にも影響する独立した作業です。安全性の設計を待たずに着手できます。

## 段階1（core）: 確率的意味論の生成

### 1.1 ランタイム側（`crates/ir-core/lean`）

- **サンプラの分布。** `SamplerLaw`（切り捨て離散ガウス、一様区間、一様剰余）ごとに`ProbComp`のサンプラを定義します。係数ごとに独立で、`tapeMeasure`の分布と一致させます。一致を示す補題を証明します。
- **ハッシュ。** BGVはハッシュを使わないので、この段階では確率的exportの対象外とし、ハッシュを使うグラフを渡すとエラーにします。random oracleとしてのモデルは、ハッシュを使うスキームの段階で`OracleSpec`として追加します。
- **プリミティブの関数版。** witnessとして出力されている`*Runs`関係ごとに、関数版と同値補題`xRuns x y ↔ y = x.f`を用意します。BGVのグラフに現れるもの（`concatRows`、`concatColumns`、`modulusReduceRuns`、`centeredRebaseRuns`、`rnsModUpRuns`、`rnsModDown`系など）から着手します。
- **trapdoorとpreimageサンプラ。** BGVは使わないので対象外とし、使うグラフを確率的exportに渡すとエラーにします。

### 1.2 生成器側（`crates/ir-core/src/lean`）

- 各stageに、既存の`generatedRoot`に加えて`Stage_x.comp : Params → Inputs → ProbComp Outputs`を出力します。型の略記（`mxxMat*`など）は既存のものを共有します。
- 定義は行列単位で出力し、係数ごとに展開しません（例: `a * u + t * e1`）。`parallel`や反復は`Fin`で添字づけた計算の列として出力し、サイズはNに依存させません。ゲームの書き換えで項を扱えるようにするためです。
- 同じグラフとパラメータからは常にバイト同一の出力にします。既存のfreshness checkの対象に含めます。

### 1.3 関係意味論との橋渡し

- 各stageについて、`comp`の台が`generatedRoot`に含まれることを示す健全性補題`support (Stage_x.comp p x) ⊆ {y | generatedRoot p x y}`を出力します。補題の証明は生成器が汎用tacticで出力し、失敗した場合はAIが補います。命題ではなく証明なので信頼基盤は増えません。
- これにより、既存の関係claimで証明したBGVのcorrectness（最悪ケース）を確率的意味論に持ち上げられます。

### 1.4 プロトコル宣言の役割付け（Rust）

`ProtocolDecl`はstageの列とideal関数だけを持ち、どのstageがKeyGenで、どの出力が公開鍵かという役割を持ちません。そこで、公開鍵型FHE用の役割宣言を追加します。

- KeyGen stageと、その出力の役割（公開鍵、評価鍵、秘密鍵）
- Enc stage、Dec stage、平文入力、暗号文出力
- 評価演算の一覧（演算名、平文での関数、暗号文の形式ごとの実装、形式の間の変換）
- 使う仮定のIDとそのパラメータ（段階2.4）

生成器は`Scheme.lean`に、役割どおりに`comp`を組み立てた`MxxSecurity.PkFhe.Scheme`のインスタンスと、`CorrectnessClaim`と`SecurityClaim`を出力します。

**完了条件:** RLWE例（`crates/dsl/examples/rlwe`）とBGVの全stageで`comp`と健全性補題が生成され、`lake build`に通ること。既存の関係claimとその証明は変更しないこと。

## 段階2（core）: 公開鍵型FHEの定義と仮定の一覧（`MxxSecurity`）

段階0で作る`crates/ir-core/lean/security`のライブラリ`MxxSecurity`に定義を置きます。この段階のLeanファイルはすべて、人がレビューする信頼基盤です。

### 2.1 構文（`MxxSecurity.PkFhe`）

- `Scheme`は、型`Msg`、`Ct`、`PK`、`EvalKey`、`SK`と、`keyGen : ProbComp (PK × EvalKey × SK)`、`enc : PK → Msg → ProbComp Ct`、`dec : SK → Ct → Msg`、`ops`からなります。
- 評価は、宣言された演算（BGV: 加算、乗算と再線形化、modulus switch）からなる直線プログラム`Circuit`をLeanで解釈する`eval`として定義します。`eval`の解釈器は`MxxSecurity`側に人が書き、演算の中身だけが生成物です。
- 回路の平文での意味`Circuit.denote`（BGVではslotごとの`mod t`演算）も同じライブラリで定義します。

### 2.2 correctness

- `AverageCaseCorrect (S) (L) (ε)`は、深さ`L`以下のすべての回路とすべての有効な平文について、`Pr[dec (eval C (enc m)) ≠ denote C m] ≤ ε`が成り立つことです。確率は`keyGen`、`enc`、`eval`の乱択についてとります。`WorstCaseCorrect (S) (L)`は、この確率が0、つまりすべての実行で正しく復号できることです。
- BGVでは、乗算深さが`L`以下の回路で`WorstCaseCorrect`とします。`L`は既存パラメータで成り立つ値を、証明前に宣言で固定します。
- 既存のパイプラインclaimは、このcorrectnessの特別な場合として段階1の橋渡しで使えます。

### 2.3 security

- IND-CPAです。敵は`PK`と`EvalKey`を受け取り、2つの平文を選び、片方の暗号文を受け取ってどちらかを当てます。VCVioの`AsymmEncAlg`のIND-CPAに、公開鍵を`PK × EvalKey`とした`AsymmEncAlg`を渡して再利用します。
- 評価鍵なしの版（公開鍵を`PK`だけとした`AsymmEncAlg`）も定義します。段階4の最初のマイルストーンで使います。

### 2.4 許可された仮定の一覧

仮定はir-coreが一元的に管理します。DSL側は一覧にあるIDを指定するだけで、新しい仮定を導入できません。

**Rust側（ir-core）**

- 閉じた一覧として`AssumptionId`を定義します。最初のエントリは次の2つです。
  - `DecisionRlwe`: 判定Ring-LWE。
  - `RlweKeySwitchCircular`: 再線形化鍵（key-switching鍵）の循環安全性。
- 各IDにはパラメータのスキーマがあります。`DecisionRlwe`なら、環次数`N`、法`Q`、秘密分布（一様三値または一様binary）、誤差分布（`sigma`とcutoff）、サンプル数です。値は既存のパラメータ束縛と同じく`IntExpr`/`RealExpr`で書き、束縛時に具体値になります。
- プロトコル宣言は、使う仮定を`(AssumptionId, パラメータ)`の列として持ちます。同じIDを異なるパラメータで複数回使えます。BGVでは`pk`用（サンプル1つ）と暗号文用（サンプル2つ）の`DecisionRlwe`を使います。
- ir-coreは宣言を検査します。未知のIDはありえない（閉じた型）ので、パラメータがスキーマに合うこと、値が範囲内であること（法が正、cutoffが正、など）を確認します。

**Lean側（`MxxSecurity.Assumptions`）**

- IDごとに、パラメータの構造体、ゲーム（実分布と一様分布の判別）、優位性を人が書きます。`DecisionRlwe`はVCVioの`NoisyLearning.Problem`のインスタンスとして定義します。
- 仮定はゲームとして書き、生成グラフを参照しません。グラフを参照すると、仮定が主張と同じになって自明になるからです。証明側が、生成されたKeyGenなどが仮定の分布と一致することを示します。
- 各仮定を、具体的安全性の述語として定義します。
  ```lean
  def Hard (id : AssumptionId) (p : id.Params) (T : Cost) (ε : ℝ≥0∞) : Prop :=
    ∀ A : id.Adversary p, cost A ≤ T → id.advantage p A ≤ ε
  ```
- IDとLeanの定式化の対応を表`MxxSecurity.Assumption.hard : AssumptionId → ...`として持ちます。Rustの`AssumptionId`の各エントリがLeanの表に存在することを、生成物の型検査とテストで確認します。

**新しい仮定の追加**は、Rustの`AssumptionId`へのエントリ追加、Leanの定式化、レビューを1つの変更として行います。

### 2.5 security statementの形

生成器は、宣言された仮定IDに対応する`Hard`を前提に置いた命題を出力します。BGVでは次の形になります。

```lean
def SecurityClaim : Prop :=
  ∀ (T : Cost) (ε₀ ε₁ ε₂ : ℝ≥0∞),
    MxxSecurity.Assumption.hard .decisionRlwe rlwe_pk_params T ε₀ →
    MxxSecurity.Assumption.hard .decisionRlwe rlwe_ct_params T ε₁ →
    MxxSecurity.Assumption.hard .rlweKeySwitchCircular circular_params T ε₂ →
    MxxSecurity.PkFhe.INDCPA scheme (T - overhead) (ε₀ + ε₁ + ε₂ + δ)
```

- `rlwe_pk_params`などは、宣言の仮定パラメータを束縛値で評価した具体値です。
- `INDCPA scheme T' ε'`は「コスト`T'`以下のすべての敵の優位性が`ε'`以下」です。
- `overhead`（リダクションのコストの増分）、各`ε_i`の係数、統計的な誤差項`δ`は宣言で具体値として与え、証明がそれを満たすことを示します。
- 仮定を前提にするので、`∀ A, ∃ B`の形で計算量に制限のない`B`がLWEを解いてしまう自明化は起きません。`T`の大きな値では前提が偽になりえますが、すべての`T`について主張するので、意味のある`T`での主張が含まれます。
- そのため、コストモデル`Cost`と`cost`の定義が必要です。ring演算、サンプリング、敵の呼び出しを数える固定のモデルを`MxxSecurity`に定義し、VCVioの`ReductionCost`の仕組みに乗せます。具体的な設計は段階2の最初に固めます。

### 2.6 定義の健全性の確認

定義が空虚でないことを確認するため、次を`MxxSecurity`の検査用モジュールとしてLeanで証明します。

- 誤差を0にした壊れたRLWE暗号では、IND-CPAの優位性が1になる敵が存在すること
- 恒等写像を「暗号化」とする自明なスキームはcorrectnessを満たすがsecurityを満たさないこと
- 誤差を0にした`DecisionRlwe`では、`Hard`が小さな`T`と`ε < 1`で成り立たないこと（仮定の定式化が空虚でないこと）

**完了条件:** `MxxSecurity.PkFhe`、仮定の一覧と定式化、コストモデル、検査用モジュールが`lake build`に通り、レビューを終えていること。

## 段階3（fhe）: BGVの宣言とexport

- `crates/fhe/src/utils/protocol.rs`に、段階1.4の公開鍵型FHEの役割宣言を作る関数を追加します。既存の`bgv_round_trip`と同じグラフを使い、演算stage（加算、乗算、再線形化、modulus switch）を個別に宣言します。
- 使う仮定として、`DecisionRlwe`を2回（`pk`用と暗号文用）と`RlweKeySwitchCircular`を1回宣言します。パラメータは`BgvParams`（`FheCommonParams`の環、秘密区間、`sigma`、cutoff、`plaintext_modulus`、ハイブリッドの補助素数）から導きます。
- `crates/fhe/examples/export_claims.rs`を拡張し、既存の`CorrectnessClaim`に加えて`Scheme.lean`、`SecurityClaim`を`crates/fhe/lean/bgv/generated/`へ出力します。GPUを使わずに生成できる経路を維持します。
- パラメータは既存の`utils::bgv_params`です。このパラメータの具体的な格子困難性（lattice estimatorによる評価）は命題に含めません。評価結果はREADMEに参考として記録するだけにします。
- CIのfreshness checkに新しい生成物を加えます。

**完了条件:** 生成物が決定的で、`lake build`で型検査に通ること。証明を`sorry`にした仮のcertificateで、axiom報告に`sorryAx`が出ることも確認します。

## 段階4（fhe）: BGVの証明

1. **correctness（既存パイプライン）**: 段階1の橋渡しで、既存の`BgvProof`の結果を`MxxSecurity.PkFhe`のcorrectnessの特別な場合へ持ち上げます。
2. **correctness（深さ`L`以下の回路）**: 演算ごとの雑音の不変条件（暗号文の位相が平文と雑音上界を持つこと）を補題にし、回路についての帰納法で示します。雑音上界の計算は既存Rustの`phase_bound`と`key_switch_noise`に対応させます。
3. **IND-CPA（評価鍵なし、RLWEだけ）**: 次のゲーム列で示します。
   - 生成された`keyGen`と`enc`が、仮定のRLWE分布と一致することを示します（`t·e`の標準形への変換を含む）。
   - `pk = (a, s·a + t·e)`を`DecisionRlwe`（`pk`用）で一様に置き換えます。
   - 一様な`pk`の下で、暗号文`(a·u + t·e1, b·u + t·e2 + m)`の`m`以外の部分を`DecisionRlwe`（暗号文用）で一様に置き換えます。
   - 最後のゲームでは、暗号文が`m`と独立になり、敵の優位性が0になります。
4. **IND-CPA（評価鍵あり）**: 3のゲーム列の最初に、`rk`を`RlweKeySwitchCircular`で一様に置き換えるhopを加えます。この仮定は`pk`を含む見え方で定式化しているので、続く2つのRLWE hopは3と同じです。
5. 段階2.5の形で、`overhead`、係数、`δ`を命題どおりに与えます。

## 作業順序と並列性

- 段階0は単独で先に進めます。correctness側の既存資産を守る作業でもあります。
- 段階1（生成器）と段階2（定義と仮定）は、互いのファイルが重ならないので並行できます。段階1.4の役割宣言と仮定宣言の形は、段階2.1の`Scheme`と段階2.4の`AssumptionId`に合わせる必要があるため、構造を先に合意します。
- 段階3は段階1と段階2が終わってからです。段階4の3は、2と並行できます。

## 決めておくべき事項

| 項目 | 推奨 | 決定時期 |
| --- | --- | --- |
| コストモデル（2.5） | ring演算、サンプリング、敵の呼び出しを数える固定モデルを、VCVioの`ReductionCost`に乗せる | 段階2の最初 |
| 仮定の誤差分布（2.4） | ランタイムと同じ切り捨て離散ガウス | 段階2 |
| `RlweKeySwitchCircular`の具体形（2.4） | 採用は決定済み。具体形は、法`Q`の`pk`を含む見え方で法`PQ`のkey-switchingサンプルが一様と区別できない、という標準形を推奨 | 段階2 |
| BGVの深さ`L`（2.2） | 既存パラメータで成り立つ値を宣言で固定 | 段階3 |
| VCVioの固定commit | 段階0時点のHEADを固定し、更新は明示的に行う | 段階0 |

## 段階5: 帰着の計算量を命題に入れる（IRの深い埋め込み）

現在の`SecurityClaim`は、生成したsimulatorを命題に埋め込み、それが有限のIR回路であることを読み手が確かめる形です。`Comp`は計算量を数えないので、証明側で任意の`Comp`をsimulatorとして使えると、仮定を総当たりで解く帰着でも命題が通ってしまいます。段階5では、命題そのものが帰着を「IRで書かれた、コストが上界以下のプログラム」に限定するようにします。

- **深い埋め込み。** IRのグラフをLeanのデータとして表す型`MxxSecurity.IR`（Rustの`NodeKind`の全種類に1対1で対応する演算、ノード、scope、ループと呼び出しを含むプログラム）を置きます。exportするのはパラメータを束縛した後のグラフで、ループ添字に依存する式だけを小さな整数式として残します。
- **関数としての意味論。** `eval : Program → List Value → Comp (Option (List Value))`を演算ごとの関数で定義します。`determined`や`Classical.epsilon`のように関係から値を選ぶ演算は使いません。サンプリングは`MxxSecurity.sample`を使います。意味論を定義していない演算は`none`を返すので、そのような演算を含むsimulatorでは存在命題を満たせず、健全性は損なわれません。演算の意味論は、BGVのsimulatorに必要なものから順に足し、最終的に全種類を覆います。
- **コスト。** 演算ごとのコストを、係数単位の演算回数として1つの表（`MxxSecurity.IR.Cost`）に定義します。ループは回数倍、呼び出しは本体のコストを加えます。この表は信頼すべき部分で、過小評価がないことをレビューで確かめます。
- **命題。** `MxxFhe.Secure`を、「各仮定について、IRプログラムで書かれたsimulatorでコストが`C`以下のものが存在し、どの敵に対してもアドバンテージの上界が成り立つ」という形にします。値とスキームの型（`PK`、`EK`、`CT`、平文）の相互変換は生成器が固定で書き出し、証明側で計算を持ち込めないようにします。
- **信頼すべき部分。** 演算の意味論がRustの実装と一致すること（correctnessの命題でもすでに信頼している部分です）、コスト表、仮定の困難性とコストの単位の換算です。
- **進捗（2026-11-09）。** 段階(1)の土台ができています。`MxxSecurity.IR.Syntax`（全58種類の演算の構文）、`MxxSecurity.IR.Eval`（関数としての意味論。行列の算術、切り出し、結合、剰余の縮約と再表現、整数と実数の演算、族と選択、係数表現の多項式の入出力、3種のサンプラ、並列と逐次のループ、呼び出し）、`MxxSecurity.IR.Cost`（コスト表）、Rustの`mxx_ir_core::lean::program::program_term`（検証済みグラフを`MxxSecurity.IR.Program`のLean項として書き出す）です。BGVの3つのsimulatorを書き出してLeanでコストを計算できることを確認しました（鍵のsetupは約44万〜74万、暗号化は約2億〜6億単位）。NTTを使う多項式の入出力は、素数の法についてだけ意味論があります（`forwardNtt`/`inverseNtt`）。RNSの基底変換、トラップドア、プレイメージ、ハッシュなどの意味論はまだありません。値の行列は法`q`だけを持ち、CRTの基底が必要な演算は型から基底を読みます（環は`q`で決まり、基底は表現の選び方なので、仮定の側は変えません）。段階(2)として、`MxxFhe.SecureWithin`（`crates/fhe/lean/spec/MxxFhe/Bounded.lean`）と、型ごとに固定の変換（`MxxSecurity.IR.ToValue`/`FromValue`、仮定の挑戦の名前付き成分`AssumptionId.challengeValues`）を定義し、生成器が`codec`、`assumptions`、`BoundedSecurityClaim`（上界は`SecurityBinding::cost_bound`、BGVでは`2^30`）と、証明用の`Simulators.lean`を書き出します。3つのsimulatorのコストが上界以下であることはカーネルの計算で証明済みです（`BgvBounded.lean`）。段階(3)も終わり、BGVの`BoundedSecurityClaim`を証明しました（`Certificate.lean`の`bounded_security`、公理は`propext`、`Classical.choice`、`Quot.sound`のみ）。IRのスロットのエンコード（逆NTT）がランタイムのエンコードと一致すること（`BgvNtt`、`BgvPrograms`）、setupと暗号化のプログラムの評価（`BgvPrograms`、`BgvReductions`）、`reduceProgram`で走らせた各帰着が生成済みの`reduce`と同じ計算であること（`BgvReductions`）を示し、既存の`security`とコストの補題から導きました。証明上の注意として、8192要素の`List.ofFn`を含む項をsimpで書き換えるとカーネルが深い再帰で失敗するため、族の長さを抽象した補題（`applyPure_fromValuesNtt_of`）を経由します。残りは段階(4)です。
- **進捗（2026-11-10）。** 段階(4)を終え、段階5は完了しました。
  - **コストの健全性の修正。** 段階(3)までの命題には、コストを過小評価できる穴が3つありました。評価が値と宣言型の一致を検査していなかったので、小さな型を宣言して大きな値を扱えました。整数演算は大きさによらず1単位なので、繰り返し二乗で指数長の整数を作れました。法の大きさは数えておらず、巨大な法の上の算術が安くなりました。いまは、各ノードの出力を宣言型と照合し（`Value.conforms`）、整数演算の結果を`intBound`（`2^256`）で打ち切り、係数のコストを法の64ビットワード数で数えます（`limbWords`）。BGVの法はどれも1ワードなので、コストの数値は変わっていません。加えて、Kronecker積、複数の積の和、SHA-256の選択列の過小評価だったコストを直しました。
  - **評価の文脈。** ハッシュ関数とバックエンドのレイアウトを`MxxSecurity.IR.Context`として評価に渡します。`SecureWithin`はバックエンドを引数に取り、ハッシュ関数をsimulatorを選んだ後で全称量化します（simulatorはハッシュ表に依存できず、零のハッシュでも働く必要があります）。生成する`BoundedSecurityClaim`は`∀ backend, SecureWithin backend …`です。
  - **残りの演算の意味論。** 全演算に意味論を与えました。ビット抽出、積和、自己同型、単項式の乗算、法の切り替え、中心化した丸め除算、RNSの基底変換（`rnsModUp`は引数の型の順序付き基底を構文に持ちます）、Kronecker積、ブロック対角、gadget分解（通常と小さい版）、係数の取り出し、整数の持ち上げ、閾値による復号、CRTの再合成、係数のビット詰め、整数の行列ベクトル積、SHA-256の選択列、ハッシュ（平文と分解付き）、整数族のハッシュ、gadgetトラップドアとその公開行列、トラップドアの生成（`trapGenPublic`、`statTrapGenPublic`の通りに標本化し、秘密の短い行列を値に持たせます）、gadgetトラップドアの原像、残りの定数行列、合成数の法のNTT（根は各素因数のネイティブ根のCRT持ち上げです）です。
  - **標本化したトラップドアによる原像。** サンプラのoracleの問い合わせを`SamplerQuery`（法則による係数1つ、または原像の要求）に広げ、答えの型を問い合わせごとに決めました。原像の要求`PreimageRequest`には、係数が`[-bound, bound]`に入り`A x = u`を満たす`x`上の、幅`sigma`の離散ガウス（有限集合上で正規化したもの）で答えます。この分布をランタイムのMicciancio–Peikertサンプラが引くことは、標本が理想的だという仮定R1と同じく信頼する部分です。原像の演算は、プログラムが生成したトラップドア値（秘密の短い行列と原像の幅を持ちます）を通してだけこの問い合わせを使い、そのトラップドア自身の公開行列を使うので、挑戦の行列のトラップドアを偽ることはできません。幅を指定しないトラップドアの幅はバックエンドの最小値（`Backend.preimageWidth`）で、`SecureWithin`はバックエンド（gadgetのレイアウトと既定の幅）を引数に取ります。これで全演算に意味論があります。
- **命題の整理（2026-11-10、ユーザー判断）。** `PublicKeyLhe`（現`Fhe`）から`bindings`、`correctness`、`security`を外し、トレイトは構文だけを持つようにしました。主張する命題は`spec::Statement`（correctnessの種類、許可された仮定とパラメータ、コストの上界）、証明に使うsimulatorは`spec::Reduction`として`export`に渡します。`Scheme.lean`の主張は`CorrectnessClaim`と`BoundedSecurityClaim`の2つだけになり、simulatorを命題に埋め込む旧`SecurityClaim`は廃止しました。simulatorの計算と、それを使うIND-CPA（`SimulatedSecurity`）は証明用の`Reductions.lean`に移り、`BoundedSecurityClaim`の証明の中間補題として使います。
- **段階。** (1) 型、値、全演算の構文、BGVのsimulatorに必要な演算の意味論とコスト、RustからLeanデータへのexport。(2) `Secure`の書き換えと、BGVの鍵のsimulator（setup）の移植。(3) 暗号化のsimulatorの移植（NTTの表などを含むグラフの意味論と、既存の閉形式との対応）。(4) 残りの演算（トラップドア、プレイメージ、ハッシュなど）の意味論。

## 手動のround trip correctnessの廃止とTFHEの移行（2026-11-10、ユーザー指示）

- **方針。** 統合テストのグラフをプロトコルとして書き出す手動のround trip correctness（`utils/protocol.rs`、`bgv::round_trip`、`lean/tfhe`と`lean/bgv`の旧証明）は廃止し、correctnessはすべてFHE spec（`Fhe`、`MxxFhe.WorstCaseCorrect`/`AverageCaseCorrect`）で述べます。GPUテストは`spec::graphs`のグラフ、つまり命題が名指すグラフを実行します。BGVの証明は`crates/fhe/lean/bgv`に移しました。
- **TFHEの公開鍵暗号化。** RLWE型の公開鍵（Joyeのcompact public keyとkey switchの組み合わせに相当）にしました。公開鍵は環の秘密鍵`z`の下のRLWE標本`(a, a z + e)`で、暗号化は`(u a + e₁, u b + e₂ + μ)`のsample extractとkey switchで`s`の下のLWE暗号文にします。key switch鍵のマスクは、ハッシュ導出をやめ、約`2^162`の環の一様剰余を`q`で割った余りにしました（確率的exportがハッシュを扱えないため）。鍵生成の乱数は、確率的exportが扱えるように根のスコープで一括して引きます。
- **TFHEのパラメータ変更（2026-11-10、ユーザー指示）。** 公開鍵暗号化では両方のNAND入力がkey switchの雑音（`Σ dᵢ ẽᵢ`、8192項、`|d⁽⁰⁾ + d⁽¹⁾| ≤ 6`）を持つため、σ `2^17`では桁を最悪値とした裾評価が約`2^-26`にしかなりません。LWEの雑音をσ `2^15`に下げ、次元を`n = 720`に上げました。lattice estimator（二値秘密、`q = 2^32`）で約`2^95`（dual hybrid）で、元の`n = 630`、σ `2^17`の約`2^93`を上回ります。同じ評価で入力側は約`2^-137`、出力側（回転指数2048通りの和集合上界込み）はそれ以下です。
- **TFHEの命題。** 1段のNANDの平均的なcorrectness（失敗確率`2^-100`）を主張し、安全性の命題はまだ出しません（`Statement::security`は任意）。
- **TFHEの正しさの証明（2026-11-10完了）。** `crates/fhe/lean/tfhe`の`Certificate.lean`が`CorrectnessClaim`（失敗確率`2^-100`）を証明し、公理は標準の3つだけです。手順は、(1) `MxxSecurity.Tape`で`Comp`の分布を標本の直積測度上の関数として表す、(2) 生成グラフの関係ごとに閉形式を示す、(3) 失敗を入力雑音（key switch誤差と、NAND入力のマスクを`2N`へ丸める誤差）と出力雑音（ブートストラップ鍵の誤差と出力のkey switch誤差）が大きい事象に含める、(4) 丸め誤差はkey switch鍵のマスクの列ごとに1座標を取り出し、ほぼ一様な剰余（法`2^162`の一様剰余を`q`で割った余り）と反転による対の議論でsub-Gaussianにする、(5) 出力雑音は回転指数2048通りそれぞれについてステップごとに剥がしてsub-Gaussianにし、和集合上界をとる、です。数値は入力側が約`2^-135`、出力側は無視できる大きさです。

## 後続段階

この計画（core部とBGV）は段階5までで完了です。以下はこの計画の範囲外で、着手する前に設計を別途まとめます（2026-11-10、ユーザー判断）。

- 秘密鍵型FHE（TFHE）: 多回IND-CPA、random oracleとしてのハッシュ、`bsk`と`ksk`の鍵循環に対する仮定を、仮定一覧へのエントリ追加として扱います。
- PKE、WE、iO: プロトコル種別ごとの構文と定義を`MxxSecurity`に加え、trapdoorとpreimageサンプラの分布モデルを決めます。

## リスク

- **ツールチェイン移行で既存証明が壊れる。** Mathlibの7版分の差があります。巨大なexportのパッケージは再検査だけでも時間がかかります。
- **VCVioのAPIの変化。** 固定commitで抑えますが、更新のたびに`MxxSecurity`の修正が必要になります。VCVioはLeanの`module`システムを使っているため、import方式の違いにも注意します。
- **大きな具体型でのelaborationコスト。** `N = 8192`の`ExactMatrix`上の一様分布や`SampleableType`インスタンスの展開で、型検査が重くなる可能性があります。行列単位の定義と、Nに依存しない補題で抑えます。
- **定義や仮定の誤り。** 安全性の命題が弱すぎたり、仮定の定式化が強すぎたり空虚だったりすると、証明が通っても意味がありません。段階2.6の検査用モジュールと人によるレビューで抑えます。
- **具体パラメータの困難性は保証しない。** 命題は仮定に対する相対的な安全性です。テスト用パラメータの格子困難性は別途評価が必要です。
