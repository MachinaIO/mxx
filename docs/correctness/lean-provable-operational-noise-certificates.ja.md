# Lean で検証できる operational-noise certificate 仕様

## 1. この文書の目的

この文書は、mxx の Rust 実装が行う operational-noise 判定について、同じ判定根拠を
Lean で独立に検証できる、現在の Tall vertical slice の実装仕様である。実装は許可されて
いるが、以下の gate と各変更の review を必須とする。現在は Security0/Security128 の最終
evidence gate に近い実装途中である。

対応する英語版は
`docs/correctness/lean-provable-operational-noise-certificates.md` である。両文書は同じ要件を
表す。記述に食い違いが生じた場合は、review 済みの英語版と現在の Rust 実装を照合し、
どちらかを黙って解釈し直すのではなく、仕様を再 review する。

この仕組みを直感的に表すと、次のようになる。

```text
既存の Rust checker
  ├─ 現在と同じ計算で residual と noise bound を求める
  └─ その計算を Lean が検算するための certificate を出力する
                         │
                         ▼
Lean kernel
  ├─ residual の意味
  ├─ noise の相殺と上界
  └─ 2 * p * noise < q という狭義不等式
     を certificate から検証する
```

Rust は certificate の作成者であり、Lean は検証者である。Rust 側の certificate 生成に
誤りがあれば、Lean の型検査が失敗しなければならない。

最初の対象は、受理済みの Tall BGG nested-RNS operational-noise パラメータ
セットを一つ証明することである。ここで Tall は対象となる統合 workload の名称、
BGG と nested-RNS はその暗号構成を指す。pinned checker の request は正確に一つの
`ResolvedAcceptanceTarget` へ解決され、その kind は `ResolvedDecoderKind::Threshold` で
なければならない。`Source.json` は、固定した Tall constructor、正確な
`OperationalCheckRequest`、pinned source/evaluator identity からなる canonical な固定
profile recipe である。projection は recipe から checker を再構成し、kind から平文 modulus
`p` を、`ResolvedAcceptanceTarget.ciphertext_modulus` から暗号文 modulus `q` を、
`ProductionRoots.residual` から residual root を取り出す。直接 Rust を実行した結果と
`Source.json` から再構成した結果は report と決定的な core counter が一致しなければならない。
certificate は狭義の noise 条件だけを証明し、runtime decoder の出力は仕様化も証明もしない。
複数パラメータを一度に扱う一般定理は初期対象に含めない。

## 2. 最初に知っておく用語

### 2.1 Operational noise と residual

暗号文の復号結果には、意図した平文成分に加えて誤差が残る。この誤差をこの文書では
operational noise と呼ぶ。

Rust checker は、復号の正しさに影響する最終的な誤差式を記号的に組み立てる。この最終式を
`residual` と呼ぶ。証明したいことは、許可された入力と sampler の値に対して residual が
十分小さく、`2 * p * noise < q` が成り立つことである。

### 2.2 Source と event

- `source` は、定数、protocol input、別 stage の出力など、式へ値を供給する入口である。
- `event` は、uniform sampling、Gaussian sampling、hash、preimage sampling、gadget
  decomposition など、個々の確率的または関係付きの値生成を表す。

同じ種類の sampler であっても、別の場所で発生した event は別物である。種類名や文字列だけで
同一視してはならない。

### 2.3 Program、family、selector、owner scope

`program` は引数を受け取り式を評価する再利用可能な計算単位である。

`family` は、`Nat` で指定した要素を取得できる値の集合である。その要素を選ぶ `Nat` を
`selector` と呼ぶ。たとえば `K(i)` の `i` が selector である。

同じ式や event が別の program 呼び出しから使われる場合、それぞれの呼び出し文脈を区別する
必要がある。この文脈を `owner scope`、ある scope の引数を別の scope へ対応付ける写像を
`scoped substitution` と呼ぶ。

### 2.4 Certificate と Lean kernel

`certificate` は、Rust checker の結論だけでなく、その結論を再検算するために必要な式、型、
関係、上界、有限表、証明手順を含むデータである。

`Lean kernel` は Lean の最小の型検査器である。この仕様では、生成器を信用する代わりに、最終
定理が kernel により型検査されることを正しさの基準とする。

### 2.5 この文書で使う補助用語

- `canonical` は、同じ意味の入力から常に一意の標準表現を得ることを意味する。
- `artifact` は、source、certificate、proof など、保存・監査する生成ファイルを意味する。
- `row` は、ID で参照できる一件分の型付き record を意味する。
- `payload` は、row や proof に実際に保存・出力するデータ全体を意味する。
- `LUT` は lookup table の略で、有限な全入力と対応する出力を列挙した表である。
- `DAG` は directed acyclic graph の略で、依存関係を表す向き付きで循環のない graph である。
- `fail closed` は、必要な根拠が不足したときに推測で続行せず、certificate 生成を失敗させる
  方針を意味する。

## 3. 既存 Rust 実装との関係

固定した source revision にある operational-noise Rust 実装を、意味論と性能の正本とする。
既存 Rust core の意味論を変更する提案には別途 review と明示的承認が必要である。
特に、次のファイルの既存動作はこの設計によって変更しない。

- `arena.rs`
- `program.rs`
- `lower.rs`
- `normal_form.rs`
- `bound.rs`
- `relation.rs`
- `report.rs`
- `simulation.rs`

certificate 対応として追加できるのは、明示的に有効化した場合だけ動く観測 hook、recorder、
serialization、Lean proof の生成、および固定 Lean 定義である。既存 Rust コアの意味論を変える
必要が生じた場合は、この仕様とは別に review と明示的な承認を必要とする。

Rust コアと Lean 側の再現結果が一致しない場合、certificate 生成を失敗させる。Lean 側の都合で
Rust の判定を近似したり、別の判定へ読み替えたりしてはならない。

## 4. 必須の正しさと効率

実装は次の条件をすべて満たさなければならない。

1. 生成される定理の型は、protocol ごとに作り替えない一つの固定型とする。
2. residual の意味、厳密な相殺、残った noise の上界、狭義不等式
   `2 * p * noise < q` を Lean kernel が検証する。
3. `native_decide`、`sorry`、生成 axiom、protocol 固有の trusted code、node 番号への依存、
   debug 文字列による identity、fixture 値の列挙による証明を使わない。
4. 定理の主張を決めるデータは、固定 profile recipe、正確な `OperationalCheckRequest`、
   pinned audited Tall constructor から決定的に作る。証明の並べ方や証明項を変えても、定理の
   主張は変わらない。
5. residual proof closure 内で、Rust checker が使った型付き identity、scope ごとの event、
   relation の適用条件、bound の transfer rule をそのまま保持する。
6. certificate 出力を無効にした通常実行では、観測・記録処理を実行せず、計算量を増やさない。
7. certificate 生成の計算量は、residual proof closure、出力する証明 context の総量、
   その closure 内の index-use LUT の全行数に対して線形とする。
8. matrix-valued family の全要素や parallel-loop の全 lane を展開しない。一つの index use が
   実際に依存する selector は列挙してよいが、無関係な index use や selector を同じ直積へ
   入れない。

## 5. 全体を構成する最小の要素

この仕様でいう簡潔さは、意味を持つ data structure、独立した概念、例外経路、通常の主処理段階
が少ないことで評価する。通常経路は「既存 source を読む、Rust の計算を記録する、Lean で検査
する」の三段階に保つ。同じ役割の表や検査規則を種類ごとに増やさず、以下の共通要素を再利用
する。

### 5.1 RowTable

多くの certificate データは、自然数 ID から一つの row を引く表として表現できる。この共通表を
`RowTable` と呼ぶ。

`RowTable` は平衡木として実装し、ID lookup が表全体の線形走査にならないようにする。同時に、
監査用の決定的な row 順序も保持する。source、event、式などで別々の表実装を作らず、同じ
`RowTable` を使う。

### 5.2 Cert.Valid と Cert.wellFormed

certificate の構造が正しいという条件を `Cert.Valid` と呼ぶ。production 規模では、row ごとの
局所的な証明を組み合わせて `Cert.Valid` を構築する。

小さな fixture では、同じ条件を Boolean で確認する `Cert.wellFormed` も利用できる。この二つは
同じ row predicate を共有し、一方だけに別の検査規則を追加してはならない。Prop の proof と
Boolean の検査結果を結ぶ共通 lemma を用意し、固定 theorem
`Cert.Valid.wellFormed` により、`Cert.Valid` から
`Cert.wellFormed cert = true` を得られるようにする。

### 5.3 生成物の分離

定理の主張を決めるデータと、その証明の書き方を分離する。`Source.json` は完全な frozen
bundle serialization ではなく、固定 profile recipe である。正確な request identity、固定
した audited Tall constructor parameter、pinned source/evaluator version だけを保持し、
bundle の実行意味を再実装しない。target/environment の重複、normal form、LUT output、bound
ledger、proof は含めない。

generator は、commit 対象の小さい ABI として次の sharded output を生成する。

- `Cert/`: statement row と局所的な validity proof
- `Proof/`: immutable history と通常の theorem application
- `Semantic/`: 到達した semantic theorem application と固定 acceptance への合成

最後に固定 `TallSemantics.Security0Accepted` module がこれらを import して kernel で検査する。
現在の profile では生成 Lean output は約 720 MB の build artifact であり、repository に
commit する必要はない。これにより、generator が証明しやすい別の主張へ差し替わることを防ぐ。

## 6. 信頼する範囲

### 6.1 固定 trusted code

信頼するコードは protocol に依存しない次の部分に限定する。

- Lean の matrix、scalar、`Value` の意味
- 到達した residual closure に必要な Tall の固定 expression/program semantics
- input と sampler の contract
- `RowTable`、`Cert.Valid`、`Cert.wellFormed` と共通 reflection lemma
- 固定 `TallSemantics.Security0Accepted` endpoint
- Rust 側で identity を補足する sidecar と canonical source projection

Rust の normalizer、relation search、bound search、proof の順序決定、proof renderer は証明の
材料を作るが、正しいものとしては信頼しない。誤った材料は Lean の検査で拒否されなければ
ならない。

ここで使う三つの用語を先に定義する。`ValueClaim` は一つの event の result についての主張で
あり、その result が記録された terms と合同で、記録された bound を満たすことを表す。owner
が同じでも、別 event の claim を一つの global value として扱うものではない。`Witness` は、
event-local claim を実行へ具体化するための environment、sampler contract、relation congruence、
honest terminal bridge を供給する。`ExactClaimAt` は claim を特定の event、owner invocation、
terms、summary、history row と結び付ける。したがって Lean は、同じ owner を持つ全 row の payload
が一致するという仮定なしに、各 theorem application を検査できる。

certificate の row は、定理が対象とする source 自体を決める。このため、source の選択と
source から certificate への canonical projection は監査対象の trusted boundary とする。
Lean kernel の検査だけでは、選択された source が実際に運用した artifact と同一であること
までは証明できない。

certificate path は、正常な generator が生成した honest run の証明 artifact を対象とする。
検証範囲は、(1) 通常の generator 出力の正当性、(2) Lean theorem が使う各事実について、記録した
LHS、RHS、owner invocation、event が実際の Rust 計算と意味的に一致すること、(3) honest run に
おける cache と frame の lifecycle が完全であること、(4) 不正な参照や dangling reference が
panic を起こす前に拒否されるための最小限の構造検査、の四つに限定する。これは一般的な悪意ある
certificate parser や、あらゆる偽造に対する網羅的な防御機構ではない。専用の mutation test や、
witness の入れ替え・偽造を網羅的に拒否する検査はこの範囲に含めない。出力を無効にした場合は、
既存 Rust checker の通常経路と性能を変えない。benchmark estimate はこの仕様の成果物に含めない。

### 6.2 Source artifact

`Source.json` は完全な frozen-bundle serialization ではなく、canonical な固定 profile recipe
である。正確な `OperationalCheckRequest`、固定した audited Tall constructor parameter、pinned
source/evaluator version を保持し、bundle の実行意味を再実装しない。target/environment の重複、
normal form、LUT output、bound ledger、proof は含めない。

canonical projection は recipe から pinned checker run を再構成し、正常に解決された
`ResolvedAcceptanceTarget` が正確に一つあることを要求する。その kind は
`ResolvedDecoderKind::Threshold { plaintext_modulus }` でなければならず、この値を `p`、
`ciphertext_modulus` を `q`、`ProductionRoots.residual` を residual root とする。
`BooleanInterval`、target の解決失敗、resolved target の欠落や複数生成、target identity、
residual modulus、threshold report の `p`/`q` の不一致は certificate 生成を拒否する。
直接 Rust run と Source から再構成した run の report field と決定的な core counter は一致しなければ
ならない。これは statement source の audit であり、`Source.json` が全 bundle object を運べるという
意味ではない。digest は監査用に含めてもよいが、semantic identity の代わりにはしない。

### 6.3 実行へ適用するための外部条件

Lean theorem は条件付きの定理である。ある実行へ適用するには、certificate の外で次を確認する
必要がある。

- deploy した frozen bundle、request、source revision、evaluator revision が、選択した
  canonical source artifact と完全に一致すること。target ID と parameter environment は request
  自体に含まれる
- residual proof closure 内の各 `SourceAccess` について、実行時に供給した値が、同じ型付き
  source、owner invocation、scoped substitution、optional `Nat` family selector に割り当てた
  値と等しく、
  `InputContract` の生の事実を満たすこと
- residual proof closure 内の各 scoped event occurrence について、実行時に生成した値が、
  その exact event と owner invocation に割り当てた値と等しく、`SamplerContract` の型、
  cutoff、support、relation 条件を満たすこと

最初の項目は Lean claim の外にある、deploy 対象との一致命題である。後ろの二項目は、Lean
claim に渡す具体的な `InputAssignment` と `SamplerAssignment` を作る。digest の一致は監査には
使えるが、上記の型付き equality の代わりにはならない。

certificate は `InputContract` や `SamplerContract` を満たす値が無条件に存在するとは主張
しない。sampler cutoff が runtime により強制されず確率的にしか成立しない場合は、contract の
外へ出る確率を別の確率 theorem で評価する必要がある。

### 6.4 Lean acceptance 後にも必要な命題

固定 acceptance module が生成された `TallSemantics.Security0Accepted` を検査した後は、noise の
狭義不等式、family selector の domain、modulus と ring の整合性、certificate validity は
証明済みであり、仮定として残らない。residual root の引数についての仮定もない。

accepted operational theorem を実際の一回の実行に適用するとき、次の contract と modeling の
仮定が残る。

```text
InputContract document inputs
SamplerContract document inputs samplers
HonestTerminalCongruence document run
RecordedCoefficientCoverage document run
```

実際の一回の実行へ結び付けるには、さらに Lean claim の外で次の三命題を確認する。

```text
DeploymentMatches(run, document)
  := run が使った bundle、request、source revision、evaluator revision の
     canonical source が、document と history と対で受理した Source.json と等しい

InputsInstantiate(run, inputs)
  := residual proof closure 内のすべての SourceAccess a について、
     inputs(a) = run が a で実際に供給した値

SamplersInstantiate(run, samplers)
  := residual proof closure 内のすべての event occurrence e と owner invocation o について、
     samplers(e, o) = run が (e, o) で実際に生成した値

HonestTerminalValuesInstantiate(run, witness)
  := 到達したすべての terminal Result event e について、
     witness.honestTerminalActual(e) = run が e で実際に生成した値
```

たとえば input coefficient bound が 8 の場合、`InputsInstantiate` は「証明中の input 値が実行時
の値そのもの」と保証し、`InputContract` は「その同じ値の全 centered coefficient が絶対値 8
以下」と保証する。

sampler event `e` の cutoff が 12 の場合、`SamplersInstantiate` は「証明中の sample が実際の
event occurrence の出力そのもの」と保証し、`SamplerContract` は「その同じ sample が cutoff
12、型、support、relation を満たす」と保証する。

具体的な `inputs` と `samplers` を示して二つの contract と terminal bridge を証明できなければ、
Lean の含意は論理的には真でも、実際の実行を一つも保証していない可能性がある。

terminal Result とは、到達した基底 transfer の直後に記録された exact Result である。今回到達する
基底 transfer は fact-store authority、program-family-fact authority、operator authority、identity、
scale の五種類だけである。各 terminal event について、`Witness.honestTerminalCongruence` は、
event 番号で引いた `honestTerminalActual` と、同じ history row に記録された exact polynomial の
評価が `q` を法として一致することを表す。Lean kernel が証明するのは、この field を仮定した条件
付きの定理である。生成器はどの row が terminal かは証明できるが、この field を証明したり作り
出したりはできない。呼び出し側が honest Rust execution の実値を `honestTerminalActual` に入れ、
congruence を示す必要がある。したがって、これは生成 proof が compile しただけでは得られない、
定理の statement と実行を結ぶ modeling assumption である。

固定した selector の honest run について、coefficient reference から使われる各 exact `Result` にも
event 単位の modeling premise が残る。その実値の centered coefficient norm は、同じ `Result`
event に正規化後の authoritative coefficient bound として記録された値以下でなければならない。
`Witness.recordedCoefficientCovers` は、正確な event、frame、owner、normalized terms、coefficient
producer、summary、recorded bound で index されたこの条件を保持する。opt-in Rust projection は、
これらの field が immutable `ProofPayload` と完全に一致することを assert し、生成 statement が別の
row や bound を選ぶことを防ぐ。ただし、この Rust assert は norm の不等式を Lean 内で証明しない。
kernel theorem は、呼び出し側が honest run に対する `Witness.recordedCoefficientCovers` を供給する
ことを仮定した条件付き定理のままである。

したがって、固定 Security0 statement の residual 値は `Env` だけの関数ではなく、selector と
その honest-run `Witness` の両方に依存する関数とする。Source/Cert 側で固定した関数が statement
であり、生成 Proof はその関数について kernel が検査する導出を与える。honest witness をその
関数へ具体化することが実行との correspondence obligation であり、proof が existential な
residual 値を新たに選ぶことはない。

さらに Lean compile は、次の二つを trusted code の監査事項として信頼する。

```text
TrustedSemanticsCorrect
  := 固定 Lean の Value、matrix、interpreter の意味が、
     pinned Rust operational-noise semantics と一致する

CanonicalProjectionCorrect
  := trusted canonical projection が pinned checker を再実行し、受理した Source.json を
     一意に解決した p、q、residual root を含む statement-bearing certificate row へ
     意味を変えず写す
```

これらは generated proof が自由に追加できる仮定ではなく、固定 trust boundary の正しさとして
code review と Rust–Lean differential test で担保する。

## 7. Lean で証明する固定の主張

### 7.1 semantic owner claim と CP0/CP1 の trust split

一回の replay における `Owner` は、scope と expression row の型付き pair である。central または
ordered monomial factor として使う owner は、その replay 内で一つの値を表し、program scope の
場合は program domain 内の各 selector について一つの値を表さなければならない。opt-in Rust
generator は Lean を出力する前に CP0 を検査する。factor owner の coefficient-normalized
exact-zero claim は一致しなければならず、異なる複数の result payload を持つ owner は factor に
現れてはならない。nonfactor owner に対する CP0 の検査対象は、複数の payload の中で exact-zero
claim と別の finite claim が共存する場合だけである。このとき、別の finite claim はすべて、
direct-survivor / sum-after-survivor として認識した fold chain から得た空の exact result でなければ
ならない。それ以外の nonfactor multiplicity は、proof reference が event 単位の claim を指定する
ため theorem-load-bearing ではない。特に singleton coefficient-finite claim と nonempty
exact-finite claim は semantic `Result` / `Transfer` proof の obligation として残す。この判定に
必要な owner、claim、frame、predecessor の identity は生成される typed proof row として保持し、
別の統計用意味論や whole-workload ledger は作らない。

CP1 では役割を分離する。`Env` を参照するのは factor owner だけであり、`ValueClaim` は event
単位の statement のままである。`ExactClaimAt` は、その claim と、それが解釈する正確な owner、
raw terms、summary、`Result` history index を一組にする。このため、算術 proof が別 event の claim
へ黙って差し替わることはない。`Witness` は factor atom を bound するほか、§6.4 の reached-only
modeling bridge を持つ。`honestTerminalActual` は terminal Result event で引き、
`honestTerminalCongruence` はその値を同じ history row の polynomial に結び付ける。許される terminal
transfer は fact-store authority、program-family-fact authority、operator authority、identity、scale
の五種類である。生成器は row lookup を証明して通常の算術 theorem を適用するが、honest execution
との congruence 自体は構成できない。

生成される Lean theorem は、これらの明示的な premise を満たすすべての witness に対して kernel が
検査する条件付き定理である。Lean compile は、honest Rust execution がその witness を供給すること
までは証明しない。CP0 は別途、その document が使う owner key と honest Rust replay の値との対応を
検査する trusted-generator assertion である。この分離によって通常 checker 経路に処理は追加されない。

`InputAssignment` は source access ごとの実際の入力値、`SamplerAssignment` は event ごとの
実際の sampler 値である。`InputContract` と `SamplerContract` は、それらの値が型・範囲・
sampler 条件を満たすことを表す。root の引数は実行者から受け取らず、certificate に記録された
closed root または family domain から内部的に決める。

固定 Lean module は、この vertical slice の ABI である `TallSemantics` を定義する。生成コードが
これを再定義したり、Tall の acceptance path を generic certificate interpreter に置き換えたり
してはならない。最終 endpoint は固定 `TallSemantics.Security0Accepted` である。これは audited
Tall document と immutable event history を、final Result、PreFold、InvocationEnd、residual
function、そして直接の `2 * p * noise < q` に結び付ける。Security0 と Security128 は同じ endpoint
shape を使い、profile parameter と生成 row だけが異なる。

生成 output は `Cert/`、`Proof/`、`Semantic/` の sharded file に分ける。小さな固定 acceptance
module がこれらを import し、完全修飾した `Security0Accepted` の theorem application を kernel に
検査させる。Tall の acceptance path は一つの固定 endpoint と到達した residual semantics だけを
扱う。

`ResidualRoot` は現在の Rust と同じ二種類だけを持つ。

- `closed`: 自由引数を持たない matrix expression
- `family`: Rust の `FamilyDomain`（`u64` の非負半開区間、Lean では `Nat` endpoint）を持つ一引数の
  matrix family

closed root は引数なしで評価する。family selector は `Nat` だけで表し、
`domain.Contains selector` を確認する。family root はこの正確な半開区間内の全 selector に
対して記号的に bound を証明する。root 引数はこの検査済み構造から内部的に決め、
型のない外部 list として受け取らない。

`Security0Accepted` は、各対象 residual の最大 centered coefficient norm を `noise` としたとき、

```text
2 * p * noise < q
```

が成り立つという主張である。centered coefficient norm は、各係数を 0 の周りの代表値として
見たときの絶対値の最大値である。`p` は plaintext modulus、`q` は ciphertext modulus である。
等号は不合格であり、整数除算で弱い条件へ書き換えてはならない。

`Cert` は平文 modulus `plaintextModulus` と暗号文 modulus `ciphertextModulus` を直接
保持する。well-formed certificate は次を満たす。

- `q > 0` および `p > 0`
- residual root は、係数を `q` を法として扱う ring `R_q` 上の closed matrix または matrix
  family
- residual の全 element の ring dimension は certificate の宣言値と一致

二つの modulus のうち `p` は `ResolvedDecoderKind::Threshold`、`q` は
`ResolvedAcceptanceTarget.ciphertext_modulus`、residual root は `ProductionRoots.residual` から、
trusted canonical projection が取り出す。生成 proof の data が別の値を選ぶことはできない。runtime decode は
この theorem の対象外である。

`Security0Accepted` は certificate 固有の数学的な最終地点である。closed root なら狭義不等式を
一つ、family root なら正確な非負半開区間内の全 selector に対して同じ不等式を証明する。
等号は不合格であり、整数除算を使う弱い条件へ書き換えてはならない。

固定 acceptance module は、生成 proof の型が完全修飾された
`TallSemantics.Security0Accepted` と一致することを検査する。runtime decode 結果は導出も検査も
しない。acceptance theorem に `#print axioms` を実行し、許可しない axiom が混入していないことも
確認する。

G1 でレビュー済みの Lean 標準 axiom の許可リストは `propext` と `Quot.sound` の二つだけである。
これは certificate 固有の信頼仮定ではなく、Lean kernel library の基礎である。`#print axioms` の結果に
`sorryAx`、生成または独自の axiom、`native_decide`、またはこの二つ以外の axiom が現れてはならない。
追加の axiom が必要になった場合は、受理前に別途設計 review を行う。

## 8. Certificate のデータモデル

certificate 内 ID は、決定的に割り当てる局所自然数である。ID 自体には意味を持たせず、意味を
決める情報は型付き row に格納する。

```text
Cert = {
  plaintextModulus,
  ciphertextModulus,
  ringDimension,
  expressions : RowTable ExprRow,
  programs    : RowTable ProgramRow,
  sources     : RowTable SourceRow,
  events      : RowTable EventRow,
  indexUses   : RowTable IndexUseLut,
  sliceGroups : RowTable IndexedSliceLutGroup,
  residualRoot
}
```

`residualRoot` は第7節の tagged `ResidualRoot` である。これと二つの modulus、
`ringDimension` が top-level の target data のすべてである。

`residual proof closure` は、`ProductionRoots.residual` だけを始点とする依存関係の推移閉包で
ある。ここで推移閉包とは、root から必要な子を順にたどり、そこからさらに必要な子も
すべて含めた最小の集合を指す。式と program の依存先、その root の評価と bound proof に実際に
必要な source、event、relation、bound fact、index use、synchronized slice group だけを含む。
`ProductionRoots.decoder` は closure の二つ目の root にしない。decoder にしか属さない式、event、
trace、`ThresholdDecode` の意味論や lemma は serialize しない。closure 内の依存先は全て serialize し、
serialize する全 row は closure に属さなければならない。これは certificate projection の
境界だけを定める。既存 Rust コアは現在どおり `ProductionRoots.decoder` を解析し続けてよい。

`Value` は現在の Rust `ResolvedValueType` が必要とする次の variant を正確に持つ。

- `Bool`
- `Int`
- `Real`
- `Bytes`
- `Matrix`
- `Trapdoor`

program と family は `Value` に追加せず、expression row から参照する。

### 8.1 SourceRow

`SourceRow` は、定数、宣言済み protocol input、特定 occurrence に属する unbound input、
producer artifact を、種類を示す tag が付いた一つの datatype で表す。完全な Rust identity、
owner scope、signature、resolved type を保持する。direct row と family row は、対応する Rust
facts から得た optional raw value contract も唯一の場所として保持する。この contract は signed
half-open range、coefficient class、canonical coefficient exclusive upper、polynomial support
upper から成る。constant row は literal value から導ける facts を重複して持たない。

family source もこの同じ `SourceRow` に格納し、その owner scope と signature で所有関係を表す。
family 専用の第二の source ID 空間は作らない。

### 8.2 EventRow

`EventRow` は uniform、Gaussian、sampled hash、trapdoor-public、preimage、gadget
decomposition event を、同様に種類 tag 付きの一つの datatype で表す。各 row は owner scope、
signature、output type、種類固有の完全な descriptor を持つ。optional raw contract に置くのは、
descriptor とは独立して記録された range、canonical coefficient、support facts だけである。
sampler cutoff や decomposition bound は descriptor から重複させない。deterministic hash は sampler
event ではなく、通常の型付き expression operator として表す。

各 descriptor の保存場所は一つだけにする。

- constant と external-input payload は `SourceRow` だけに置く。
- sampler と transform descriptor は `EventRow` だけに置く。
- source や event を読む `ExprRow` は、型付き参照と scoped access 情報だけを持つ。
- source/event identity ではない operator 固有 parameter は、その operator の `ExprRow` に置く。
- `ProgramRow` は signature と body を持ち、参照先 descriptor のコピーを持たない。

`Cert.Valid` は、別 row で descriptor を再定義したり上書きしたりする certificate を拒否する。

### 8.3 Identity の取得

構造化された `PlannedWire`、`ProtocolInputId`、`ProgramOccurrence`、artifact identity、sampler
情報が残っている `lower.rs` の段階で、recorder 用 sidecar に identity を記録する。sidecar は
生成された expression/event ID とこの構造化 identity を対応付ける。

対応が欠ける場合、競合する場合、異なる identity が一つへ潰れる場合は certificate 生成を拒否
する。既存 arena の interning や identity は変更しない。debug 表示文字列や digest で構造化
identity を置き換えてはならない。

deterministic hash の identity は、definition、version、key の長さと bytes、output type、tag
prefix、binary/decimal/little-endian-u64/dynamic tag group とその順序境界、decomposition
parameter を含む。owner scope の下で完全に評価した型付き query が同じ場合だけ同一とする。

到達した `GadgetDecompose` transform に元の `SampleEventId` がない場合は、型付き scoped
transform expression と正確な parameter から canonical event reference を導出する。
`PackPolynomialCoefficients` に必要な意味論と lemma が coverage gate に揃うまでは、そこへ到達
した certificate の生成を拒否する。

## 9. Interpreter と certificate の構造検査

`interpreter` は、certificate の式を実際の値へ評価する固定 Lean 関数である。現在の Rust
`ExprArena` と `ProgramArena` のうち、対象から到達可能な意味論を正確に再現する。

Lean の関数を total にするため、評価回数の上限を表す `fuel` と fallback を持たせる。
certificate の validity proof により、正しい入力では fuel が尽きず fallback に到達しないことを
証明する。

一つの大きな式を再帰的に一度で簡約せず、通常 node、`ProgramCall`、`IndexUse` ごとの
fuel-stable な一段 lemma を用いる。Rust は子の評価結果を memoize し、隣接する fuel 値に対応
した通常の Lean `have`、すなわち名前付きの局所中間結果を出力する。

`Cert.Valid` は少なくとも次を検査する。

- expression 参照が dense topological order にあり、expression child、program call、producer
  artifact、event operand、hash query、relation link を合わせた dependency graph が非巡回で
  あること
- operator の arity と `Bytes` を含む全 `Value` type が正確であること
- matrix の係数数、modulus、ring dimension、rows、columns、logical shape が正しいこと
- residual root が closed または一引数 family として正しく分類され、family なら正確な非負
  `FamilyDomain`、closed なら自由引数がないこと
- plaintext/ciphertext modulus が正で、residual の modulus と ring が正確に一致すること
- program signature、引数 ownership、family domain、call substitution が正しいこと
- source、event、relation link、index-use row の owner が一意で、serialize した全 row が
  residual proof closure に正確に属すること
- hash descriptor と scope substitution が完全であること
- slice、係数、table reference、index consumer が範囲内であること
- 検査済み dependency DAG から十分な fuel が得られること

Tall target では、canonical projection が frozen Rust source から再構成した pinned checker run
の `ProductionRoots.residual` から、宣言済み residual root と、`u64`/`Nat` endpoint の正確な
`FamilyDomain` を選ぶ。
`Cert.Valid` はその構造と residual-only closure を検査する。どちらも実行時に渡す仮定ではない。

Rust の物理的な `MatrixLayout` は source artifact に残し、既存 Rust arena が引き続き検査する。
Lean は interpreter が使う logical routing と shape をすべて検査するが、使わない stride の物理
検査を重複実装しない。

## 10. Input と sampler の contract

### 10.1 SourceAccess と InputAssignment

`SourceAccess` は「どの source を、どの呼び出し文脈と selector で読んだか」を表し、次を含む。

- source reference
- normalized owner invocation
- 検査済みの scoped substitution
- family access の場合だけ存在する、評価済みの optional `Nat` family selector

`InputAssignment` は `SourceAccess` から `Value` への total function である。同じ owner invocation
内でも、同じ family を異なる selector で読めば別の access になる。

`Cert.Valid` は substitution と owner program signature の一致、および `domain.Contains selector`
を検査する。`InputContract` は valid な各 access の resolved type と、Rust analysis が実際に
使った生の事実だけを要求する。

polynomial に関する次の三種類の情報を混同しない。

- centered coefficient bound: 中心化した係数の絶対値上界
- canonical coefficient exclusive upper: canonical 係数が入る半開区間の上端
- polynomial-support upper: それ以降の係数が 0 になる位置の上端

係数から index を取り出すときの domain 根拠に使えるのは canonical coefficient exclusive
upper だけである。support upper は owner、source、family access を正確に選んだ値で、ring
dimension 以下でなければならず、それ以後の polynomial position が 0 になることを表す。
現在の integer fact は signed half-open range として記録する。family domain と family-access
selector は `Nat` として扱うが、他の integer fact の signed 表現は変更しない。派生する range、support、
sparsity、constant 性は Lean で証明し、入力条件として追加しない。

### 10.2 SamplerAssignment と SamplerContract

`SamplerAssignment` は event reference と owner arguments から `Value` への total function である。
`SamplerContract` は tagged event row の種類に応じて、現在の Rust event と同じ type、support、
cutoff、relation 条件を要求する。

gadget decomposition では「出力の bound」と「入力へ戻る recomposition relation」を別の条件と
して扱う。

ここで `D` は decomposition で得た digit matrix、`G` は digit から元の値を再構成する gadget
matrix である。`ExactZero` は Rust が「厳密に 0」と証明済みであることを表す fact である。

- regular decomposition の bound は `max(base / 2, 1)`
- small decomposition の bound は `base - 1`
- `G * D ≡ input` は、Rust コアがその event に relation を登録した場合だけ利用可能
- 現在の small decomposition の relation は `ExactZero` 適用条件を満たす場合だけ記録

preimage、hash、decomposition の relation でも、event、scope、argument substitution、type、
descriptor の条件を Rust が登録したものより強くしてはならない。

Tall の universal preimage relation では、public matrix `B` は selector に依存せず、正しい式は

```text
B * K(i) = T(i)
```

である。validity proof は `B` の program root が program argument に到達せず、selector を
変えても同じ値になることを再帰的に証明する。`K(i)` と `T(i)` は同じ `Nat` family selector
を使う。他の signed integer arithmetic fact の意味は signed のまま保つ。

## 11. Index を安全に使うための LUT

### 11.1 Index consumer と frontier

`index consumer` は、計算された整数を実際に index として使う操作である。対象は次に限定する。

- family lookup
- `ExplicitElement` の operand 0
- `IndexedSlice` の動的な row start、row end、column start、column end

整数型の式を走査して推測するのではなく、これらの generic API で型付き consumer を登録する。
hash tag、scale factor、comparison、固定 descriptor は、整数であっても index
consumer ではない。

各登録済み use について、`IndexUsePlan.index : ExprId` が Lean へ射影する唯一の計算 root
である。projection は既存の typed expression table をたどり、Rust の
`evaluate_typed_index` と同じ `Add`、`Sub`、`Mul`、`Div`、`Rem`、`Negate` semantics を使う。
typed expression table が index 計算の唯一の表現であり、未対応の expression または演算は推測せず
fail closed とする。

ある index 計算が依存する有限入力の集合を `frontier` と呼ぶ。異なる scope にある frontier は
型付き `ScopedExpressionRef` と、合成済みの明示的 `ProgramCall` substitution で表す。

### 11.2 IndexUseLut

`IndexUseLut` は、frontier の可能な入力組と、そのとき得られる index 出力をすべて記録する有限表
である。各 LUT の identity は次を含む。

- owner と canonical consumer
- 対象 operand と use kind
- 正確な `ExprId` index 計算 root
- 出力が入るべき domain
- 固定 parameter と optional group
- 順序付き frontier identity と各 domain

consumer ごとの要求 domain は次のとおりである。

- family lookup: 非負 endpoint の `FamilyDomain`
- `ExplicitElement`: 正確な branch domain
- dynamic row start/end: `[0, input.rows + 1)`
- dynamic column start/end: `[0, input.columns + 1)`

Rust は記録した frontier の順序と domain、および固定 evaluator の意味論で有限直積を完全に
列挙する。frontier が空の closed computation では、0 個の軸の直積を要素数 1 とし、row を
一つだけ持つ。各 raw tuple と output は、出力前に同じ Rust typed-index evaluator と完全一致
することを検査する。LUT が別の LUT 出力に依存する場合は、dependency DAG の topological
order で出力する。

domain の欠落や競合、dependency cycle、integer conversion の overflow、0 除算、evaluator の
panic、評価失敗が一つでもあれば certificate 生成を拒否する。

Lean は row 数、mixed-radix で復元した入力 tuple、実際の tuple が domain 内であること、全出力が
consumer range 内であることを検査する。production table 全体を一つの巨大な `rfl` や `decide`
で簡約せず、row または subtree ごとの有界 proof を合成する。

`IndexedSlice` の四つの動的 index は、同じ frontier を持つ一つの
`IndexedSliceLutGroup` にまとめる。各 row で start/end の順序と、slice の正確な rows/columns を
同時に証明する。

一つの LUT の row 数は、frontier の各 domain cardinality の積である。row 数、bytes、時間、
domain による意味上の cutoff を設けず、表の一部だけを証明済みとして扱わない。Lean は typed
`ExprId` から各 row を再構成し、raw tuple/output の対応と consumer range を検査する。storage
は lossless streaming、厳密な deduplication、compression を利用してよいが、評価する tuple を
変えてはならない。`IndexedSlice` の四つの動的 index は同じ frontier を持つ
`IndexedSliceLutGroup` に残し、順序と extent を一緒に検査する。検査していない trusted range や
別の index-expression 表現を根拠の代用にしない。

## 12. Rust の計算を Lean で再生する方法

### 12.1 PolynomialNF と relation

`PolynomialNF` は、residual を「符号付き係数と、順序を持つ factor の積の和」として表す Rust
側の正規形である。Lean は別の normalizer で答えを探すのではなく、Rust が行った局所変形を
順番に検査する。

`relation` は、たとえば sampler 出力と入力の間に成立する、Rust が登録した等式または合同関係
である。relation は登録時の event、scope、引数、型、適用条件が一致するときだけ使用する。

`MatrixModEq` は、二つの matrix が modulus の下で同じ係数を表すという関係である。

証明は通常の Lean `have` を次の順で並べる。

この `have` 列そのものが Lean に渡す proof plan である。Rust 内部だけで使う replay record は
Lean へ serialize せず、certificate に別の step graph や plan interpreter を追加しない。

1. `evalClosedResidual` または `evalFamilyResidual` から現在の ordered-product `PolynomialNF`
   まで、局所的な意味の等式を再生する。
2. Rust コアが登録した、scope と条件が正確に一致する relation だけを適用する。
3. 正負の multiplicity が一致する項を厳密に相殺する。
4. 相殺後に残った bounded term と、その根拠に必要な subexpression だけを評価する。
5. matrix の well-formedness を使い、`MatrixModEq` をまたいで bound を移す。
6. `2 * p * noise < q` を直接検査し、`TallSemantics.Security0Accepted` を完了する。

### 12.2 Recorder が保持する情報

recorder は各 `have` を生成できるだけの次の情報を保持する。

- 変形前の正確な polynomial
- 加算・乗算 context の prefix と suffix
- 適用した rule の identity と parameter
- 相殺した係数と残った項
- coefficient merge と survivor fold
- fold 前の最終 polynomial

0 ではない term の factor 列は nonempty とする。factor のない非零 term を作ったり、zero matrix
や架空の汎用 identity matrix を乗算単位として使ったりしない。

### 12.3 Cache と frame lifecycle

honest run の frame lifecycle は、frame の生成、owner-local cache state、event history、frame の
終了を時系列どおり記録し、対応する Rust run と照合する。現在の Tall profile では specialization
cache hit は 0 件であるため、汎用 specialization-cache replay や cache の owner 間移植を実装
する必要はない。将来到達した cache reuse についても、その consuming frame 自身の relation、fold、
merge evidence を出力し、別 owner の記録を移植したり Lean 側で normalization search を行ったり
しない。owner-local の lifecycle event が不足する場合は certificate 生成を失敗させる。

### 12.4 Noise bound

Lean の bound lemma は、受理された実行で Rust が使った rule を近似せず再現する。ここで
`Large` は有限の利用可能な上界を得られなかったことを表す Rust の分類である。対象には次を
含む。

- `ExactZero` による消滅
- `Large` または missing bound の拒否
- scalar broadcast
- polynomial-support factor
- constant-polynomial factor
- tensor rule
- matrix の正確な inner dimension
- Rust が実際に利用した、証明済み zero-row reduction
- CRT、すなわち Chinese Remainder Theorem による reconstruction coefficient
- `bound.rs` と `normal_form.rs` の operator 固有 transfer rule

安全側であっても不要に大きい bound へ置き換えない。bound が大きくなると Rust が受理した
parameter を Lean が拒否し、同じ判定を証明したことにならないためである。

certificate の `MonomialProduct` theorem API は、各段の product factor を 1 とする
nonempty fold である。相殺後に残る 0 ではない coefficient magnitude は、直後の独立した
`Scale` event で適用する。Rust 内部の product helper が持つ任意 factor は certificate API
には含めず、G0 が受理する `MonomialProduct` call site の factor はすべて 1 とする。

G0 では、正確な residual proof closure の各 `ObservedCoverage` row から、private な residual
coverage row を一対一で作る。この変換は意味論の検索に使わず、通常の checker 経路も変更しない。
各 row は kind、正確な count、sort 済みで重複のない site をそのまま保持し、repository-relative
で安定した Rust item と、`CheckedLean`、`G2LeanObligation`、`RejectBeforeGeneration` のいずれかを
付ける。`CheckedLean` は、正確な semantics lemma と transfer lemma が既に Lean で compile 済みの
場合だけ使用する。それ以外の到達可能な operator、transform、sampler、relation、bound は明示的な
G2 obligation とする。regular/small gadget decomposition と、plain/decomposed/small-decomposed hash
sampling はそれぞれ別の row にする。

`ThresholdDecode`、`BoundAuthority::Unavailable`、raw `EventKind::Trapdoor` は、matrix 生成または
canonical event projection より前に拒否する。decoder-only expression は residual closure に含めない。
したがって schema version 6 の CPU evidence に reject-tagged row が現れることはない。

到達した操作に lemma がない場合、`Large` が残る場合、bound がない場合、side condition を証明
できない場合は certificate 生成を拒否する。

## 13. Recorder、生成物、CI

recorder は opt-in の追加機能とする。既存の construction、relation、bound decision point に観測
call を追加してよいが、ordering、interning、fact、normalization、relation selection、bound、
acceptance、diagnostic を変更してはならない。

同じ request を recording off と on で実行し、recorder 専用 metric を除く Rust report bytes と
core counter が一致することを differential test で確認する。

比較対象は、通常 checker が生成する意味論的な report field と決定的な core counter である。
elapsed time、RSS/GPU 観測値、recorder 専用 metric は checker の意味論ではないため、この一致
比較から除外する。

`docs/correctness/tall-operational-noise-certificate-g0.json` は source construction と実現可能性
だけを確認する決定的な G0 review evidence である。これは G3 certificate artifact ではなく、
`Source.json` や acceptance artifact として使用してはならない。

evidence は、網羅的な静的 constructor で作る二つの Tall profile をこの順序で使用する。
security-0 profile は、乗算 1 回、CRT depth 7、`log2(n) = 5`、28-bit CRT modulus、自動選択される
6-bit nested-RNS p basis、14-bit gadget base、未 reduction 乗算 2 回、scale 64、error sigma 4、
trapdoor sigma 4.578 に固定する。security-128 profile は CRT depth 20 と `log2(n) = 15` だけが
異なり、review 済みの静的 security lower bound は 177 bit である。これらの profile は環境変数を
読み込まず、parameter search や estimator を実行しない。このファイルは gate 前の CPU 観測で
あり、Tall の実行結果ではない。

schema version 6 では、正確な `N` の row と descriptor inventory を、typed certificate schema と
共通の canonical statement-row projection から導出する。特に event row は、到達した各 gadget
decomposition を正確な closed scope または program scope で含み、expression row は対応する event
reference だけを保持する。これにより expression、program、source、event の count は一つの authority
に従い、runtime acceptance と通常 checker 経路は変わらない。

clean-room regeneration は、同じ commit にある `Source.json` recipe から、独立した完全生成を
二回実行する。各 run は request、Rust report、typed residual closure、`Cert/`、`Proof/`、
`Semantic/` の全生成 file を再構築し、path-relative に byte 単位で比較する。比較の前に、
直接 Rust run と Source 再構成 run の report field と決定的な core counter が一致することを確認する。
`Source.json` に derived な normal form、LUT output、bound ledger、proof を入れず、recipe が
statement の入力を一意に決める。

G3 の pre-serialization/render gate では、constant-polynomial fact と polynomial-support fact を
authoritative な owner/source/family-selected fact から再導出し、記録された値と一致しなければ拒否する。
recorder が出力したというだけで、記録済み fact を信頼してはならない。

CI は固定 acceptance module を compile し、`#print axioms` の結果に `sorryAx`、生成 axiom、
`native_decide`、許可していない非標準 axiom があれば失敗する。

## 14. 計算量

次の三つの量で certificate 生成の大きさを表す。

- `N`: residual proof closure 内の expression、program、source、event row の正確な論理 item 数
- `T`: 出力する proof/context payload の正確な論理 item 数
- `L`: residual proof closure 内の exhaustive index-use table の正確な大きさ

`T` は `have` の個数ではない。serialized predecessor polynomial、prefix/suffix context、rule
parameter、coefficient merge、survivor fold、final polynomial の全量を含む。

`L` は row 数だけでなく、各 row の tuple と proof payload も含む。許可する size 指標は、これらの
論理 item 数、canonical encoded byte 数、artifact が存在する場合の実生成 artifact byte 数、
recorder/generator が保持する論理 item の peak（任意で canonical retained byte 数）だけである。
`size_of`、RSS、elapsed time、benchmark estimation、runtime/GPU estimate、現在の checker との
benchmark 比較は明示的に除外する。

既存 Rust checker の計算を除き、recording と rendering の時間・空間計算量を

```text
O(N + T + L)
```

とする。`N` に matrix family や parallel-loop lane の論理個数を含めない。

## 15. 実装を進めるための gate

各 gate は、その段階を通過するまで次段階へ進まないための受入条件である。

### G0: 既存の構造境界

既存 Rust 通常経路、SchemaV1、G0 の index-use 列挙は変更しない。opt-in projection は typed
`ExprId`、frontier の順序と domain、SliceGroup row を保持し、raw LUT の tuple/output を
Rust evaluator と完全一致させる。

### G1: 固定 Lean core

次の固定 module と toy fixture を build し、axiom scan を通す。

- `lean/Mxx/Certificate/OperationalNoise/Core.lean`
- `lean/Mxx/Certificate/OperationalNoise/Fixtures.lean`

closed certificate、index ごとに異なる sampler 値を持つ family-root certificate、狭義不等式、
malformed data の拒否を検査する。gate は `cd lean && lake build
Mxx.Certificate.OperationalNoise.Fixtures` と axiom scan である。

### G2: 到達した semantic replay

選択した run の residual closure に現れた semantic theorem だけを、Lean compile が指摘した順に
追加する。対象は Add/Sub/Product/Tensor merge、relation の prefix/source/suffix 再構成、
`BoundTransfer`、`SurvivorFold`、`PreFoldPolynomial → InvocationEnd` である。未到達 variant の
lemma、全体 coverage matrix、completeness ledger は作らない。到達したが未対応の操作、bound、
side condition は fail closed とする。

### Security0 fixed acceptance

同じ generator から `Cert/`、`Proof/`、`Semantic/` を生成し、直接 Rust run と Source 再構成 run
の report/core-counter parity、到達 residual closure の identity/relation/bound/frame evidence、
固定 `TallSemantics.Security0Accepted` の kernel compile、許可された axiom を確認する。

### Security128

同じ generator と ABI を Security128 に適用し、fixed acceptance compile、正確な deterministic
count/byte metrics、二回の完全生成の path-relative byte equality を確認する。elapsed time、RSS、
`size_of`、benchmark estimate、runtime/GPU estimate は記録対象外である。

現在は Security0/Security128 の最終 evidence gate に近い実装途中である。構造 replay、固定 Lean
semantics、到達 semantic replay、generator ABI は実装済みで、fresh parity、exact metrics、
clean-room evidence を完了確認として残す。Rust compile や unit test だけでは certificate の受理に
ならず、条件付き Lean theorem だけでも特定 runtime の外部条件を証明しない。

## 16. 完了条件

この設計の実装が完了したと言えるのは、少なくとも次をすべて満たしたときである。

1. 既存 Rust checker の意味論、受理判定、通常時性能、SchemaV1 が変わっていない。recording
   off/on の Rust report と決定的 core counter も一致する。
2. Security0 と Security128 の generator が、同じ固定 ABI と profile recipe から生成できる。
3. 両 profile の fixed `TallSemantics.Security0Accepted` が Lean kernel で compile し、axiom
   inventory が `propext` と `Quot.sound` の許可範囲内である。
4. 同じ commit の `Source.json` recipe から独立した完全生成を二回行い、全 generated path の
   bytes が一致する。生成された約 720 MB の Lean artifact は commit 不要である。
5. reached residual closure の全 index use、LHS/RHS、owner invocation、event、relation、bound、
   frame lifecycle が typed identity を保って Lean proof に接続される。specialization cache hit
   は実測 0 件であり、汎用 cache support は要求しない。
6. exact row/file/count/byte metrics と Rust–Source parity が記録され、elapsed time、RSS、
   `size_of`、benchmark estimate、runtime/GPU estimate に依存しない。
