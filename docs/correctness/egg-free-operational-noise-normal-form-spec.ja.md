# egg 非依存 Operational Noise 正規形仕様

## 1. この文書の位置付け

この文書は、Operational Noise checker の exact signal 相殺と noise 上界計算を、egg/e-graph に依存せず実装するための仕様である。ここでいう **exact signal** とは、暗号化誤差ではなく、公開鍵、secret、gadget、preimage などからなる厳密な代数項である。これらは登録済みの数学的関係によって完全に相殺されなければならない。

この文書は将来実装の設計仕様であり、現在の実装が完成済みであるとは主張しない。現在の egg 実装にある final-leaf filter は移行中の一時的な安全策であり、egg と一緒に削除する。

用語は、最初に直感的な意味と役割を説明し、その後に形式的な定義を与える。

## 2. 目的と非目的

### 2.1 目的

1. Tall と Diamond WE の honest な protocol 定義で、すべての exact signal を正しく相殺する。
2. 残った noise だけについて、過小評価しない有限の係数上界を計算する。
3. 同じ入力から必ず同じ正規形を得る。
4. 行列積の深さやループ回数に対して、不必要な全探索、Cartesian selector 展開、e-graph saturation を行わない。
5. 新しい cache database を作らず、既存の lowering 結果、symbol table、relation registry、型・bound 契約を再利用する。
6. 実装者が protocol 名、node 番号、fixture の値に依存した例外処理を追加できないほど具体的に規則を定義する。

### 2.2 非目的

- 悪意を持って循環させた protocol の受理。
- 実行時の観測値や候補値からの bound 推測。
- 任意の非可換代数を自動証明する一般-purpose CAS。
- relation が登録されていない式の経験的な相殺。
- selector の全組合せの列挙。
- 後方互換性のための旧 egg 表現の保持。

## 3. 初学者向け用語集

- **DAG**: 同じ部分式を複数箇所から共有できる、有向非循環グラフ。式木をそのまま複製せず、同じ計算を一度だけ処理するために使う。
- **normal form（正規形、NF）**: 同じ数学的な式を必ず同じ形で表したもの。比較と相殺を構造比較だけで行える。
- **monomial（単項式）**: 加算を含まない、順序付き factor の積。行列積なので factor 順序は交換しない。
- **factor（因子）**: monomial を構成する一個の値。public matrix、secret、gadget、scalar など。
- **exact signal**: 近似誤差ではなく、relation によって厳密に消えるべき代数項。
- **bounded-only**: 構成要素がすべて有限係数上界を持つ noise 項。内部の式を一個の安全な上界へ要約できる。
- **selector**: Switch または dynamic family access で、どの case を選ぶか決める整数値。
- **relation**: `B*K=P` のように、登録済み identity が一致した場合だけ使える厳密な等式。
- **producer DAG rank**: relation target を生成した producer DAG 上の整礎な順位。source より target が必ず小さくなるよう登録時に検証し、再帰停止性を証明する。
- **fail-closed**: 根拠が不足したときに推測して続行せず、型付きエラーで拒否すること。
- **first-Large witness**: 最終 residual が Large のとき、決定的な走査順で最初に見つかる原因 atom。診断専用で、永続 field や cache に保存しない。

## 4. 全体アーキテクチャ

### 4.1 直感的な流れ

checker は、式を何度も別表現へ探索するのではなく、一度だけ構造を読み、exact signal を「項の和」の形へ正規化する。各項では因子の順序を保存する。noise だけからなる項は頻繁に一つの要約 noise 項へまとめる。最後に exact signal が残っていないことを確認し、要約 noise の上界を threshold と比較する。

### 4.2 固定 pipeline

処理順は次で固定する。後段を先に実行してはならない。

1. 入力契約と owner-aware identity の解決。
2. source bound と integer domain の解決。
3. 式 DAG の bottom-up 正規化。各積は後述する deterministic product constructor だけで作る。
4. 各演算直後の bounded-only 集約。
5. exact monomial の canonical 化と同符号・逆符号の相殺。
6. 最終 residual が bounded-only であることの確認。
7. 厳密 threshold 判定。

この順序を e-graph saturation にしてはならない。一方、relation target の正規化と、その結果が新たに露出させる Switch/relation の処理は product constructor の定義に含まれる有限再帰である。「全 phase を一回だけ実行する」という意味ではない。

## 5. 再利用する既存構造と削除する egg 部品

### 5.1 再利用するもの

- lowering が生成する owner-aware atom/source identity。
- symbol table と、型、matrix metadata、integer domain の解決結果。
- `RelationRegistration` が持つ source、expected public、target、trapdoor、ordered indices。
- protocol/artifact binding による producer-output bound の継承。
- 既存の `BoundClass` と matrix metadata の意味論。
- sampler descriptor と sampler interner。Gaussian/UniformInterval のために既存 interner を再利用してよいが、relation は登録しない。
- stored family cases。family の論理的な全要素を再生成してはならない。

### 5.2 削除するもの

- operational-noise 用 egg language、e-class、rewrite runner、saturation loop。
- extraction cost による raw relation lhs の選択。
- structural preference、selected-redex epoch、e-class alternative の再抽出。
- relation 適用後に同じ e-class から raw `B * K` を再選択する経路。
- final-leaf filter。これは egg の同値 class から raw lhs が戻ることだけを防ぐ暫定策である。
- source hash が旧 Lean checker の生成物だけを守るなら、その operational checker 用 check。
- Lean operational checker 本体と、その生成・連携専用コード。

削除 inventory は最低でも次を名前で確認する。

- `crates/correctness/src/operational_noise/extract.rs` と `ProposalCost`。
- egg runner、normalization epoch、selected phase、structural preference。
- relation searcher/applier と replacement materialization。
- e-class analysis merge と relation provenance merge。
- rewrite ownership budget、reservation counter、saturation iteration budget。
- operational checker からの egg Cargo dependency。

削除前に、それぞれの責務が新 pipeline のどの関数へ移ったかを migration ledger に一対一で記録する。

## 6. Canonical PolynomialNF

### 6.1 直感的な意味

**PolynomialNF** は、行列式を「符号付き monomial の和」として保持する小さな正規形である。monomial は「順序付き因子列」であり、行列積の因子順序を勝手に交換しない。完全に bounded な項は、内部構造を捨てて一個の bounded noise atom にまとめる。

### 6.2 形式的定義

```text
PolynomialNF = {
  exact_terms: ordered map MonomialKey -> SignedMultiplicity,
  bounded_summary: ExactZero | Bounded(MatrixBound),
}

MonomialKey = ordered list of ExactFactor
SignedMultiplicity = nonzero signed integer
```

`MonomialKey` の比較順は、factor identity の辞書順とする。挿入順や hash iteration 順を使わない。同じ key の係数を加算し、0 になった key は即座に削除する。

`bounded_summary` は数学的な式の同値表現ではなく、その地点までに集約した noise の安全な上界である。後から Large と乗算して内部構造が必要になるケースはサポートしない。したがって、bounded-only 項を要約した後に exact/Large factor と乗算する式は fail-closed とする。Tall と Diamond WE の受理対象 protocol はこの形を必要としないことを acceptance test で固定する。

### 6.3 ExactFactor identity

直感的には、ExactFactor identity は「同じ見た目」ではなく「同じ所有者の同じ runtime 座標にある同じ値」を表す。

形式的には、atom identity は少なくとも次を含む。

```text
AtomIdentity = (
  source owner,
  source kind,
  output port,
  coordinate_binders,
  ordered runtime Atom.indices,
  public/target/layout identity,
  optional trapdoor identity
)
```

runtime coordinate は binder の個数や位置だけで推測しない。`coordinate_binders` と ordered `Atom.indices` の組を使う。canonical comparison が必要な既存 ID は、所有者を失わない canonical resolver を一回通す。

trapdoor は protocol input でもよい。relation 比較では trapdoor の有無だけでなく、同じ input owner と同じ ordered coordinates であることを要求する。

## 7. 正規化規則

### 7.1 Zero

`ExactZero` は行列全体が厳密に 0 だと証明された状態である。

- `0 + X = X`
- `-0 = 0`
- `0 * X = X * 0 = 0`
- integer scalar `0 * X = 0`
- CRT reconstruction coefficient が 0 の入力は、その入力が Large でも寄与 0。

zero annihilation は Large 判定より先に行う。

### 7.2 Add と Negate

- `Add` は子の `exact_terms` を key ごとに加算する。
- bounded summary 同士は係数上界を加算する。
- `Negate` は exact multiplicity の符号を反転する。bounded 上界は変えない。
- `X - X` と `X + (-X)` は、同じ canonical monomial key なら必ず相殺する。
- 構造が曖昧だからという理由で複数の結合順を試行してはならない。Add は一度 flatten し、ordered map に入れる。

### 7.3 deterministic product constructor と全面展開

すべての積は、次の一個の constructor でのみ作る。各手順を省略または並べ替えてはならない。

1. すべての child を先に `PolynomialNF` へ正規化する。
2. child の Add を分配し、積を ordered factor list へ flatten する。
3. 現在見えている Switch の共通 prefix、suffix、加算項を外へ出し、Switch scope を最小化する。
4. ordered factor list を左から走査し、最左の applicable checked relation を一つ選ぶ。
5. relation target を再帰的に `PolynomialNF` へ正規化する。
6. 元の prefix と suffix を、target の各 monomial へ順序を保って再接続する。
7. target によって新しく露出した Switch を再び最小化する。
8. central scalar の canonicalization、同一 monomial の係数加算、逆符号相殺、bounded-only fold を行う。
9. relation が残る場合は手順 4 へ戻る。ただし後述する rank multiset が厳密に減る場合だけ続行する。

複数の形を試して最良候補を選んではならない。この constructor の出力が唯一の canonical result である。

積は child の exact monomial を順序を保って分配する。

```text
(A + B) * (C + D)
  -> A*C + A*D + B*C + B*D
```

`A*C` と `C*A` を同一視しない。行列積の結合だけを flatten し、`(A*B)*C` と `A*(B*C)` は同じ ordered factor list `[A,B,C]` にする。

全面展開中も、各二項乗算の直後に bounded-only monomial を `bounded_summary` へ集約する。exact signal 項数は practical range に留まるという受理対象の前提を採用する。根拠のない項数 budget をハードコードして拒否してはならない。allocation failure、整数 overflow、型不整合はエラーとする。

### 7.4 central scalar

**central scalar** は、対象 ring/matrix 積の左右どちらへ移しても数学的に同じ定数多項式 scalar である。

- central であることが型と metadata から証明された factor だけを monomial の既定位置へ移す。
- scalar key は値と owner-resolved matrix/ring type を含む。
- 非定数多項式、矩形行列、秘密行列、gadget を central と仮定しない。
- scalar の積は、整数または仕様で明示された quotient-ring arithmetic の場合だけ数値的にまとめる。modulus、代表元規約、型が一つでも未指定なら数値 fold せず、scalar identity の決定的な順に保持・sort する。0 が厳密に証明できた場合だけ zero annihilation を再適用する。

### 7.5 bounded-only aggregation

**bounded-only monomial** は、全 factor が有限上界を持ち、exact/Large signal factor を一つも含まない項である。

各 Add、Negate、Multiply、Tensor、CRT、Switch/Select の構築直後に bounded-only 項を一個の `bounded_summary` へまとめる。まとめる際は演算規則に従って上界を先に計算する。元の bounded 式木は保持しない。これにより全面展開で noise 項数が増え続けることを防ぐ。

### 7.6 exact preimage relation `B * K = P`

直感的には、登録された public matrix `B` と、その trapdoor から作った preimage `K` の積を、登録 target `P` へ置き換える規則である。sampler error をこの等式へ勝手に追加しない。

形式的には、ordered monomial 内の隣接境界 `[..., B, K, ...]` に対し、次の full match key が一意に一致する場合だけ置換する。

relation の full match key は次の tuple 全体である。

```text
(source, ordered indices, public, target, matrix type, layout,
 trapdoor, selector identity and reachable-case mapping)
```

この key の全 field が一致する registration だけを applicable とする。該当 0 件は「この境界には適用不能」でありエラーではない。完全に同じ target を表す重複 registration は一つへ deduplicate する。異なる target を持つ候補が 2 件以上なら `AmbiguousRelation` として fail-closed にする。

置換は一方向 `B*K -> P` だけである。逆方向へ展開しない。prefix と suffix は順序を保つ。

```text
prefix * B * K * suffix
  -> prefix * P * suffix
```

一つの monomial に複数の独立境界があれば、product constructor が常に最左の applicable 境界を処理する。`E_B * K`、`K * B`、異なる public、異なる座標は置換しない。

relation の停止性は raw な境界個数では判定しない。各 applicable relation source は producer DAG 上の自然数 rank を持つ。現在の全 pending relation の rank を multiset にし、降順 vector として比較する。target を再帰正規化した後の rank multiset が、適用前より辞書式に厳密に小さい場合だけ再帰を続ける。rank は登録時に producer DAG から計算し、名前、node 番号、式サイズから推測しない。同 rank の循環または増加は型付き termination error とする。

### 7.7 Switch scope minimization

**Switch scope minimization** は selector に依存しない共通計算を cases の外へ出し、Switch が本当に選ぶ部分だけを残す規則である。

同じ selector の各 case について、ordered prefix と ordered suffix の最大共通部分を一回計算する。

```text
Switch(s, A*X0*D, A*X1*D)
  -> A * Switch(s, X0, X1) * D
```

加算でも selector 非依存の同じ canonical term を外へ出す。

```text
Switch(s, A+B0, A+B1)
  -> A + Switch(s, B0, B1)
```

`Switch(s, case0*G, case1*G)` の `G` が selector 非依存なら必ず外へ出す。これは相殺より前に完了させる。

異なる selector 同士を Cartesian 展開しない。同じ selector の Switch 同士だけを case-wise に合わせる。selector domain から reachable でない case は参照しない。reachable case の対応が一意でない場合は fail-closed とする。

### 7.8 family

family は runtime に選ばれる候補集合である。論理上の全 index を列挙せず、lowering が一度保存した cases を使う。

- static access は指定 case だけを使う。
- dynamic access は validated integer domain 内で reachable な stored cases の最大を取る。
- shared affine template がレビュー済みなら endpoint rule を使ってよい。
- selector の Cartesian product を作らない。
- nested family は Tall/Diamond の acceptance fixtures が必要とする深さだけを、明示的な stored-case nesting として扱う。暗黙の flatten や全組合せ展開はしない。

### 7.9 全 operation の PolynomialNF transfer

次の表が operation ごとの完全な dispatch 表である。「保持」は exact monomial key と bounded summary の両方へ同じ構造操作を適用することを意味する。表にない operation は推測せず `UnsupportedOperation` にする。

| operation | `PolynomialNF` の exact 処理 | bounded 処理と検証 |
| --- | --- | --- |
| Zero | 空の exact map | `ExactZero` |
| Add/Sub/Negate | key ごとの符号付き加算 | 上界加算、Negate は上界不変 |
| MatrixMultiply | 7.3 の constructor | zero-first、`K*R*A*B`、直後に fold |
| Transpose | 型を検証して `T(A+B)=T(A)+T(B)`、`T(-A)=-T(A)`、`T(A*B)=T(B)*T(A)`、`T(T(A))=A` の順に正規化する。それ以上分解できない atom だけに transpose view identity を付ける | 上界保持、rows/columns を交換 |
| Slice | owner-resolved slice spec を exact identity に保持 | 上界保持、範囲と output shape を検証 |
| Concat | axis、arity、各 input shape、対応位置がすべて一致する pointwise Add に限り、対応位置ごとに分配する。それ以外は axis と ordered inputs を一個の canonical structural factor として保持する | reachable inputs の最大、axis/shape 検証 |
| Tensor | ordered tensor factor pair を分配するが matrix inner-sum factor を加えない | zero/Large-first、polynomial factor `R` |
| exact integer scale | scalar 0 は zero。証明済み central scalar だけ 7.4 で移動 | `abs(s)*B` |
| interval integer scale | exact signal へ掛かる場合は未確定 scalar identity として保持し、相殺に使用しない | `max(abs(min),abs(max))*B` |
| CRT recompose | nonzero reconstruction coefficient を exact central scalar として各 input に掛け、Add する | `sum abs(c_i)B_i`、zero coefficient は入力を読まず 0 |
| LiftConstantPolynomial | integer exact/domain identityを定数多項式 factorへ変換 | `max(abs(min),abs(max))`、constant metadata=true |
| PackPolynomialCoefficients | ordered Bool bit identities と bit weight を持つ exact pack factor。一般 polynomial 積へ読み替えない | 8.6 の bit bound 式、shape/bit count を検証 |
| HashPlain | query identity と ordered arguments を exact atom identity に保持 | authoritative hard range がなければ `Large` |
| Select | selector identity と ordered cases を Switch と同じ canonical form にする | reachable cases の最大、domain 検証 |
| FamilyGetStatic | owner と static index で一 case の NF を参照 | その case の bound |
| FamilyGetDynamic | selector identity と stored reachable cases の NF | stored cases の最大。全 index 列挙禁止 |
| Identity/rotation/permutation view | exact factor に owner-resolved view spec を保持。証明済み合成だけ fold | `Bounded(1)` または入力 bound 保持 |
| MatrixScale/other coefficient-preserving view | exact identity と ordered child を保持 | operation 固有 scalar rule、または bound 保持 |

Transpose、Slice、Concat、Tensor、Select などの structural operation を `Existing` のような不透明 leaf にして後から別 extractor へ渡してはならない。上表の canonical constructor が最終表現を直接作る。

## 8. source と bound の規則

### 8.1 BoundClass

- `ExactZero`: 全係数が厳密に 0。
- `Bounded(B)`: 全多項式係数 `c` が `|c| <= B`。
- `Large`: protocol が小さい有限係数 bound を使用しないと明示した値。
- missing/unspecified は `Large` ではなく contract error。

### 8.2 ProtocolInput と GraphWire

ProtocolInput と通常 GraphWire を種類だけで Large にしない。次の authoritative source をちょうど一つ解決する。

1. exact upstream protocol/artifact binding の producer output bound。
2. explicit `MatrixBounded(B)`。
3. `MatrixExact` の canonical contract `0 <= c < U`。`0 < U <= Q` を検証して `Bounded(U-1)`。
4. explicit `Large`。

何もなければ `MissingInputBoundContract`。名前、shape、runtime candidate から推測しない。`is_constant_polynomial` は bound 値とは別の乗算 metadata である。

### 8.3 Bool と Int

Bool/Int は matrix BoundClass ではなく integer domain を持つ。Bool は `[0,1]`。`[min,max]` を定数多項式へ lift した上界は `max(abs(min), abs(max))`。domain 欠落はエラー。

`ExtractCoefficient` は入力 matrix に authoritative `canonical_coefficient_exclusive_upper_bound = U` があれば selector-only `[0,U-1]` を使う。なければ完全 modulus range へ fallback する。runtime 観測値を使わない。selector-only provenance は Divide/Remainder 後も保持し、range check、IntCompare、BitExtract、FamilyGetDynamic、Select だけが消費できる。matrix scale、dimension、sampler cutoff、noise arithmetic では拒否する。

### 8.4 sampler と decomposition

- Gaussian + explicit nonnegative hard cutoff `C` -> `Bounded(C)`。sigma から推測しない。
- UniformInterval `[min,max]` -> `Bounded(max(abs(min),abs(max)))`。
- UniformResidue -> `Large`。
- Preimage sampler -> relation の使用有無と独立に explicit cutoff の `Bounded`。
- decomposition digit、base > 1:
  - regular digit -> `Bounded(max(floor(base/2),1))`
  - small digit -> `Bounded(base-1)`
- regular Gadget matrix -> `Large`。
- small Gadget matrix -> `Bounded(base-1)`。

Gadget matrix と decomposition digit の helper を共有しない。

### 8.5 matrix constants

- Zero -> `ExactZero`
- Identity、UnitRow、UnitColumn、valid rotation/permutation -> `Bounded(1)`
- explicit polynomial -> 最大絶対係数。全 0 なら `ExactZero`
- PowerOfBase -> `Bounded(abs(base)^exponent)`
- invalid base、shape、index -> error。Large への fallback 禁止

### 8.6 PackPolynomialCoefficients

係数 major、little-endian Bool bits から

```text
c_j = sum_k 2^k bit[j,k]
B_j = sum_k 2^k max(bit[j,k])
B(output) = max_j min(B_j, q-1)
```

を計算する。Bool family、1x1 output、正の bit width、bit count `ring_dimension * width` を検証する。常に Large にしない。既知 zero bit は上界を狭めてよい。

### 8.7 演算 bound

- Add/Sub: zero は単位元、bounded 同士は加算、それ以外は Large。相関 Large の相殺は同じ exact monomial identity の場合だけ。
- matrix product: zero を最優先。Large factor があれば Large。`Bounded(A)*Bounded(B)` は `Bounded(K*R*A*B)`。`K` は非 zero の可能性がある inner summand 数、1x1 scalar なら 1。`R=1` はどちらかが定数多項式、それ以外は ring dimension。
- integer scalar scale: `S=max(abs(min),abs(max))`、`Bounded(B)->Bounded(SB)`。非 zero scalar と Large は Large。
- Tensor: zero/Large 優先と polynomial factor `R` は同じ。matrix inner sum はない。
- Concat と Switch/Select: reachable alternatives の最大。和ではない。
- CRT: `sum_i abs(c_i) B_i`。`c_i=0` は Large input でも zero 寄与。
- transpose、slice、係数保存 view は class を保存。
- HashPlain は authoritative hard output range がなければ Large。
- SequentialState は recurrence 内で previous carried bound を継承。外で未解決なら error。SequentialRecurrence が initial/transition を評価する。

### 8.8 first-Large witness

first-Large witness は診断のためだけに、その場で計算する。走査順は `PolynomialNF` の ordered monomial key、factor index、Switch case index の辞書順とする。最初に `Large` を返した authoritative source identity と、root からそこまでの operation path だけを報告する。候補探索順、rayon completion 順、hash iteration 順を使わない。analysis field、e-class、永続 cache、protocol artifact へ保存しない。

## 9. fail-closed と deferred cases

次は受理せず、型付きエラーにする。

- bound contract、integer domain、producer binding の欠落。
- matrix type、layout、owner、runtime coordinate、trapdoor の不一致。
- relation registration が複数候補になり一意でない。
- ordered factor を交換しないと適用できない relation。
- bounded summary を後で exact/Large factor と乗算する式。
- 異なる selector の Switch を Cartesian に組み合わせる必要がある式。
- stored cases で表現できない family。
- unsupported operation、invalid shape/base/index、arithmetic overflow。
- 最終 residual に exact/Large monomial が残る場合。

CyclicGraphDependency の完全な悪意入力対策は deferred でよい。ただし honest protocol のヒューマンエラーとして直接 cycle を検出できる DFS の visiting 状態は保持する。根拠のない所有要素 budget は設けない。

slot-transfer Tall encoding は別仕様で未実装なら、対応 gate に明示的 `Unimplemented` を返す。Tall integration fixture が cyclic rotation だけで reconstruct できる構成では、その gate を生成しない。

## 10. 決定性、停止性、confluence

### 10.1 決定性

- map/set iteration は ordered container または明示 sort。
- atom、monomial、case の比較 key を仕様化し、node insertion order を使わない。
- applicable relation は 6.2 の完全 match key 全体について一意でなければ error。同じ source identity を共有していても、ordered coordinates、public/target/layout/trapdoor/selector provenance などが異なる登録は許可する。
- 同じ入力 bundle と parameter environment から byte-for-byte 同じ診断順を得る。

### 10.2 停止性

- 式 DAG は bottom-up に各 node 一回 memoize。
- Add/Multiply flatten は DAG の edge を有限回訪問。
- relation は一方向で、各再帰が producer-DAG rank multiset を辞書式に厳密減少させる。
- Switch scope minimization は共通 prefix/suffix を外へ出すだけで、再び内へ戻す規則を持たない。
- bounded aggregation は項を減らすだけで、再展開しない。

### 10.3 confluence

任意順 rewrite の confluence に依存しない。固定 phase 順、ordered map、左からの relation 適用により canonical result を直接構築する。実装テストでは、Add/Multiply の異なる結合木、入力 node insertion 順、thread scheduling を変えて同じ `PolynomialNF` になることを確認する。

## 11. 計算量、メモリ、並列性

`N` を reachable DAG node 数、`E` を edge 数、`T` を実際に生成される exact monomial 数、`F` を stored family case 数、`L` を reachable stored cases の `PolynomialNF` 総サイズ、`G` を relation target 正規化で実際に生成された NF 総サイズとする。

- source/bound 解決: `O(N + E)`。
- Add flatten と canonical insertion: `O(T log T)`。
- Multiply: 出力 exact monomial 数に比例する。全面展開そのものの `T` は増え得るが、bounded-only 項は各演算直後に一項へ集約する。
- relation lookup: 各 factor 境界について full match key の ordered-map lookup `O(log R)`。target 処理を含む総時間は lookup 数と生成サイズ `G` に比例し、raw logical family size には比例しない。
- Switch/family: case 数だけでなく、実際に読む stored case NF の総サイズ `L` に対して `O(L)`。selector Cartesian product 禁止。
- family maximum: reachable stored cases を一回走査する。利用箇所ごとに NF を再構築せず、既存 DAG memo の結果を再利用する。

新しい永続 cache database は作らない。一 simulation job 内の node memo、interned atom identity、stored family cases を再利用する。memo の寿命は job 内に限定する。

独立 node/case の bound 計算は rayon で並列化してよい。ただし、同じ ordered result を得るため、各 worker の局所結果を deterministic key で merge する。小さい loop を無条件に parallelize して overhead を増やさず、既存の configurable batch size を使う。peak memory は worker 数と active batch に比例させ、全 protocol loop count に比例させない。

## 12. 段階的 migration

### Stage 0: evidence 固定

- Tall、Diamond WE、noiseless fixture の exact source/bound/relation chain を ledger 化。
- 現行 checker の最初の Large witness、時間、peak RAM、relation count を保存。
- `input_max_plaintext_norm_ranges` など既存 field の意味を producer code と runtime 使用箇所から確認する。

### Stage 1: egg 非依存 identity/bound resolver

- 既存 owner-aware identity と bound resolver を pure API として切り出す。
- egg と新 API の differential test を追加。

### Stage 2: PolynomialNF builder

- zero、Add/Negate、ordered Multiply、central scalar、bounded aggregation を実装。
- relation と Switch はまだ fail-closed。

### Stage 3: exact relation

- `B*K=P` の一方向規則、prefix/suffix、複数境界、trapdoor input、runtime coordinates を実装。

### Stage 4: Switch と family

- scope minimization、same-selector case-wise 処理、stored family cases を実装。

### Stage 5: checker 接続

- 最終 bound と threshold 判定を新 pipeline へ接続。
- egg checker と並走する differential mode はテスト専用に限定。

### Stage 6: egg 削除

- egg language、runner、rewrite、extractor、preference、final-leaf filter を削除。
- Cargo dependency と dead diagnostics を削除。
- 旧 Lean checker と専用 source-hash check を削除。

各 Stage は独立 commit とし、次 Stage へ進む前に focused tests と reviewer acceptance を得る。

## 13. acceptance gates

### 13.1 focused tests

- `0*Large=0`、非 zero `*Large=Large`。
- Add/Negate の全結合順で同じ相殺。
- ordered product flatten。factor swap は相殺しない。
- bounded-only aggregation を各演算後に実行し、上界が未集約計算以上で一致。
- `prefix*B*K*suffix -> prefix*P*suffix`。
- `E_B*K`、`K*B`、wrong public、wrong coordinate、wrong trapdoor は保持。
- 一 monomial の複数 relation 境界。
- producer-DAG rank multiset が減る nested relation は成功し、同 rank cycle と rank 増加は拒否。
- full match key の該当 0 件は不適用、同 target 重複は deduplicate、異 target 複数は ambiguous rejection。
- trapdoor が protocol input の relation。
- Switch の共通 prefix/suffix、共通 Add 項、`Switch(cases*G)->Switch(cases)*G`。
- same-selector success、different-selector fail-closed。
- family static/dynamic boundary、`U<=count` success、`U>count` rejection。
- source contract の全経路と missing contract。
- MatrixExact `U-1`、constant metadata の有無。
- regular Gadget Large と decomposition digit bounded。
- Pack general/known-zero bits。
- CRT zero coefficient。
- Transpose の Add/Negate/積への分配、積の factor 反転、二重 Transpose 除去。
- Concat は整列した pointwise Add だけを分配し、axis、arity、shape、対応位置のいずれかが異なる場合は canonical structural factor を保持。
- operation table の全行について exact identity、shape、bound の focused test。
- first-Large witness が insertion 順、hash seed、rayon thread 数に依存しない。

### 13.2 differential tests

- egg 版が正しく完了する既存 fixture では同じ最終 bound。
- egg 版が raw lhs を再選択する fixture では、新版が canonical target を一度だけ生成。
- node insertion 順、Add/Multiply association、rayon thread count を変えて同一 NF。
- 新版が受理し egg 版が unsupported になる場合は、仕様上の規則と witness を記録する。

### 13.3 noiseless runtime gate

small nested-RNS parameter を使い、すべての加法 noise を 0 にする。Tall output encoding を、checker と独立に計算した期待 plaintext product と runtime で厳密比較し、残差が完全に 0 であることを要求する。単なる checker success では代用しない。

### 13.4 Tall gate

- 既存 Tall integration test を使い、環境変数で benchmark estimation の到達点を制御する。新しい統合テスト target を増やさない。
- 合意済み parameter を維持し、nested-RNS scale は `32`。
- noise simulation が最後まで成功することを必須とする。
- その後の benchmark estimate が out of VRAM で終了することは許容する。
- progress log は phase、processed/total nodes、exact term count、bounded aggregation count、relation remaining/applied、Switch cases processed、elapsed time を bounded cadence で出す。
- node 番号や Tall 固有名は一時診断以外の分岐に使わない。

### 13.5 Diamond WE gate

Diamond WE/iO も Rust checker を使う。Diamond WE の既存 integration path で exact signal が残らず、finite bound と strict threshold 判定まで完了する。iO crate が明示的に disabled の期間は、disabled 状態を変更せず compile gate だけ記録する。

### 13.6 noisy Runpod gate

GPU が必要な noiseless/noisy integration は Runpod で実行する。noisy Tall の固定条件は `MXX_TALL_NESTED_RNS_SECURITY_BITS=0`、`MXX_TALL_NESTED_RNS_MIN_LOG_RING_DIMENSION=3`、`MXX_TALL_NESTED_RNS_MAX_LOG_RING_DIMENSION=3`、`MXX_TALL_NESTED_RNS_ERROR_SIGMA>0` とし、最大 4 台の RTX 5090 を使用してよい。実行 commit、全環境変数、GPU 台数、開始終了時刻、peak VRAM/RAM、完全ログを保存する。Tall と Diamond を同時実行しない。

noise simulation の完遂と strict threshold の成立は実機試験へ進むための前提条件であり、それ自体を最終 acceptance と数えてはならない。その simulation が選んだ parameter を固定し、`error_sigma > 0` の実際の Tall runtime roundtrip を、同一 commit・同一 parameter で 3 回連続実行する。3 回すべてで decoded/runtime output が期待値と一致した場合だけ acceptance とする。

実機 roundtrip が失敗した場合は、まず仕様で許された `crt_depth` の escalation を行い、simulation と strict threshold を再度通してから実機試験をやり直す。それでも失敗する、または runtime noise が checker bound を超える場合は、parameter をさらに緩める前に underestimation audit を行い、欠落項、誤った constant-polynomial factor、inner-sum 数、ring-dimension factor、CRT coefficient を照合する。benchmark OOM は noise simulation 完了後だけ許容する。

### 13.7 最終 threshold

finite residual bound `B` に対してのみ、次を厳密に判定する。

```text
2 * plaintext_modulus * B < ciphertext_modulus
```

等号は拒否する。Large、contract error、未相殺 exact term がある場合は threshold を計算する前に拒否する。

## 14. 複雑性・単純性監査

新しい概念またはデータ構造を 3 個導入するたびに監査する。監査表には次を必ず記載する。

1. 解決する一般的な問題。
2. 既存構造を再利用できない理由。
3. correctness 上の利点。
4. time complexity と memory complexity の差分。
5. 削除できる旧コード。
6. Tall/Diamond の具体的な必要性。
7. より単純な代替案を棄却した根拠。

次の場合は変更を棄却する。

- protocol 名、node 番号、fixture 値に依存する。
- 新 cache/database が既存 memo や symbol table と重複する。
- 同じ relation/bound 判定を二箇所に実装する。
- correctness を増やさず traversal 回数または peak memory を増やす。
- fail-closed の理由を型付きで説明できない。
- 複数候補を順に試すことでしか動かず、canonical rule を定義できない。

過去の最適化と結果は `docs/correctness/exact-signal-large-debugging-history.md` に追記し、仕様変更前に同じ案を既に試していないか確認する。新仕様の実装が acceptance gates を通過した後、移行用の egg compatibility code と一時診断を残さない。
