import Mxx.Certificate.OperationalNoise.ToyGenerated.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyGenerated

open ToyABI

private def o (row : Nat) : ToyOwner := ⟨.closed 12, row⟩
private def m (rows : List Nat) : ToyMonomial := ⟨[], rows.map o⟩
private def t (coefficient : Int) (monomial : ToyMonomial) : ToyTerm :=
  ⟨monomial, coefficient⟩

def events : List ToyEvent :=
  [ .invocationStart (o 12),
    .boundTransfer (o 3) (.authority .operator),
    .result (o 3) (.exact [t 1 (m [3])] .exactZero),
    .boundTransfer (o 1) (.authority .operator),
    .result (o 1) (.exact [t 1 (m [1])] .exactZero),
    .boundTransfer (o 6) (.authority .operator),
    .result (o 6) (.coefficient (.finite 1)),
    .boundTransfer (o 5) (.authority .operator),
    .result (o 5) (.coefficient .exactZero),
    .predecessor (o 7) 0 5 8,
    .boundTransfer (o 7) (.authority (.relationPreimageSource 2)),
    .result (o 7) (.exact [t 1 (m [7])] .exactZero),
    .predecessor (o 9) 0 7 11,
    .predecessor (o 9) 1 6 6,
    .boundTransfer (o 9) (.scale
        (.predecessor 0 .coefficient) (.value (.predecessor 1 .coefficient))),
    .result (o 9) (.exact [t 1 (m [7])] .exactZero),
    .boundTransfer (o 0) (.authority .operator),
    .result (o 0) (.exact [t 1 (m [0])] .exactZero),
    .predecessor (o 10) 0 0 17,
    .predecessor (o 10) 1 9 15,
    .boundTransfer (o 10) (.product (.predecessor 0 .coefficient) (.predecessor 1 .coefficient)
        ⟨false, false, none, none, none⟩),
    .coefficientMerge (⟨o 10, .operator (⟨17, 0⟩, ⟨15, 0⟩), m [0, 7], 1⟩),
    .invocationStart (o 8),
    .boundTransfer (o 5) (.authority .operator),
    .result (o 5) (.coefficient .exactZero),
    .predecessor (o 7) 0 5 24,
    .boundTransfer (o 7) (.authority (.relationPreimageSource 2)),
    .result (o 7) (.exact [t 1 (m [7])] .exactZero),
    .boundTransfer (o 0) (.authority .operator),
    .result (o 0) (.exact [t 1 (m [0])] .exactZero),
    .predecessor (o 8) 0 0 29,
    .predecessor (o 8) 1 7 27,
    .boundTransfer (o 8) (.product (.predecessor 0 .coefficient) (.predecessor 1 .coefficient)
        ⟨false, false, none, none, none⟩),
    .coefficientMerge (⟨o 8, .operator (⟨29, 0⟩, ⟨27, 0⟩), m [0, 7], 1⟩),
    .result (o 8) (.exact [t 1 (m [0, 7])] .exactZero),
    .preFoldPolynomial [t 1 (m [0, 7])] .exactZero none,
    .invocationEnd (o 8) (.exact [t 1 (m [0, 7])] .exactZero),
    .invocationStart (o 1),
    .boundTransfer (o 1) (.authority .operator),
    .result (o 1) (.exact [t 1 (m [1])] .exactZero),
    .preFoldPolynomial [t 1 (m [1])] .exactZero none,
    .invocationEnd (o 1) (.exact [t 1 (m [1])] .exactZero),
    .specializationComputed (o 5) ⟨0, 2, 4⟩ ⟨22, 42⟩,
    .appliedUniversal (o 10) (m [0, 7]) 1 0 2 42 (m [0, 7]) none 41,
    .coefficientMerge (⟨o 10, .relation 43 0, m [1], 1⟩),
    .result (o 10) (.exact [t 1 (m [1])] .exactZero),
    .predecessor (o 11) 0 10 45,
    .predecessor (o 11) 1 1 4,
    .boundTransfer (o 11) (.sum [.predecessor 0 .coefficient, .predecessor 1 .coefficient]),
    .coefficientMerge (⟨o 11, .operator (⟨45, 0⟩, ⟨4, 0⟩), m [1], -1⟩),
    .result (o 11) (.exact [] .exactZero),
    .predecessor (o 12) 0 11 50,
    .predecessor (o 12) 1 3 2,
    .boundTransfer (o 12) (.sum [.predecessor 0 .coefficient, .predecessor 1 .coefficient]),
    .boundTransfer (o 12) (.monomialProduct (m [3]) [⟨.result 2 .coefficient, false, none⟩]),
    .survivorFold 1 54,
    .result (o 12) (.exact [] (.finite 1)),
    .preFoldPolynomial [] (.finite 1) (some (.result 56 .summary)),
    .invocationEnd (o 12) (.exact [] (.finite 1)) ]

theorem proofValid : ToyValid source document rows events := by
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  intro index indexBound
  rfl

end Mxx.Certificate.OperationalNoise.ToyGenerated
