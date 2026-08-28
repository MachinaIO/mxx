import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard058

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10237
def owner : Owner := ⟨.program ⟨214⟩, ⟨25166⟩⟩
def transferEvent : Nat := 10237
def frameStart : Nat := 10123
def rule : BoundRule := .sum [.predecessor 0 10235 .coefficient, .predecessor 1 10236 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10235 .coefficient)
      LeftBound10233.bound (LeftBound10233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10236 .coefficient)
      LeftBound10214.bound (LeftBound10214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10233.bound, LeftBound10214.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10233.bound, LeftBound10214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10233.actual selector witness, LeftBound10214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10237

namespace LeftBound10250
def owner : Owner := ⟨.program ⟨214⟩, ⟨25164⟩⟩
def transferEvent : Nat := 10250
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10248 .coefficient, .predecessor 1 10249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10248 .coefficient)
      LeftBound10071.bound (LeftBound10071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10249 .coefficient)
      LeftBound10054.bound (LeftBound10054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10071.bound, LeftBound10054.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10071.bound, LeftBound10054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10071.actual selector witness, LeftBound10054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10250

namespace LeftBound10253
def owner : Owner := ⟨.program ⟨214⟩, ⟨25164⟩⟩
def transferEvent : Nat := 10253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 10247 .summary, .result 10061 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10247 .summary)
      LeftBound10073.bound (LeftBound10073.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19763⟩⟩) (rawTerms := some (Proof.Events040.exact10247RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10061 .summary)
      LeftBound10056.bound (LeftBound10056.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25163⟩⟩) (rawTerms := some (Proof.Events039.exact10061RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10073.bound, LeftBound10056.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10073.bound, LeftBound10056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10073.actual selector witness, LeftBound10056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10253

namespace LeftBound10257
def owner : Owner := ⟨.program ⟨214⟩, ⟨28571⟩⟩
def transferEvent : Nat := 10257
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10255 .coefficient) (.predecessor 1 10256 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10255 .coefficient)
      LeftBound10250.bound (LeftBound10250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10256 .coefficient)
      LeftAuthority9957.bound (LeftAuthority9957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9957.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10250.bound LeftAuthority9957.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10250.bound, LeftAuthority9957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10250.actual selector witness) * (LeftAuthority9957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10257

namespace LeftBound10258
def owner : Owner := ⟨.program ⟨214⟩, ⟨28571⟩⟩
def transferEvent : Nat := 10258
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28569⟩⟩]⟩ [⟨.result 9958 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9958 .coefficient)
      LeftAuthority9957.bound (LeftAuthority9957.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28569⟩⟩) (rawTerms := some (Proof.Events038.exact9958RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9957.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9957.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9957.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9957.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10258

namespace LeftBound10259
def owner : Owner := ⟨.program ⟨214⟩, ⟨28571⟩⟩
def transferEvent : Nat := 10259
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10254 .summary) (.transfer 10258) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10254 .summary)
      LeftBound10253.bound (LeftBound10253.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25164⟩⟩) (rawTerms := some (Proof.Events040.exact10254RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10258)
      LeftBound10258.bound (LeftBound10258.actual selector witness) := by
  exact .transfer (LeftBound10258.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10253.bound LeftBound10258.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10253.bound, LeftBound10258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10253.actual selector witness) * (LeftBound10258.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10259

namespace LeftBound10270
def owner : Owner := ⟨.program ⟨214⟩, ⟨21850⟩⟩
def transferEvent : Nat := 10270
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 10268 .coefficient) (.value (.predecessor 1 10269 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10268 .coefficient)
      LeftAuthority10266.bound (LeftAuthority10266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10269 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10266.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10266.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10266.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10270

namespace LeftBound10274
def owner : Owner := ⟨.program ⟨214⟩, ⟨21851⟩⟩
def transferEvent : Nat := 10274
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10272 .coefficient) (.predecessor 1 10273 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10272 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10273 .coefficient)
      LeftBound10270.bound (LeftBound10270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10270.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound10270.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound10270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound10270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10274

namespace LeftBound10275
def owner : Owner := ⟨.program ⟨214⟩, ⟨21851⟩⟩
def transferEvent : Nat := 10275
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21848⟩⟩]⟩ [⟨.result 10267 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10267 .coefficient)
      LeftAuthority10266.bound (LeftAuthority10266.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21848⟩⟩) (rawTerms := some (Proof.Events040.exact10267RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10266.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10266.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10266.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10275

namespace LeftBound10276
def owner : Owner := ⟨.program ⟨214⟩, ⟨21851⟩⟩
def transferEvent : Nat := 10276
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 10275) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10275)
      LeftBound10275.bound (LeftBound10275.actual selector witness) := by
  exact .transfer (LeftBound10275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound10275.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound10275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound10275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10276

namespace LeftBound10371
def owner : Owner := ⟨.program ⟨214⟩, ⟨16279⟩⟩
def transferEvent : Nat := 10371
def frameStart : Nat := 10332
def rule : BoundRule := .identity (.predecessor 0 10370 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10370 .coefficient)
      LeftAuthority10368.bound (LeftAuthority10368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10368.derived selector witness)

def rawBound : CoeffClass := LeftAuthority10368.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority10368.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10371

namespace LeftBound10388
def owner : Owner := ⟨.program ⟨214⟩, ⟨16353⟩⟩
def transferEvent : Nat := 10388
def frameStart : Nat := 10332
def rule : BoundRule := .sum [.predecessor 0 10386 .coefficient, .predecessor 1 10387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10386 .coefficient)
      LeftBound10371.bound (LeftBound10371.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10387 .coefficient)
      LeftAuthority10384.bound (LeftAuthority10384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10371.bound, LeftAuthority10384.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10371.bound, LeftAuthority10384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10371.actual selector witness, LeftAuthority10384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10388

namespace LeftBound10391
def owner : Owner := ⟨.program ⟨214⟩, ⟨16354⟩⟩
def transferEvent : Nat := 10391
def frameStart : Nat := 10332
def rule : BoundRule := .identity (.predecessor 0 10390 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10390 .coefficient)
      LeftBound10388.bound (LeftBound10388.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10388.derived selector witness)

def rawBound : CoeffClass := LeftBound10388.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10388.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10391

namespace LeftBound10397
def owner : Owner := ⟨.program ⟨214⟩, ⟨16355⟩⟩
def transferEvent : Nat := 10397
def frameStart : Nat := 10332
def rule : BoundRule := .product (.predecessor 0 10395 .coefficient) (.predecessor 1 10396 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10395 .coefficient)
      LeftAuthority10393.bound (LeftAuthority10393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10393.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10396 .coefficient)
      LeftBound10391.bound (LeftBound10391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority10393.bound LeftBound10391.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10393.bound, LeftBound10391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority10393.actual selector witness) * (LeftBound10391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10397

namespace LeftBound10405
def owner : Owner := ⟨.program ⟨214⟩, ⟨16356⟩⟩
def transferEvent : Nat := 10405
def frameStart : Nat := 10332
def rule : BoundRule := .sum [.predecessor 0 10403 .coefficient, .predecessor 1 10404 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10403 .coefficient)
      LeftAuthority10401.bound (LeftAuthority10401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10404 .coefficient)
      LeftBound10397.bound (LeftBound10397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority10401.bound, LeftBound10397.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10401.bound, LeftBound10397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority10401.actual selector witness, LeftBound10397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10405

namespace LeftBound10409
def owner : Owner := ⟨.program ⟨214⟩, ⟨28570⟩⟩
def transferEvent : Nat := 10409
def frameStart : Nat := 10332
def rule : BoundRule := .product (.predecessor 0 10407 .coefficient) (.predecessor 1 10408 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10407 .coefficient)
      LeftBound10405.bound (LeftBound10405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10408 .coefficient)
      LeftAuthority10382.bound (LeftAuthority10382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10382.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10405.bound LeftAuthority10382.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10405.bound, LeftAuthority10382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10405.actual selector witness) * (LeftAuthority10382.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10409

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
