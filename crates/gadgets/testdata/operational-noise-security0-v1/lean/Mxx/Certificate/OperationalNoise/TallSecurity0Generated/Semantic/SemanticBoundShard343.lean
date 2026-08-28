import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard342

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51199
def owner : Owner := ⟨.program ⟨214⟩, ⟨7263⟩⟩
def transferEvent : Nat := 51199
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51197 .coefficient) (.predecessor 1 51198 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51197 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51198 .coefficient)
      LeftBound7013.bound (LeftBound7013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound7013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound7013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound7013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51199

namespace LeftBound51204
def owner : Owner := ⟨.program ⟨214⟩, ⟨10247⟩⟩
def transferEvent : Nat := 51204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51202 .coefficient, .predecessor 1 51203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51202 .coefficient)
      LeftBound51199.bound (LeftBound51199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51203 .coefficient)
      LeftBound51194.bound (LeftBound51194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51199.bound, LeftBound51194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51199.bound, LeftBound51194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51199.actual selector witness, LeftBound51194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51204

namespace LeftBound51208
def owner : Owner := ⟨.program ⟨214⟩, ⟨10248⟩⟩
def transferEvent : Nat := 51208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51206 .coefficient, .predecessor 1 51207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51206 .coefficient)
      LeftBound51204.bound (LeftBound51204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51207 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51204.bound, LeftBound7005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51204.bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51204.actual selector witness, LeftBound7005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51208

namespace LeftBound51209
def owner : Owner := ⟨.program ⟨214⟩, ⟨10248⟩⟩
def transferEvent : Nat := 51209
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩ [⟨.result 7006 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7006 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨83⟩⟩) (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7005.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51209

namespace LeftBound51214
def owner : Owner := ⟨.program ⟨214⟩, ⟨10249⟩⟩
def transferEvent : Nat := 51214
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51212 .coefficient) (.predecessor 1 51213 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51212 .coefficient)
      LeftBound51208.bound (LeftBound51208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51213 .coefficient)
      LeftBound7002.bound (LeftBound7002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51208.bound LeftBound7002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51208.bound, LeftBound7002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51208.actual selector witness) * (LeftBound7002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51214

namespace LeftBound51215
def owner : Owner := ⟨.program ⟨214⟩, ⟨10249⟩⟩
def transferEvent : Nat := 51215
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩ [⟨.result 6999 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6999 .coefficient)
      LeftAuthority6998.bound (LeftAuthority6998.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7879⟩⟩) (rawTerms := some (Proof.Events027.exact6999RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6998.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6998.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6998.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51215

namespace LeftBound51216
def owner : Owner := ⟨.program ⟨214⟩, ⟨10249⟩⟩
def transferEvent : Nat := 51216
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51211 .summary) (.transfer 51215) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51211 .summary)
      LeftBound51209.bound (LeftBound51209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10248⟩⟩) (rawTerms := some (Proof.Events200.exact51211RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51215)
      LeftBound51215.bound (LeftBound51215.actual selector witness) := by
  exact .transfer (LeftBound51215.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51209.bound LeftBound51215.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51209.bound, LeftBound51215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51209.actual selector witness) * (LeftBound51215.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51216

namespace LeftBound51224
def owner : Owner := ⟨.program ⟨214⟩, ⟨13169⟩⟩
def transferEvent : Nat := 51224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51222 .coefficient, .predecessor 1 51223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51222 .coefficient)
      LeftBound51214.bound (LeftBound51214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51223 .coefficient)
      LeftBound51186.bound (LeftBound51186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51214.bound, LeftBound51186.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51214.bound, LeftBound51186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51214.actual selector witness, LeftBound51186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51224

namespace LeftBound51226
def owner : Owner := ⟨.program ⟨214⟩, ⟨13169⟩⟩
def transferEvent : Nat := 51226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51221 .summary, .result 51191 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51221 .summary)
      LeftBound51216.bound (LeftBound51216.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10249⟩⟩) (rawTerms := some (Proof.Events200.exact51221RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51191 .summary)
      LeftBound51188.bound (LeftBound51188.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13168⟩⟩) (rawTerms := some (Proof.Events199.exact51191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51216.bound, LeftBound51188.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51216.bound, LeftBound51188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51216.actual selector witness, LeftBound51188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51226

namespace LeftBound51230
def owner : Owner := ⟨.program ⟨214⟩, ⟨25687⟩⟩
def transferEvent : Nat := 51230
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51228 .coefficient) (.predecessor 1 51229 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51228 .coefficient)
      LeftBound51224.bound (LeftBound51224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51229 .coefficient)
      LeftAuthority51162.bound (LeftAuthority51162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51162.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51224.bound LeftAuthority51162.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51224.bound, LeftAuthority51162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51224.actual selector witness) * (LeftAuthority51162.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51230

namespace LeftBound51231
def owner : Owner := ⟨.program ⟨214⟩, ⟨25687⟩⟩
def transferEvent : Nat := 51231
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25686⟩⟩]⟩ [⟨.result 51163 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51163 .coefficient)
      LeftAuthority51162.bound (LeftAuthority51162.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25686⟩⟩) (rawTerms := some (Proof.Events199.exact51163RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51162.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51162.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51162.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51231

namespace LeftBound51232
def owner : Owner := ⟨.program ⟨214⟩, ⟨25687⟩⟩
def transferEvent : Nat := 51232
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51227 .summary) (.transfer 51231) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51227 .summary)
      LeftBound51226.bound (LeftBound51226.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13169⟩⟩) (rawTerms := some (Proof.Events200.exact51227RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51231)
      LeftBound51231.bound (LeftBound51231.actual selector witness) := by
  exact .transfer (LeftBound51231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51226.bound LeftBound51231.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51226.bound, LeftBound51231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51226.actual selector witness) * (LeftBound51231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51232

namespace LeftBound51243
def owner : Owner := ⟨.program ⟨214⟩, ⟨20182⟩⟩
def transferEvent : Nat := 51243
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 51241 .coefficient) (.value (.predecessor 1 51242 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51241 .coefficient)
      LeftAuthority51239.bound (LeftAuthority51239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51242 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51239.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51239.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51239.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51243

namespace LeftBound51247
def owner : Owner := ⟨.program ⟨214⟩, ⟨20183⟩⟩
def transferEvent : Nat := 51247
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51245 .coefficient) (.predecessor 1 51246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51245 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51246 .coefficient)
      LeftBound51243.bound (LeftBound51243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51243.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound51243.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound51243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound51243.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51247

namespace LeftBound51248
def owner : Owner := ⟨.program ⟨214⟩, ⟨20183⟩⟩
def transferEvent : Nat := 51248
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20180⟩⟩]⟩ [⟨.result 51240 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51240 .coefficient)
      LeftAuthority51239.bound (LeftAuthority51239.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20180⟩⟩) (rawTerms := some (Proof.Events200.exact51240RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51239.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51239.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51239.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51248

namespace LeftBound51249
def owner : Owner := ⟨.program ⟨214⟩, ⟨20183⟩⟩
def transferEvent : Nat := 51249
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 51248) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51248)
      LeftBound51248.bound (LeftBound51248.actual selector witness) := by
  exact .transfer (LeftBound51248.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound51248.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound51248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound51248.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51249

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
