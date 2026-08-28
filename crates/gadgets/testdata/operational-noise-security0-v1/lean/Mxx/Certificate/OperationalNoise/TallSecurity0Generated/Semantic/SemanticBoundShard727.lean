import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard726

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106062
def owner : Owner := ⟨.program ⟨214⟩, ⟨21032⟩⟩
def transferEvent : Nat := 106062
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106060 .coefficient) (.predecessor 1 106061 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106060 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106061 .coefficient)
      LeftBound106058.bound (LeftBound106058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106058.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106058.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106058.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106062

namespace LeftBound106063
def owner : Owner := ⟨.program ⟨214⟩, ⟨21032⟩⟩
def transferEvent : Nat := 106063
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21029⟩⟩]⟩ [⟨.result 106055 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106055 .coefficient)
      LeftAuthority106054.bound (LeftAuthority106054.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21029⟩⟩) (rawTerms := some (Proof.Events414.exact106055RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106054.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106054.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106054.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106063

namespace LeftBound106064
def owner : Owner := ⟨.program ⟨214⟩, ⟨21032⟩⟩
def transferEvent : Nat := 106064
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 106063) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106063)
      LeftBound106063.bound (LeftBound106063.actual selector witness) := by
  exact .transfer (LeftBound106063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound106063.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound106063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound106063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106064

namespace LeftBound106135
def owner : Owner := ⟨.program ⟨214⟩, ⟨15693⟩⟩
def transferEvent : Nat := 106135
def frameStart : Nat := 106108
def rule : BoundRule := .identity (.predecessor 0 106134 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106134 .coefficient)
      LeftAuthority106132.bound (LeftAuthority106132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106132.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106132.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority106132.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106135

namespace LeftBound106152
def owner : Owner := ⟨.program ⟨214⟩, ⟨15769⟩⟩
def transferEvent : Nat := 106152
def frameStart : Nat := 106108
def rule : BoundRule := .sum [.predecessor 0 106150 .coefficient, .predecessor 1 106151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106150 .coefficient)
      LeftBound106135.bound (LeftBound106135.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106151 .coefficient)
      LeftAuthority106148.bound (LeftAuthority106148.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106135.bound, LeftAuthority106148.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106135.bound, LeftAuthority106148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106135.actual selector witness, LeftAuthority106148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106152

namespace LeftBound106155
def owner : Owner := ⟨.program ⟨214⟩, ⟨15770⟩⟩
def transferEvent : Nat := 106155
def frameStart : Nat := 106108
def rule : BoundRule := .identity (.predecessor 0 106154 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106154 .coefficient)
      LeftBound106152.bound (LeftBound106152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106152.derived selector witness)

def rawBound : CoeffClass := LeftBound106152.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound106152.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106155

namespace LeftBound106161
def owner : Owner := ⟨.program ⟨214⟩, ⟨15771⟩⟩
def transferEvent : Nat := 106161
def frameStart : Nat := 106108
def rule : BoundRule := .product (.predecessor 0 106159 .coefficient) (.predecessor 1 106160 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106159 .coefficient)
      LeftAuthority106157.bound (LeftAuthority106157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106160 .coefficient)
      LeftBound106155.bound (LeftBound106155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority106157.bound LeftBound106155.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106157.bound, LeftBound106155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority106157.actual selector witness) * (LeftBound106155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106161

namespace LeftBound106169
def owner : Owner := ⟨.program ⟨214⟩, ⟨15772⟩⟩
def transferEvent : Nat := 106169
def frameStart : Nat := 106108
def rule : BoundRule := .sum [.predecessor 0 106167 .coefficient, .predecessor 1 106168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106167 .coefficient)
      LeftAuthority106165.bound (LeftAuthority106165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106168 .coefficient)
      LeftBound106161.bound (LeftBound106161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106165.bound, LeftBound106161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106165.bound, LeftBound106161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106165.actual selector witness, LeftBound106161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106169

namespace LeftBound106173
def owner : Owner := ⟨.program ⟨214⟩, ⟨27391⟩⟩
def transferEvent : Nat := 106173
def frameStart : Nat := 106108
def rule : BoundRule := .product (.predecessor 0 106171 .coefficient) (.predecessor 1 106172 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106171 .coefficient)
      LeftBound106169.bound (LeftBound106169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106172 .coefficient)
      LeftAuthority106146.bound (LeftAuthority106146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106169.bound LeftAuthority106146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106169.bound, LeftAuthority106146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106169.actual selector witness) * (LeftAuthority106146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106173

namespace LeftBound106184
def owner : Owner := ⟨.program ⟨214⟩, ⟨17430⟩⟩
def transferEvent : Nat := 106184
def frameStart : Nat := 106108
def rule : BoundRule := .product (.predecessor 0 106182 .coefficient) (.predecessor 1 106183 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106182 .coefficient)
      LeftAuthority106157.bound (LeftAuthority106157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106183 .coefficient)
      LeftAuthority106180.bound (LeftAuthority106180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106180.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106157.bound LeftAuthority106180.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106157.bound, LeftAuthority106180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority106157.actual selector witness) * (LeftAuthority106180.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106184

namespace LeftBound106192
def owner : Owner := ⟨.program ⟨214⟩, ⟨17431⟩⟩
def transferEvent : Nat := 106192
def frameStart : Nat := 106108
def rule : BoundRule := .sum [.predecessor 0 106190 .coefficient, .predecessor 1 106191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106190 .coefficient)
      LeftAuthority106188.bound (LeftAuthority106188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106191 .coefficient)
      LeftBound106184.bound (LeftBound106184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106188.bound, LeftBound106184.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106188.bound, LeftBound106184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106188.actual selector witness, LeftBound106184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106192

namespace LeftBound106196
def owner : Owner := ⟨.program ⟨214⟩, ⟨27396⟩⟩
def transferEvent : Nat := 106196
def frameStart : Nat := 106108
def rule : BoundRule := .sum [.predecessor 0 106194 .coefficient, .predecessor 1 106195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106194 .coefficient)
      LeftBound106192.bound (LeftBound106192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106195 .coefficient)
      LeftBound106173.bound (LeftBound106173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106192.bound, LeftBound106173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106192.bound, LeftBound106173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106192.actual selector witness, LeftBound106173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106196

namespace LeftBound106209
def owner : Owner := ⟨.program ⟨214⟩, ⟨27393⟩⟩
def transferEvent : Nat := 106209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106207 .coefficient, .predecessor 1 106208 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106207 .coefficient)
      LeftBound106062.bound (LeftBound106062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106208 .coefficient)
      LeftBound106045.bound (LeftBound106045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106045.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106062.bound, LeftBound106045.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106062.bound, LeftBound106045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106062.actual selector witness, LeftBound106045.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106209

namespace LeftBound106212
def owner : Owner := ⟨.program ⟨214⟩, ⟨27393⟩⟩
def transferEvent : Nat := 106212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106206 .summary, .result 106052 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106206 .summary)
      LeftBound106064.bound (LeftBound106064.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21032⟩⟩) (rawTerms := some (Proof.Events414.exact106206RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106052 .summary)
      LeftBound106047.bound (LeftBound106047.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27392⟩⟩) (rawTerms := some (Proof.Events414.exact106052RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106064.bound, LeftBound106047.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106064.bound, LeftBound106047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106064.actual selector witness, LeftBound106047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106212

namespace LeftBound106216
def owner : Owner := ⟨.program ⟨214⟩, ⟨27394⟩⟩
def transferEvent : Nat := 106216
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106214 .coefficient) (.predecessor 1 106215 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106214 .coefficient)
      LeftBound106209.bound (LeftBound106209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106215 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106209.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106209.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106209.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106216

namespace LeftBound106217
def owner : Owner := ⟨.program ⟨214⟩, ⟨27394⟩⟩
def transferEvent : Nat := 106217
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩ [⟨.result 5755 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5755 .coefficient)
      LeftAuthority5754.bound (LeftAuthority5754.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6647⟩⟩) (rawTerms := some (Proof.Events022.exact5755RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5754.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5754.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5754.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106217

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
