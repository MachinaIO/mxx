import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard260

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39159
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 39159
def frameStart : Nat := 39084
def rule : BoundRule := .scale (.predecessor 0 39157 .coefficient) (.value (.predecessor 1 39158 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39157 .coefficient)
      LeftAuthority39155.bound (LeftAuthority39155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39158 .coefficient)
      LeftAuthority39146.bound (LeftAuthority39146.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39146.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39155.bound LeftAuthority39146.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39155.bound, LeftAuthority39146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39155.actual selector witness) * (LeftAuthority39146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39159

namespace LeftBound39162
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 39162
def frameStart : Nat := 39084
def rule : BoundRule := .identity (.predecessor 0 39161 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39161 .coefficient)
      LeftAuthority39149.bound (LeftAuthority39149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39149.derived selector witness)

def rawBound : CoeffClass := LeftAuthority39149.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority39149.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39162

namespace LeftBound39166
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 39166
def frameStart : Nat := 39084
def rule : BoundRule := .product (.predecessor 0 39164 .coefficient) (.predecessor 1 39165 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39164 .coefficient)
      LeftBound39162.bound (LeftBound39162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39165 .coefficient)
      LeftBound39159.bound (LeftBound39159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39162.bound LeftBound39159.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39162.bound, LeftBound39159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39162.actual selector witness) * (LeftBound39159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39166

namespace LeftBound39171
def owner : Owner := ⟨.program ⟨214⟩, ⟨12064⟩⟩
def transferEvent : Nat := 39171
def frameStart : Nat := 39084
def rule : BoundRule := .sum [.predecessor 0 39169 .coefficient, .predecessor 1 39170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39169 .coefficient)
      LeftBound39166.bound (LeftBound39166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39170 .coefficient)
      LeftBound39143.bound (LeftBound39143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39143.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39166.bound, LeftBound39143.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39166.bound, LeftBound39143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39166.actual selector witness, LeftBound39143.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39171

namespace LeftBound39175
def owner : Owner := ⟨.program ⟨214⟩, ⟨25232⟩⟩
def transferEvent : Nat := 39175
def frameStart : Nat := 39084
def rule : BoundRule := .product (.predecessor 0 39173 .coefficient) (.predecessor 1 39174 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39173 .coefficient)
      LeftBound39171.bound (LeftBound39171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39174 .coefficient)
      LeftAuthority39128.bound (LeftAuthority39128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39171.bound LeftAuthority39128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39171.bound, LeftAuthority39128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39171.actual selector witness) * (LeftAuthority39128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39175

namespace LeftBound39186
def owner : Owner := ⟨.program ⟨214⟩, ⟨16391⟩⟩
def transferEvent : Nat := 39186
def frameStart : Nat := 39084
def rule : BoundRule := .product (.predecessor 0 39184 .coefficient) (.predecessor 1 39185 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39184 .coefficient)
      LeftAuthority39139.bound (LeftAuthority39139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39185 .coefficient)
      LeftAuthority39182.bound (LeftAuthority39182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39139.bound LeftAuthority39182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39139.bound, LeftAuthority39182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39139.actual selector witness) * (LeftAuthority39182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39186

namespace LeftBound39194
def owner : Owner := ⟨.program ⟨214⟩, ⟨16392⟩⟩
def transferEvent : Nat := 39194
def frameStart : Nat := 39084
def rule : BoundRule := .sum [.predecessor 0 39192 .coefficient, .predecessor 1 39193 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39192 .coefficient)
      LeftAuthority39190.bound (LeftAuthority39190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39193 .coefficient)
      LeftBound39186.bound (LeftBound39186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39190.bound, LeftBound39186.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39190.bound, LeftBound39186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39190.actual selector witness, LeftBound39186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39194

namespace LeftBound39198
def owner : Owner := ⟨.program ⟨214⟩, ⟨25233⟩⟩
def transferEvent : Nat := 39198
def frameStart : Nat := 39084
def rule : BoundRule := .sum [.predecessor 0 39196 .coefficient, .predecessor 1 39197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39196 .coefficient)
      LeftBound39194.bound (LeftBound39194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39197 .coefficient)
      LeftBound39175.bound (LeftBound39175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39194.bound, LeftBound39175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39194.bound, LeftBound39175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39194.actual selector witness, LeftBound39175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39198

namespace LeftBound39211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25231⟩⟩
def transferEvent : Nat := 39211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39209 .coefficient, .predecessor 1 39210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39209 .coefficient)
      LeftBound39032.bound (LeftBound39032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39210 .coefficient)
      LeftBound39015.bound (LeftBound39015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39032.bound, LeftBound39015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39032.bound, LeftBound39015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39032.actual selector witness, LeftBound39015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39211

namespace LeftBound39214
def owner : Owner := ⟨.program ⟨214⟩, ⟨25231⟩⟩
def transferEvent : Nat := 39214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39208 .summary, .result 39022 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39208 .summary)
      LeftBound39034.bound (LeftBound39034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19827⟩⟩) (rawTerms := some (Proof.Events153.exact39208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39022 .summary)
      LeftBound39017.bound (LeftBound39017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25230⟩⟩) (rawTerms := some (Proof.Events152.exact39022RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39034.bound, LeftBound39017.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39034.bound, LeftBound39017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39034.actual selector witness, LeftBound39017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39214

namespace LeftBound39218
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def transferEvent : Nat := 39218
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39216 .coefficient) (.predecessor 1 39217 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39216 .coefficient)
      LeftBound39211.bound (LeftBound39211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39217 .coefficient)
      LeftAuthority38937.bound (LeftAuthority38937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39211.bound LeftAuthority38937.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39211.bound, LeftAuthority38937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39211.actual selector witness) * (LeftAuthority38937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39218

namespace LeftBound39219
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def transferEvent : Nat := 39219
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28760⟩⟩]⟩ [⟨.result 38938 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38938 .coefficient)
      LeftAuthority38937.bound (LeftAuthority38937.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28760⟩⟩) (rawTerms := some (Proof.Events152.exact38938RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38937.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38937.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38937.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39219

namespace LeftBound39220
def owner : Owner := ⟨.program ⟨214⟩, ⟨28762⟩⟩
def transferEvent : Nat := 39220
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39215 .summary) (.transfer 39219) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39215 .summary)
      LeftBound39214.bound (LeftBound39214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25231⟩⟩) (rawTerms := some (Proof.Events153.exact39215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39219)
      LeftBound39219.bound (LeftBound39219.actual selector witness) := by
  exact .transfer (LeftBound39219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39214.bound LeftBound39219.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39214.bound, LeftBound39219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39214.actual selector witness) * (LeftBound39219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39220

namespace LeftBound39231
def owner : Owner := ⟨.program ⟨214⟩, ⟨21986⟩⟩
def transferEvent : Nat := 39231
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 39229 .coefficient) (.value (.predecessor 1 39230 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39229 .coefficient)
      LeftAuthority39227.bound (LeftAuthority39227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39230 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39227.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39227.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39227.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39231

namespace LeftBound39235
def owner : Owner := ⟨.program ⟨214⟩, ⟨21987⟩⟩
def transferEvent : Nat := 39235
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39233 .coefficient) (.predecessor 1 39234 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39233 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39234 .coefficient)
      LeftBound39231.bound (LeftBound39231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound39231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound39231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound39231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39235

namespace LeftBound39236
def owner : Owner := ⟨.program ⟨214⟩, ⟨21987⟩⟩
def transferEvent : Nat := 39236
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21984⟩⟩]⟩ [⟨.result 39228 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39228 .coefficient)
      LeftAuthority39227.bound (LeftAuthority39227.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21984⟩⟩) (rawTerms := some (Proof.Events153.exact39228RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39227.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39227.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39227.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39236

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
