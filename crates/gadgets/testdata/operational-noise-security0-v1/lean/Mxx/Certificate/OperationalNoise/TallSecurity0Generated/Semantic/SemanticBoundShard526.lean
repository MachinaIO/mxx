import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard486
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard525

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78065
def owner : Owner := ⟨.program ⟨214⟩, ⟨15892⟩⟩
def transferEvent : Nat := 78065
def frameStart : Nat := 78009
def rule : BoundRule := .sum [.predecessor 0 78063 .coefficient, .predecessor 1 78064 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78063 .coefficient)
      LeftBound78048.bound (LeftBound78048.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78064 .coefficient)
      LeftAuthority78061.bound (LeftAuthority78061.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority78061.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78048.bound, LeftAuthority78061.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78048.bound, LeftAuthority78061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78048.actual selector witness, LeftAuthority78061.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78065

namespace LeftBound78068
def owner : Owner := ⟨.program ⟨214⟩, ⟨15893⟩⟩
def transferEvent : Nat := 78068
def frameStart : Nat := 78009
def rule : BoundRule := .identity (.predecessor 0 78067 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78067 .coefficient)
      LeftBound78065.bound (LeftBound78065.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78065.derived selector witness)

def rawBound : CoeffClass := LeftBound78065.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound78065.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78068

namespace LeftBound78074
def owner : Owner := ⟨.program ⟨214⟩, ⟨15894⟩⟩
def transferEvent : Nat := 78074
def frameStart : Nat := 78009
def rule : BoundRule := .product (.predecessor 0 78072 .coefficient) (.predecessor 1 78073 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78072 .coefficient)
      LeftAuthority78070.bound (LeftAuthority78070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78073 .coefficient)
      LeftBound78068.bound (LeftBound78068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78068.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority78070.bound LeftBound78068.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78070.bound, LeftBound78068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority78070.actual selector witness) * (LeftBound78068.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78074

namespace LeftBound78082
def owner : Owner := ⟨.program ⟨214⟩, ⟨15895⟩⟩
def transferEvent : Nat := 78082
def frameStart : Nat := 78009
def rule : BoundRule := .sum [.predecessor 0 78080 .coefficient, .predecessor 1 78081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78080 .coefficient)
      LeftAuthority78078.bound (LeftAuthority78078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78081 .coefficient)
      LeftBound78074.bound (LeftBound78074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78078.bound, LeftBound78074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78078.bound, LeftBound78074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78078.actual selector witness, LeftBound78074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78082

namespace LeftBound78086
def owner : Owner := ⟨.program ⟨214⟩, ⟨27630⟩⟩
def transferEvent : Nat := 78086
def frameStart : Nat := 78009
def rule : BoundRule := .product (.predecessor 0 78084 .coefficient) (.predecessor 1 78085 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78084 .coefficient)
      LeftBound78082.bound (LeftBound78082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78085 .coefficient)
      LeftAuthority78059.bound (LeftAuthority78059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78059.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78082.bound LeftAuthority78059.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78082.bound, LeftAuthority78059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78082.actual selector witness) * (LeftAuthority78059.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78086

namespace LeftBound78097
def owner : Owner := ⟨.program ⟨214⟩, ⟨17219⟩⟩
def transferEvent : Nat := 78097
def frameStart : Nat := 78009
def rule : BoundRule := .product (.predecessor 0 78095 .coefficient) (.predecessor 1 78096 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78095 .coefficient)
      LeftAuthority78070.bound (LeftAuthority78070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact78071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78070.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78096 .coefficient)
      LeftAuthority78093.bound (LeftAuthority78093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78070.bound LeftAuthority78093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78070.bound, LeftAuthority78093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority78070.actual selector witness) * (LeftAuthority78093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78097

namespace LeftBound78105
def owner : Owner := ⟨.program ⟨214⟩, ⟨17220⟩⟩
def transferEvent : Nat := 78105
def frameStart : Nat := 78009
def rule : BoundRule := .sum [.predecessor 0 78103 .coefficient, .predecessor 1 78104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78103 .coefficient)
      LeftAuthority78101.bound (LeftAuthority78101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78104 .coefficient)
      LeftBound78097.bound (LeftBound78097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78101.bound, LeftBound78097.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78101.bound, LeftBound78097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78101.actual selector witness, LeftBound78097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78105

namespace LeftBound78109
def owner : Owner := ⟨.program ⟨214⟩, ⟨27635⟩⟩
def transferEvent : Nat := 78109
def frameStart : Nat := 78009
def rule : BoundRule := .sum [.predecessor 0 78107 .coefficient, .predecessor 1 78108 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78107 .coefficient)
      LeftBound78105.bound (LeftBound78105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78108 .coefficient)
      LeftBound78086.bound (LeftBound78086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78105.bound, LeftBound78086.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78105.bound, LeftBound78086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78105.actual selector witness, LeftBound78086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78109

namespace LeftBound78122
def owner : Owner := ⟨.program ⟨214⟩, ⟨27632⟩⟩
def transferEvent : Nat := 78122
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78120 .coefficient, .predecessor 1 78121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78120 .coefficient)
      LeftBound77951.bound (LeftBound77951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78121 .coefficient)
      LeftBound77934.bound (LeftBound77934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77951.bound, LeftBound77934.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77951.bound, LeftBound77934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77951.actual selector witness, LeftBound77934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78122

namespace LeftBound78125
def owner : Owner := ⟨.program ⟨214⟩, ⟨27632⟩⟩
def transferEvent : Nat := 78125
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 78119 .summary, .result 77941 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78119 .summary)
      LeftBound77953.bound (LeftBound77953.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21183⟩⟩) (rawTerms := some (Proof.Events305.exact78119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77941 .summary)
      LeftBound77936.bound (LeftBound77936.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27631⟩⟩) (rawTerms := some (Proof.Events304.exact77941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77953.bound, LeftBound77936.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77953.bound, LeftBound77936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77953.actual selector witness, LeftBound77936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78125

namespace LeftBound78129
def owner : Owner := ⟨.program ⟨214⟩, ⟨27633⟩⟩
def transferEvent : Nat := 78129
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78127 .coefficient) (.predecessor 1 78128 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78127 .coefficient)
      LeftBound78122.bound (LeftBound78122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78128 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78122.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78122.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78122.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78129

namespace LeftBound78130
def owner : Owner := ⟨.program ⟨214⟩, ⟨27633⟩⟩
def transferEvent : Nat := 78130
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩ [⟨.result 5735 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5735 .coefficient)
      LeftAuthority5734.bound (LeftAuthority5734.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6643⟩⟩) (rawTerms := some (Proof.Events022.exact5735RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5734.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5734.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5734.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78130

namespace LeftBound78131
def owner : Owner := ⟨.program ⟨214⟩, ⟨27633⟩⟩
def transferEvent : Nat := 78131
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 78126 .summary) (.transfer 78130) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78126 .summary)
      LeftBound78125.bound (LeftBound78125.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27632⟩⟩) (rawTerms := some (Proof.Events305.exact78126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78130)
      LeftBound78130.bound (LeftBound78130.actual selector witness) := by
  exact .transfer (LeftBound78130.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78125.bound LeftBound78130.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78125.bound, LeftBound78130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78125.actual selector witness) * (LeftBound78130.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78131

namespace LeftBound78146
def owner : Owner := ⟨.program ⟨214⟩, ⟨27414⟩⟩
def transferEvent : Nat := 78146
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78144 .coefficient) (.predecessor 1 78145 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78144 .coefficient)
      LeftBound71353.bound (LeftBound71353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78145 .coefficient)
      LeftAuthority78142.bound (LeftAuthority78142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71353.bound LeftAuthority78142.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71353.bound, LeftAuthority78142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71353.actual selector witness) * (LeftAuthority78142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78146

namespace LeftBound78147
def owner : Owner := ⟨.program ⟨214⟩, ⟨27414⟩⟩
def transferEvent : Nat := 78147
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27412⟩⟩]⟩ [⟨.result 78143 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78143 .coefficient)
      LeftAuthority78142.bound (LeftAuthority78142.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27412⟩⟩) (rawTerms := some (Proof.Events305.exact78143RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78142.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78142.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78142.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78147

namespace LeftBound78148
def owner : Owner := ⟨.program ⟨214⟩, ⟨27414⟩⟩
def transferEvent : Nat := 78148
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71357 .summary) (.transfer 78147) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71357 .summary)
      LeftBound71356.bound (LeftBound71356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25909⟩⟩) (rawTerms := some (Proof.Events278.exact71357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78147)
      LeftBound78147.bound (LeftBound78147.actual selector witness) := by
  exact .transfer (LeftBound78147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71356.bound LeftBound78147.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71356.bound, LeftBound78147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71356.actual selector witness) * (LeftBound78147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78148

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
