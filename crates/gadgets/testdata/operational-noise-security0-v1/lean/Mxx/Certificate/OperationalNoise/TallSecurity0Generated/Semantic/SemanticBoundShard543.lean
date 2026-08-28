import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80010
def owner : Owner := ⟨.program ⟨214⟩, ⟨5541⟩⟩
def transferEvent : Nat := 80010
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩ [⟨.result 6548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6548 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22⟩⟩) (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6547.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80010

namespace LeftBound80015
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def transferEvent : Nat := 80015
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80013 .coefficient) (.predecessor 1 80014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80013 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80014 .coefficient)
      LeftBound80000.bound (LeftBound80000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound80000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound80000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound80000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80015

namespace LeftBound80016
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def transferEvent : Nat := 80016
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20248⟩⟩]⟩ [⟨.result 79997 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79997 .coefficient)
      LeftAuthority79996.bound (LeftAuthority79996.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20248⟩⟩) (rawTerms := some (Proof.Events312.exact79997RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79996.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79996.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79996.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80016

namespace LeftBound80017
def owner : Owner := ⟨.program ⟨214⟩, ⟨20251⟩⟩
def transferEvent : Nat := 80017
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 80016) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80016)
      LeftBound80016.bound (LeftBound80016.actual selector witness) := by
  exact .transfer (LeftBound80016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound80016.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound80016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound80016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80017

namespace LeftBound80096
def owner : Owner := ⟨.program ⟨214⟩, ⟨13351⟩⟩
def transferEvent : Nat := 80096
def frameStart : Nat := 80067
def rule : BoundRule := .product (.predecessor 0 80094 .coefficient) (.predecessor 1 80095 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80094 .coefficient)
      LeftAuthority80092.bound (LeftAuthority80092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80095 .coefficient)
      LeftAuthority80089.bound (LeftAuthority80089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority80092.bound LeftAuthority80089.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80092.bound, LeftAuthority80089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority80092.actual selector witness) * (LeftAuthority80089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80096

namespace LeftBound80100
def owner : Owner := ⟨.program ⟨214⟩, ⟨13352⟩⟩
def transferEvent : Nat := 80100
def frameStart : Nat := 80067
def rule : BoundRule := .identity (.predecessor 0 80099 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80099 .coefficient)
      LeftBound80096.bound (LeftBound80096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80096.derived selector witness)

def rawBound : CoeffClass := LeftBound80096.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80096.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80100

namespace LeftBound80117
def owner : Owner := ⟨.program ⟨214⟩, ⟨13446⟩⟩
def transferEvent : Nat := 80117
def frameStart : Nat := 80067
def rule : BoundRule := .sum [.predecessor 0 80115 .coefficient, .predecessor 1 80116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80115 .coefficient)
      LeftBound80100.bound (LeftBound80100.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80116 .coefficient)
      LeftAuthority80113.bound (LeftAuthority80113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80100.bound, LeftAuthority80113.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80100.bound, LeftAuthority80113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80100.actual selector witness, LeftAuthority80113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80117

namespace LeftBound80120
def owner : Owner := ⟨.program ⟨214⟩, ⟨13447⟩⟩
def transferEvent : Nat := 80120
def frameStart : Nat := 80067
def rule : BoundRule := .identity (.predecessor 0 80119 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80119 .coefficient)
      LeftBound80117.bound (LeftBound80117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80117.derived selector witness)

def rawBound : CoeffClass := LeftBound80117.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80117.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80120

namespace LeftBound80126
def owner : Owner := ⟨.program ⟨214⟩, ⟨13448⟩⟩
def transferEvent : Nat := 80126
def frameStart : Nat := 80067
def rule : BoundRule := .product (.predecessor 0 80124 .coefficient) (.predecessor 1 80125 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80124 .coefficient)
      LeftAuthority80122.bound (LeftAuthority80122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80125 .coefficient)
      LeftBound80120.bound (LeftBound80120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority80122.bound LeftBound80120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80122.bound, LeftBound80120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority80122.actual selector witness) * (LeftBound80120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80126

namespace LeftBound80140
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 80140
def frameStart : Nat := 80067
def rule : BoundRule := .scale (.predecessor 0 80138 .coefficient) (.value (.predecessor 1 80139 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80138 .coefficient)
      LeftAuthority80136.bound (LeftAuthority80136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80136.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80139 .coefficient)
      LeftAuthority80070.bound (LeftAuthority80070.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80070.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80136.bound LeftAuthority80070.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80136.bound, LeftAuthority80070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80136.actual selector witness) * (LeftAuthority80070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80140

namespace LeftBound80143
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 80143
def frameStart : Nat := 80067
def rule : BoundRule := .identity (.predecessor 0 80142 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80142 .coefficient)
      LeftAuthority80130.bound (LeftAuthority80130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80130.derived selector witness)

def rawBound : CoeffClass := LeftAuthority80130.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority80130.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80143

namespace LeftBound80147
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 80147
def frameStart : Nat := 80067
def rule : BoundRule := .product (.predecessor 0 80145 .coefficient) (.predecessor 1 80146 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80145 .coefficient)
      LeftBound80143.bound (LeftBound80143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80146 .coefficient)
      LeftBound80140.bound (LeftBound80140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80140.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80143.bound LeftBound80140.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80143.bound, LeftBound80140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80143.actual selector witness) * (LeftBound80140.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80147

namespace LeftBound80152
def owner : Owner := ⟨.program ⟨214⟩, ⟨13449⟩⟩
def transferEvent : Nat := 80152
def frameStart : Nat := 80067
def rule : BoundRule := .sum [.predecessor 0 80150 .coefficient, .predecessor 1 80151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80150 .coefficient)
      LeftBound80147.bound (LeftBound80147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80151 .coefficient)
      LeftBound80126.bound (LeftBound80126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80147.bound, LeftBound80126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80147.bound, LeftBound80126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80147.actual selector witness, LeftBound80126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80152

namespace LeftBound80156
def owner : Owner := ⟨.program ⟨214⟩, ⟨25761⟩⟩
def transferEvent : Nat := 80156
def frameStart : Nat := 80067
def rule : BoundRule := .product (.predecessor 0 80154 .coefficient) (.predecessor 1 80155 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80154 .coefficient)
      LeftBound80152.bound (LeftBound80152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80155 .coefficient)
      LeftAuthority80111.bound (LeftAuthority80111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80152.bound LeftAuthority80111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80152.bound, LeftAuthority80111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80152.actual selector witness) * (LeftAuthority80111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80156

namespace LeftBound80167
def owner : Owner := ⟨.program ⟨214⟩, ⟨17013⟩⟩
def transferEvent : Nat := 80167
def frameStart : Nat := 80067
def rule : BoundRule := .product (.predecessor 0 80165 .coefficient) (.predecessor 1 80166 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80165 .coefficient)
      LeftAuthority80122.bound (LeftAuthority80122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80166 .coefficient)
      LeftAuthority80163.bound (LeftAuthority80163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority80122.bound LeftAuthority80163.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80122.bound, LeftAuthority80163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority80122.actual selector witness) * (LeftAuthority80163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80167

namespace LeftBound80175
def owner : Owner := ⟨.program ⟨214⟩, ⟨17014⟩⟩
def transferEvent : Nat := 80175
def frameStart : Nat := 80067
def rule : BoundRule := .sum [.predecessor 0 80173 .coefficient, .predecessor 1 80174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80173 .coefficient)
      LeftAuthority80171.bound (LeftAuthority80171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80171.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80174 .coefficient)
      LeftBound80167.bound (LeftBound80167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80171.bound, LeftBound80167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80171.bound, LeftBound80167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority80171.actual selector witness, LeftBound80167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80175

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
