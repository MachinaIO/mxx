import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard571
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard572

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84019
def owner : Owner := ⟨.program ⟨214⟩, ⟨26224⟩⟩
def transferEvent : Nat := 84019
def frameStart : Nat := 83907
def rule : BoundRule := .sum [.predecessor 0 84017 .coefficient, .predecessor 1 84018 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84017 .coefficient)
      LeftBound84015.bound (LeftBound84015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84018 .coefficient)
      LeftBound83996.bound (LeftBound83996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84015.bound, LeftBound83996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84015.bound, LeftBound83996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84015.actual selector witness, LeftBound83996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84019

namespace LeftBound84032
def owner : Owner := ⟨.program ⟨214⟩, ⟨26222⟩⟩
def transferEvent : Nat := 84032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84030 .coefficient, .predecessor 1 84031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84030 .coefficient)
      LeftBound83855.bound (LeftBound83855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84031 .coefficient)
      LeftBound83838.bound (LeftBound83838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83855.bound, LeftBound83838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83855.bound, LeftBound83838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83855.actual selector witness, LeftBound83838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84032

namespace LeftBound84035
def owner : Owner := ⟨.program ⟨214⟩, ⟨26222⟩⟩
def transferEvent : Nat := 84035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84029 .summary, .result 83845 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84029 .summary)
      LeftBound83857.bound (LeftBound83857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19675⟩⟩) (rawTerms := some (Proof.Events328.exact84029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83845 .summary)
      LeftBound83840.bound (LeftBound83840.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26221⟩⟩) (rawTerms := some (Proof.Events327.exact83845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83857.bound, LeftBound83840.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83857.bound, LeftBound83840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83857.actual selector witness, LeftBound83840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84035

namespace LeftBound84039
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def transferEvent : Nat := 84039
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84037 .coefficient) (.predecessor 1 84038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84037 .coefficient)
      LeftBound84032.bound (LeftBound84032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84038 .coefficient)
      LeftAuthority83760.bound (LeftAuthority83760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83760.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84032.bound LeftAuthority83760.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84032.bound, LeftAuthority83760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84032.actual selector witness) * (LeftAuthority83760.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84039

namespace LeftBound84040
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def transferEvent : Nat := 84040
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28300⟩⟩]⟩ [⟨.result 83761 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83761 .coefficient)
      LeftAuthority83760.bound (LeftAuthority83760.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28300⟩⟩) (rawTerms := some (Proof.Events327.exact83761RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83760.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83760.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83760.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84040

namespace LeftBound84041
def owner : Owner := ⟨.program ⟨214⟩, ⟨28302⟩⟩
def transferEvent : Nat := 84041
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84036 .summary) (.transfer 84040) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84036 .summary)
      LeftBound84035.bound (LeftBound84035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26222⟩⟩) (rawTerms := some (Proof.Events328.exact84036RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84040)
      LeftBound84040.bound (LeftBound84040.actual selector witness) := by
  exact .transfer (LeftBound84040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84035.bound LeftBound84040.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84035.bound, LeftBound84040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84035.actual selector witness) * (LeftBound84040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84041

namespace LeftBound84052
def owner : Owner := ⟨.program ⟨214⟩, ⟨21690⟩⟩
def transferEvent : Nat := 84052
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 84050 .coefficient) (.value (.predecessor 1 84051 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84050 .coefficient)
      LeftAuthority84048.bound (LeftAuthority84048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84051 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84048.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84048.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84048.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84052

namespace LeftBound84056
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def transferEvent : Nat := 84056
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84054 .coefficient) (.predecessor 1 84055 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84054 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84055 .coefficient)
      LeftBound84052.bound (LeftBound84052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound84052.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound84052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound84052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84056

namespace LeftBound84057
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def transferEvent : Nat := 84057
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21688⟩⟩]⟩ [⟨.result 84049 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84049 .coefficient)
      LeftAuthority84048.bound (LeftAuthority84048.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21688⟩⟩) (rawTerms := some (Proof.Events328.exact84049RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84048.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84048.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84048.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84057

namespace LeftBound84058
def owner : Owner := ⟨.program ⟨214⟩, ⟨21691⟩⟩
def transferEvent : Nat := 84058
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 84057) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84057)
      LeftBound84057.bound (LeftBound84057.actual selector witness) := by
  exact .transfer (LeftBound84057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound84057.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound84057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound84057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84058

namespace LeftBound84153
def owner : Owner := ⟨.program ⟨214⟩, ⟨16179⟩⟩
def transferEvent : Nat := 84153
def frameStart : Nat := 84114
def rule : BoundRule := .identity (.predecessor 0 84152 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84152 .coefficient)
      LeftAuthority84150.bound (LeftAuthority84150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84150.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84150.derived selector witness)

def rawBound : CoeffClass := LeftAuthority84150.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority84150.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84153

namespace LeftBound84170
def owner : Owner := ⟨.program ⟨214⟩, ⟨16218⟩⟩
def transferEvent : Nat := 84170
def frameStart : Nat := 84114
def rule : BoundRule := .sum [.predecessor 0 84168 .coefficient, .predecessor 1 84169 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84168 .coefficient)
      LeftBound84153.bound (LeftBound84153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84169 .coefficient)
      LeftAuthority84166.bound (LeftAuthority84166.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84153.bound, LeftAuthority84166.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84153.bound, LeftAuthority84166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84153.actual selector witness, LeftAuthority84166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84170

namespace LeftBound84173
def owner : Owner := ⟨.program ⟨214⟩, ⟨16219⟩⟩
def transferEvent : Nat := 84173
def frameStart : Nat := 84114
def rule : BoundRule := .identity (.predecessor 0 84172 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84172 .coefficient)
      LeftBound84170.bound (LeftBound84170.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84170.derived selector witness)

def rawBound : CoeffClass := LeftBound84170.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84170.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84173

namespace LeftBound84179
def owner : Owner := ⟨.program ⟨214⟩, ⟨16220⟩⟩
def transferEvent : Nat := 84179
def frameStart : Nat := 84114
def rule : BoundRule := .product (.predecessor 0 84177 .coefficient) (.predecessor 1 84178 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84177 .coefficient)
      LeftAuthority84175.bound (LeftAuthority84175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84175.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84178 .coefficient)
      LeftBound84173.bound (LeftBound84173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority84175.bound LeftBound84173.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84175.bound, LeftBound84173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority84175.actual selector witness) * (LeftBound84173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84179

namespace LeftBound84187
def owner : Owner := ⟨.program ⟨214⟩, ⟨16221⟩⟩
def transferEvent : Nat := 84187
def frameStart : Nat := 84114
def rule : BoundRule := .sum [.predecessor 0 84185 .coefficient, .predecessor 1 84186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84185 .coefficient)
      LeftAuthority84183.bound (LeftAuthority84183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84186 .coefficient)
      LeftBound84179.bound (LeftBound84179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84183.bound, LeftBound84179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84183.bound, LeftBound84179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84183.actual selector witness, LeftBound84179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84187

namespace LeftBound84191
def owner : Owner := ⟨.program ⟨214⟩, ⟨28301⟩⟩
def transferEvent : Nat := 84191
def frameStart : Nat := 84114
def rule : BoundRule := .product (.predecessor 0 84189 .coefficient) (.predecessor 1 84190 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84189 .coefficient)
      LeftBound84187.bound (LeftBound84187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84190 .coefficient)
      LeftAuthority84164.bound (LeftAuthority84164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84187.bound LeftAuthority84164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84187.bound, LeftAuthority84164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84187.actual selector witness) * (LeftAuthority84164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84191

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
