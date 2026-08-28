import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1076
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1077
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1079

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound163136
def owner : Owner := ⟨.program ⟨257⟩, ⟨17674⟩⟩
def transferEvent : Nat := 163136
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163131 .summary) (.transfer 163135) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163131 .summary)
      LeftBound163130.bound (LeftBound163130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17673⟩⟩) (rawTerms := some (Proof.Events637.exact163131RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 163135)
      LeftBound163135.bound (LeftBound163135.actual selector witness) := by
  exact .transfer (LeftBound163135.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound163130.bound LeftBound163135.bound
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163130.bound, LeftBound163135.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound163130.actual selector witness) * (LeftBound163135.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163136

namespace LeftBound163144
def owner : Owner := ⟨.program ⟨257⟩, ⟨7076⟩⟩
def transferEvent : Nat := 163144
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 163142 .coefficient) (.predecessor 1 163143 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163142 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163143 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound163144

namespace LeftBound163149
def owner : Owner := ⟨.program ⟨257⟩, ⟨8256⟩⟩
def transferEvent : Nat := 163149
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 163147 .coefficient) (.predecessor 1 163148 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163147 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163148 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound15895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound15895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound163149

namespace LeftBound163154
def owner : Owner := ⟨.program ⟨257⟩, ⟨9353⟩⟩
def transferEvent : Nat := 163154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163152 .coefficient, .predecessor 1 163153 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163152 .coefficient)
      LeftBound163149.bound (LeftBound163149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163153 .coefficient)
      LeftBound163144.bound (LeftBound163144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163149.bound, LeftBound163144.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163149.bound, LeftBound163144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163149.actual selector witness, LeftBound163144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163154

namespace LeftBound163158
def owner : Owner := ⟨.program ⟨257⟩, ⟨9354⟩⟩
def transferEvent : Nat := 163158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163156 .coefficient, .predecessor 1 163157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163156 .coefficient)
      LeftBound163154.bound (LeftBound163154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163154.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163157 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163154.bound, LeftBound31515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163154.bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163154.actual selector witness, LeftBound31515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163158

namespace LeftBound163159
def owner : Owner := ⟨.program ⟨257⟩, ⟨9354⟩⟩
def transferEvent : Nat := 163159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩ [⟨.result 31516 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31516 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨118⟩⟩) (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound31515.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound31515.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound163159

namespace LeftBound163164
def owner : Owner := ⟨.program ⟨257⟩, ⟨9468⟩⟩
def transferEvent : Nat := 163164
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163162 .coefficient, .predecessor 1 163163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163162 .coefficient)
      LeftBound163158.bound (LeftBound163158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163163 .coefficient)
      LeftBound163158.bound (LeftBound163158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163158.bound, LeftBound163158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163158.bound, LeftBound163158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163158.actual selector witness, LeftBound163158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163164

namespace LeftBound163167
def owner : Owner := ⟨.program ⟨257⟩, ⟨9468⟩⟩
def transferEvent : Nat := 163167
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163161 .summary, .result 163161 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163161 .summary)
      LeftBound163159.bound (LeftBound163159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9354⟩⟩) (rawTerms := some (Proof.Events637.exact163161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163161 .summary)
      LeftBound163159.bound (LeftBound163159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9354⟩⟩) (rawTerms := some (Proof.Events637.exact163161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163159.bound, LeftBound163159.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163159.bound, LeftBound163159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163159.actual selector witness, LeftBound163159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163167

namespace LeftBound163171
def owner : Owner := ⟨.program ⟨257⟩, ⟨17675⟩⟩
def transferEvent : Nat := 163171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163169 .coefficient, .predecessor 1 163170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163169 .coefficient)
      LeftBound163164.bound (LeftBound163164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163170 .coefficient)
      LeftBound163134.bound (LeftBound163134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163164.bound, LeftBound163134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163164.bound, LeftBound163134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163164.actual selector witness, LeftBound163134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163171

namespace LeftBound163172
def owner : Owner := ⟨.program ⟨257⟩, ⟨17675⟩⟩
def transferEvent : Nat := 163172
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163168 .summary, .result 163141 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163168 .summary)
      LeftBound163167.bound (LeftBound163167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9468⟩⟩) (rawTerms := some (Proof.Events637.exact163168RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163141 .summary)
      LeftBound163136.bound (LeftBound163136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17674⟩⟩) (rawTerms := some (Proof.Events637.exact163141RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163167.bound, LeftBound163136.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163167.bound, LeftBound163136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163167.actual selector witness, LeftBound163136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163172

namespace LeftBound163176
def owner : Owner := ⟨.program ⟨257⟩, ⟨20557⟩⟩
def transferEvent : Nat := 163176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163174 .coefficient, .predecessor 1 163175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163174 .coefficient)
      LeftBound163171.bound (LeftBound163171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163175 .coefficient)
      LeftBound162922.bound (LeftBound162922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events636.exact162929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163171.bound, LeftBound162922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163171.bound, LeftBound162922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163171.actual selector witness, LeftBound162922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163176

namespace LeftBound163177
def owner : Owner := ⟨.program ⟨257⟩, ⟨20557⟩⟩
def transferEvent : Nat := 163177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163173 .summary, .result 162929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163173 .summary)
      LeftBound163172.bound (LeftBound163172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17675⟩⟩) (rawTerms := some (Proof.Events637.exact163173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162929 .summary)
      LeftBound162924.bound (LeftBound162924.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20556⟩⟩) (rawTerms := some (Proof.Events636.exact162929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163172.bound, LeftBound162924.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163172.bound, LeftBound162924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163172.actual selector witness, LeftBound162924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163177

namespace LeftBound163181
def owner : Owner := ⟨.program ⟨257⟩, ⟨23777⟩⟩
def transferEvent : Nat := 163181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163179 .coefficient, .predecessor 1 163180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163179 .coefficient)
      LeftBound163176.bound (LeftBound163176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163180 .coefficient)
      LeftBound162710.bound (LeftBound162710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events635.exact162717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163176.bound, LeftBound162710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163176.bound, LeftBound162710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163176.actual selector witness, LeftBound162710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163181

namespace LeftBound163182
def owner : Owner := ⟨.program ⟨257⟩, ⟨23777⟩⟩
def transferEvent : Nat := 163182
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163178 .summary, .result 162717 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163178 .summary)
      LeftBound163177.bound (LeftBound163177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20557⟩⟩) (rawTerms := some (Proof.Events637.exact163178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162717 .summary)
      LeftBound162712.bound (LeftBound162712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23776⟩⟩) (rawTerms := some (Proof.Events635.exact162717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163177.bound, LeftBound162712.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163177.bound, LeftBound162712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163177.actual selector witness, LeftBound162712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163182

namespace LeftBound163186
def owner : Owner := ⟨.program ⟨257⟩, ⟨33797⟩⟩
def transferEvent : Nat := 163186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163184 .coefficient, .predecessor 1 163185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163184 .coefficient)
      LeftBound163181.bound (LeftBound163181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163185 .coefficient)
      LeftBound162498.bound (LeftBound162498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events634.exact162505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163181.bound, LeftBound162498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163181.bound, LeftBound162498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163181.actual selector witness, LeftBound162498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163186

namespace LeftBound163187
def owner : Owner := ⟨.program ⟨257⟩, ⟨33797⟩⟩
def transferEvent : Nat := 163187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163183 .summary, .result 162505 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163183 .summary)
      LeftBound163182.bound (LeftBound163182.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23777⟩⟩) (rawTerms := some (Proof.Events637.exact163183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162505 .summary)
      LeftBound162500.bound (LeftBound162500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33796⟩⟩) (rawTerms := some (Proof.Events634.exact162505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163182.bound, LeftBound162500.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163182.bound, LeftBound162500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163182.actual selector witness, LeftBound162500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163187

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
