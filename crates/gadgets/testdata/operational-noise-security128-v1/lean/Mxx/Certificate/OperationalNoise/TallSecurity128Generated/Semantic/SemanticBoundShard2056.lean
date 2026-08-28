import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2055

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound302111
def owner : Owner := ⟨.program ⟨257⟩, ⟨12535⟩⟩
def transferEvent : Nat := 302111
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302106 .summary) (.transfer 302110) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302106 .summary)
      LeftBound302104.bound (LeftBound302104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12534⟩⟩) (rawTerms := some (Proof.Events1180.exact302106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302110)
      LeftBound302110.bound (LeftBound302110.actual selector witness) := by
  exact .transfer (LeftBound302110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302104.bound LeftBound302110.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302104.bound, LeftBound302110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302104.actual selector witness) * (LeftBound302110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302111

namespace LeftBound302119
def owner : Owner := ⟨.program ⟨257⟩, ⟨18041⟩⟩
def transferEvent : Nat := 302119
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 302117 .coefficient, .predecessor 1 302118 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302117 .coefficient)
      LeftBound302109.bound (LeftBound302109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302118 .coefficient)
      LeftBound302081.bound (LeftBound302081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302081.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302109.bound, LeftBound302081.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302109.bound, LeftBound302081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302109.actual selector witness, LeftBound302081.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302119

namespace LeftBound302121
def owner : Owner := ⟨.program ⟨257⟩, ⟨18041⟩⟩
def transferEvent : Nat := 302121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 302116 .summary, .result 302086 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302116 .summary)
      LeftBound302111.bound (LeftBound302111.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12535⟩⟩) (rawTerms := some (Proof.Events1180.exact302116RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302086 .summary)
      LeftBound302083.bound (LeftBound302083.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18040⟩⟩) (rawTerms := some (Proof.Events1180.exact302086RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302111.bound, LeftBound302083.bound]
def bound : CoeffClass := .finite ⟨279175430144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302111.bound, LeftBound302083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302111.actual selector witness, LeftBound302083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302121

namespace LeftBound302125
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def transferEvent : Nat := 302125
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302123 .coefficient) (.predecessor 1 302124 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302123 .coefficient)
      LeftBound302119.bound (LeftBound302119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302124 .coefficient)
      LeftAuthority302057.bound (LeftAuthority302057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1179.exact302058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302119.bound LeftAuthority302057.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302119.bound, LeftAuthority302057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302119.actual selector witness) * (LeftAuthority302057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302125

namespace LeftBound302126
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def transferEvent : Nat := 302126
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20109⟩⟩]⟩ [⟨.result 302058 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302058 .coefficient)
      LeftAuthority302057.bound (LeftAuthority302057.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20109⟩⟩) (rawTerms := some (Proof.Events1179.exact302058RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302057.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority302057.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority302057.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302126

namespace LeftBound302127
def owner : Owner := ⟨.program ⟨257⟩, ⟨20110⟩⟩
def transferEvent : Nat := 302127
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 302122 .summary) (.transfer 302126) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302122 .summary)
      LeftBound302121.bound (LeftBound302121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18041⟩⟩) (rawTerms := some (Proof.Events1180.exact302122RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound302121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302126)
      LeftBound302126.bound (LeftBound302126.actual selector witness) := by
  exact .transfer (LeftBound302126.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302121.bound LeftBound302126.bound
def bound : CoeffClass := .finite ⟨2997623355788031426560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302121.bound, LeftBound302126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302121.actual selector witness) * (LeftBound302126.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302127

namespace LeftBound302138
def owner : Owner := ⟨.program ⟨257⟩, ⟨19051⟩⟩
def transferEvent : Nat := 302138
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 302136 .coefficient) (.value (.predecessor 1 302137 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302136 .coefficient)
      LeftAuthority302134.bound (LeftAuthority302134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302137 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority302134.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302134.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority302134.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound302138

namespace LeftBound302142
def owner : Owner := ⟨.program ⟨257⟩, ⟨19052⟩⟩
def transferEvent : Nat := 302142
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 302140 .coefficient) (.predecessor 1 302141 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302140 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302141 .coefficient)
      LeftBound302138.bound (LeftBound302138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302138.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound302138.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound302138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound302138.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302142

namespace LeftBound302143
def owner : Owner := ⟨.program ⟨257⟩, ⟨19052⟩⟩
def transferEvent : Nat := 302143
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨19049⟩⟩]⟩ [⟨.result 302135 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 302135 .coefficient)
      LeftAuthority302134.bound (LeftAuthority302134.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨19049⟩⟩) (rawTerms := some (Proof.Events1180.exact302135RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302134.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority302134.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority302134.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound302143

namespace LeftBound302144
def owner : Owner := ⟨.program ⟨257⟩, ⟨19052⟩⟩
def transferEvent : Nat := 302144
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295195 .summary) (.transfer 302143) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295195 .summary)
      LeftBound295193.bound (LeftBound295193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨2380⟩⟩) (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 302143)
      LeftBound302143.bound (LeftBound302143.actual selector witness) := by
  exact .transfer (LeftBound302143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295193.bound LeftBound302143.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295193.bound, LeftBound302143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295193.actual selector witness) * (LeftBound302143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302144

namespace LeftBound302199
def owner : Owner := ⟨.program ⟨257⟩, ⟨18035⟩⟩
def transferEvent : Nat := 302199
def frameStart : Nat := 302182
def rule : BoundRule := .product (.predecessor 0 302197 .coefficient) (.predecessor 1 302198 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302197 .coefficient)
      LeftAuthority302195.bound (LeftAuthority302195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302195.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302198 .coefficient)
      LeftAuthority302192.bound (LeftAuthority302192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority302195.bound LeftAuthority302192.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302195.bound, LeftAuthority302192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority302195.actual selector witness) * (LeftAuthority302192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302199

namespace LeftBound302203
def owner : Owner := ⟨.program ⟨257⟩, ⟨18036⟩⟩
def transferEvent : Nat := 302203
def frameStart : Nat := 302182
def rule : BoundRule := .identity (.predecessor 0 302202 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302202 .coefficient)
      LeftBound302199.bound (LeftBound302199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302199.derived selector witness)

def rawBound : CoeffClass := LeftBound302199.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound302199.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound302203

namespace LeftBound302220
def owner : Owner := ⟨.program ⟨257⟩, ⟨19946⟩⟩
def transferEvent : Nat := 302220
def frameStart : Nat := 302182
def rule : BoundRule := .sum [.predecessor 0 302218 .coefficient, .predecessor 1 302219 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302218 .coefficient)
      LeftBound302203.bound (LeftBound302203.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound302203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302219 .coefficient)
      LeftAuthority302216.bound (LeftAuthority302216.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority302216.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound302203.bound, LeftAuthority302216.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302203.bound, LeftAuthority302216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound302203.actual selector witness, LeftAuthority302216.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound302220

namespace LeftBound302223
def owner : Owner := ⟨.program ⟨257⟩, ⟨19947⟩⟩
def transferEvent : Nat := 302223
def frameStart : Nat := 302182
def rule : BoundRule := .identity (.predecessor 0 302222 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302222 .coefficient)
      LeftBound302220.bound (LeftBound302220.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound302220.derived selector witness)

def rawBound : CoeffClass := LeftBound302220.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound302220.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound302223

namespace LeftBound302229
def owner : Owner := ⟨.program ⟨257⟩, ⟨19948⟩⟩
def transferEvent : Nat := 302229
def frameStart : Nat := 302182
def rule : BoundRule := .product (.predecessor 0 302227 .coefficient) (.predecessor 1 302228 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302227 .coefficient)
      LeftAuthority302225.bound (LeftAuthority302225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302228 .coefficient)
      LeftBound302223.bound (LeftBound302223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority302225.bound LeftBound302223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302225.bound, LeftBound302223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority302225.actual selector witness) * (LeftBound302223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound302229

namespace LeftBound302245
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 302245
def frameStart : Nat := 302182
def rule : BoundRule := .scale (.predecessor 0 302243 .coefficient) (.value (.predecessor 1 302244 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 302243 .coefficient)
      LeftAuthority302241.bound (LeftAuthority302241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority302241.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority302241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 302244 .coefficient)
      LeftAuthority302232.bound (LeftAuthority302232.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority302232.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority302241.bound LeftAuthority302232.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority302241.bound, LeftAuthority302232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority302241.actual selector witness) * (LeftAuthority302232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound302245

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
