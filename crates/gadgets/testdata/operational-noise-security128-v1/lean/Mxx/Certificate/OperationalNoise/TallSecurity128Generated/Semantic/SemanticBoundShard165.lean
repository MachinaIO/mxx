import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard136
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31233
def owner : Owner := ⟨.program ⟨257⟩, ⟨20031⟩⟩
def transferEvent : Nat := 31233
def frameStart : Nat := 31174
def rule : BoundRule := .identity (.predecessor 0 31232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31232 .coefficient)
      LeftBound31230.bound (LeftBound31230.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31230.derived selector witness)

def rawBound : CoeffClass := LeftBound31230.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound31230.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31233

namespace LeftBound31239
def owner : Owner := ⟨.program ⟨257⟩, ⟨20032⟩⟩
def transferEvent : Nat := 31239
def frameStart : Nat := 31174
def rule : BoundRule := .product (.predecessor 0 31237 .coefficient) (.predecessor 1 31238 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31237 .coefficient)
      LeftAuthority31235.bound (LeftAuthority31235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31238 .coefficient)
      LeftBound31233.bound (LeftBound31233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority31235.bound LeftBound31233.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31235.bound, LeftBound31233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority31235.actual selector witness) * (LeftBound31233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31239

namespace LeftBound31247
def owner : Owner := ⟨.program ⟨257⟩, ⟨20033⟩⟩
def transferEvent : Nat := 31247
def frameStart : Nat := 31174
def rule : BoundRule := .sum [.predecessor 0 31245 .coefficient, .predecessor 1 31246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31245 .coefficient)
      LeftAuthority31243.bound (LeftAuthority31243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31246 .coefficient)
      LeftBound31239.bound (LeftBound31239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31243.bound, LeftBound31239.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31243.bound, LeftBound31239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority31243.actual selector witness, LeftBound31239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31247

namespace LeftBound31251
def owner : Owner := ⟨.program ⟨257⟩, ⟨20376⟩⟩
def transferEvent : Nat := 31251
def frameStart : Nat := 31174
def rule : BoundRule := .product (.predecessor 0 31249 .coefficient) (.predecessor 1 31250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31249 .coefficient)
      LeftBound31247.bound (LeftBound31247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31250 .coefficient)
      LeftAuthority31224.bound (LeftAuthority31224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31224.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound31247.bound LeftAuthority31224.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31247.bound, LeftAuthority31224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound31247.actual selector witness) * (LeftAuthority31224.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31251

namespace LeftBound31262
def owner : Owner := ⟨.program ⟨257⟩, ⟨18698⟩⟩
def transferEvent : Nat := 31262
def frameStart : Nat := 31174
def rule : BoundRule := .product (.predecessor 0 31260 .coefficient) (.predecessor 1 31261 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31260 .coefficient)
      LeftAuthority31235.bound (LeftAuthority31235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31261 .coefficient)
      LeftAuthority31258.bound (LeftAuthority31258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31258.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority31235.bound LeftAuthority31258.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31235.bound, LeftAuthority31258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority31235.actual selector witness) * (LeftAuthority31258.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31262

namespace LeftBound31270
def owner : Owner := ⟨.program ⟨257⟩, ⟨18699⟩⟩
def transferEvent : Nat := 31270
def frameStart : Nat := 31174
def rule : BoundRule := .sum [.predecessor 0 31268 .coefficient, .predecessor 1 31269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31268 .coefficient)
      LeftAuthority31266.bound (LeftAuthority31266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31269 .coefficient)
      LeftBound31262.bound (LeftBound31262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31266.bound, LeftBound31262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31266.bound, LeftBound31262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority31266.actual selector witness, LeftBound31262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31270

namespace LeftBound31274
def owner : Owner := ⟨.program ⟨257⟩, ⟨20381⟩⟩
def transferEvent : Nat := 31274
def frameStart : Nat := 31174
def rule : BoundRule := .sum [.predecessor 0 31272 .coefficient, .predecessor 1 31273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31272 .coefficient)
      LeftBound31270.bound (LeftBound31270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31273 .coefficient)
      LeftBound31251.bound (LeftBound31251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31270.bound, LeftBound31251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31270.bound, LeftBound31251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31270.actual selector witness, LeftBound31251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31274

namespace LeftBound31287
def owner : Owner := ⟨.program ⟨257⟩, ⟨20378⟩⟩
def transferEvent : Nat := 31287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31285 .coefficient, .predecessor 1 31286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31285 .coefficient)
      LeftBound31116.bound (LeftBound31116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31286 .coefficient)
      LeftBound31099.bound (LeftBound31099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31116.bound, LeftBound31099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31116.bound, LeftBound31099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31116.actual selector witness, LeftBound31099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31287

namespace LeftBound31290
def owner : Owner := ⟨.program ⟨257⟩, ⟨20378⟩⟩
def transferEvent : Nat := 31290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31284 .summary, .result 31106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31284 .summary)
      LeftBound31118.bound (LeftBound31118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19281⟩⟩) (rawTerms := some (Proof.Events122.exact31284RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31106 .summary)
      LeftBound31101.bound (LeftBound31101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20377⟩⟩) (rawTerms := some (Proof.Events121.exact31106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31118.bound, LeftBound31101.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31118.bound, LeftBound31101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31118.actual selector witness, LeftBound31101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31290

namespace LeftBound31294
def owner : Owner := ⟨.program ⟨257⟩, ⟨20379⟩⟩
def transferEvent : Nat := 31294
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31292 .coefficient) (.predecessor 1 31293 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31292 .coefficient)
      LeftBound31287.bound (LeftBound31287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31293 .coefficient)
      LeftBound15861.bound (LeftBound15861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound31287.bound LeftBound15861.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31287.bound, LeftBound15861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound31287.actual selector witness) * (LeftBound15861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31294

namespace LeftBound31295
def owner : Owner := ⟨.program ⟨257⟩, ⟨20379⟩⟩
def transferEvent : Nat := 31295
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩ [⟨.result 15858 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15858 .coefficient)
      LeftAuthority15857.bound (LeftAuthority15857.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7165⟩⟩) (rawTerms := some (Proof.Events061.exact15858RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15857.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15857.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15857.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31295

namespace LeftBound31296
def owner : Owner := ⟨.program ⟨257⟩, ⟨20379⟩⟩
def transferEvent : Nat := 31296
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 31291 .summary) (.transfer 31295) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31291 .summary)
      LeftBound31290.bound (LeftBound31290.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20378⟩⟩) (rawTerms := some (Proof.Events122.exact31291RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 31295)
      LeftBound31295.bound (LeftBound31295.actual selector witness) := by
  exact .transfer (LeftBound31295.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound31290.bound LeftBound31295.bound
def bound : CoeffClass := .finite ⟨345625740372465499945107099923406305361920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31290.bound, LeftBound31295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound31290.actual selector witness) * (LeftBound31295.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31296

namespace LeftBound31311
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def transferEvent : Nat := 31311
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31309 .coefficient) (.predecessor 1 31310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31309 .coefficient)
      LeftBound25868.bound (LeftBound25868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31310 .coefficient)
      LeftAuthority31307.bound (LeftAuthority31307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25868.bound LeftAuthority31307.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25868.bound, LeftAuthority31307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25868.actual selector witness) * (LeftAuthority31307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31311

namespace LeftBound31312
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def transferEvent : Nat := 31312
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17510⟩⟩]⟩ [⟨.result 31308 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31308 .coefficient)
      LeftAuthority31307.bound (LeftAuthority31307.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17510⟩⟩) (rawTerms := some (Proof.Events122.exact31308RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31307.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31307.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority31307.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31312

namespace LeftBound31313
def owner : Owner := ⟨.program ⟨257⟩, ⟨17512⟩⟩
def transferEvent : Nat := 31313
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25872 .summary) (.transfer 31312) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25872 .summary)
      LeftBound25871.bound (LeftBound25871.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17265⟩⟩) (rawTerms := some (Proof.Events101.exact25872RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 31312)
      LeftBound31312.bound (LeftBound31312.actual selector witness) := by
  exact .transfer (LeftBound31312.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25871.bound LeftBound31312.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25871.bound, LeftBound31312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25871.actual selector witness) * (LeftBound31312.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31313

namespace LeftBound31324
def owner : Owner := ⟨.program ⟨257⟩, ⟨16420⟩⟩
def transferEvent : Nat := 31324
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 31322 .coefficient) (.value (.predecessor 1 31323 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31322 .coefficient)
      LeftAuthority31320.bound (LeftAuthority31320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31323 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority31320.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31320.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority31320.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound31324

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
