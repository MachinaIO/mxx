import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1137
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1141
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1144
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1145
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1148
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1151

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound172291
def owner : Owner := ⟨.program ⟨257⟩, ⟨16100⟩⟩
def transferEvent : Nat := 172291
def frameStart : Nat := 172203
def rule : BoundRule := .product (.predecessor 0 172289 .coefficient) (.predecessor 1 172290 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172289 .coefficient)
      LeftAuthority172264.bound (LeftAuthority172264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172290 .coefficient)
      LeftAuthority172287.bound (LeftAuthority172287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172287.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority172264.bound LeftAuthority172287.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172264.bound, LeftAuthority172287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority172264.actual selector witness) * (LeftAuthority172287.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172291

namespace LeftBound172299
def owner : Owner := ⟨.program ⟨257⟩, ⟨16101⟩⟩
def transferEvent : Nat := 172299
def frameStart : Nat := 172203
def rule : BoundRule := .sum [.predecessor 0 172297 .coefficient, .predecessor 1 172298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172297 .coefficient)
      LeftAuthority172295.bound (LeftAuthority172295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172298 .coefficient)
      LeftBound172291.bound (LeftBound172291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority172295.bound, LeftBound172291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172295.bound, LeftBound172291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority172295.actual selector witness, LeftBound172291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172299

namespace LeftBound172303
def owner : Owner := ⟨.program ⟨257⟩, ⟨17877⟩⟩
def transferEvent : Nat := 172303
def frameStart : Nat := 172203
def rule : BoundRule := .sum [.predecessor 0 172301 .coefficient, .predecessor 1 172302 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172301 .coefficient)
      LeftBound172299.bound (LeftBound172299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172302 .coefficient)
      LeftBound172280.bound (LeftBound172280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172299.bound, LeftBound172280.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172299.bound, LeftBound172280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172299.actual selector witness, LeftBound172280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172303

namespace LeftBound172316
def owner : Owner := ⟨.program ⟨257⟩, ⟨17876⟩⟩
def transferEvent : Nat := 172316
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172314 .coefficient, .predecessor 1 172315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172314 .coefficient)
      LeftBound172145.bound (LeftBound172145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172315 .coefficient)
      LeftBound172128.bound (LeftBound172128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172145.bound, LeftBound172128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172145.bound, LeftBound172128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172145.actual selector witness, LeftBound172128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172316

namespace LeftBound172319
def owner : Owner := ⟨.program ⟨257⟩, ⟨17876⟩⟩
def transferEvent : Nat := 172319
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172313 .summary, .result 172135 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172313 .summary)
      LeftBound172147.bound (LeftBound172147.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16679⟩⟩) (rawTerms := some (Proof.Events673.exact172313RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172135 .summary)
      LeftBound172130.bound (LeftBound172130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17875⟩⟩) (rawTerms := some (Proof.Events672.exact172135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172147.bound, LeftBound172130.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172147.bound, LeftBound172130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172147.actual selector witness, LeftBound172130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172319

namespace LeftBound172323
def owner : Owner := ⟨.program ⟨257⟩, ⟨20780⟩⟩
def transferEvent : Nat := 172323
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172321 .coefficient, .predecessor 1 172322 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172321 .coefficient)
      LeftBound172316.bound (LeftBound172316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172322 .coefficient)
      LeftBound171834.bound (LeftBound171834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact171838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172316.bound, LeftBound171834.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172316.bound, LeftBound171834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172316.actual selector witness, LeftBound171834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172323

namespace LeftBound172324
def owner : Owner := ⟨.program ⟨257⟩, ⟨20780⟩⟩
def transferEvent : Nat := 172324
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172320 .summary, .result 171838 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172320 .summary)
      LeftBound172319.bound (LeftBound172319.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17876⟩⟩) (rawTerms := some (Proof.Events673.exact172320RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171838 .summary)
      LeftBound171837.bound (LeftBound171837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20779⟩⟩) (rawTerms := some (Proof.Events671.exact171838RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound171837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172319.bound, LeftBound171837.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172319.bound, LeftBound171837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172319.actual selector witness, LeftBound171837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172324

namespace LeftBound172328
def owner : Owner := ⟨.program ⟨257⟩, ⟨24000⟩⟩
def transferEvent : Nat := 172328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172326 .coefficient, .predecessor 1 172327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172326 .coefficient)
      LeftBound172323.bound (LeftBound172323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172327 .coefficient)
      LeftBound171352.bound (LeftBound171352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172323.bound, LeftBound171352.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172323.bound, LeftBound171352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172323.actual selector witness, LeftBound171352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172328

namespace LeftBound172329
def owner : Owner := ⟨.program ⟨257⟩, ⟨24000⟩⟩
def transferEvent : Nat := 172329
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172325 .summary, .result 171356 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172325 .summary)
      LeftBound172324.bound (LeftBound172324.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20780⟩⟩) (rawTerms := some (Proof.Events673.exact172325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171356 .summary)
      LeftBound171355.bound (LeftBound171355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23999⟩⟩) (rawTerms := some (Proof.Events669.exact171356RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound171355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172324.bound, LeftBound171355.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172324.bound, LeftBound171355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172324.actual selector witness, LeftBound171355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172329

namespace LeftBound172333
def owner : Owner := ⟨.program ⟨257⟩, ⟨34020⟩⟩
def transferEvent : Nat := 172333
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172331 .coefficient, .predecessor 1 172332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172331 .coefficient)
      LeftBound172328.bound (LeftBound172328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172332 .coefficient)
      LeftBound170870.bound (LeftBound170870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events667.exact170874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound170870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound170870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172328.bound, LeftBound170870.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172328.bound, LeftBound170870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172328.actual selector witness, LeftBound170870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172333

namespace LeftBound172334
def owner : Owner := ⟨.program ⟨257⟩, ⟨34020⟩⟩
def transferEvent : Nat := 172334
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172330 .summary, .result 170874 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172330 .summary)
      LeftBound172329.bound (LeftBound172329.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24000⟩⟩) (rawTerms := some (Proof.Events673.exact172330RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 170874 .summary)
      LeftBound170873.bound (LeftBound170873.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34019⟩⟩) (rawTerms := some (Proof.Events667.exact170874RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound170873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172329.bound, LeftBound170873.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172329.bound, LeftBound170873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172329.actual selector witness, LeftBound170873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172334

namespace LeftBound172338
def owner : Owner := ⟨.program ⟨257⟩, ⟨53080⟩⟩
def transferEvent : Nat := 172338
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172336 .coefficient, .predecessor 1 172337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172336 .coefficient)
      LeftBound172333.bound (LeftBound172333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172337 .coefficient)
      LeftBound170388.bound (LeftBound170388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events665.exact170392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound170388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound170388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172333.bound, LeftBound170388.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172333.bound, LeftBound170388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172333.actual selector witness, LeftBound170388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172338

namespace LeftBound172339
def owner : Owner := ⟨.program ⟨257⟩, ⟨53080⟩⟩
def transferEvent : Nat := 172339
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172335 .summary, .result 170392 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172335 .summary)
      LeftBound172334.bound (LeftBound172334.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34020⟩⟩) (rawTerms := some (Proof.Events673.exact172335RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 170392 .summary)
      LeftBound170391.bound (LeftBound170391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53079⟩⟩) (rawTerms := some (Proof.Events665.exact170392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound170391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172334.bound, LeftBound170391.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172334.bound, LeftBound170391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172334.actual selector witness, LeftBound170391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172339

namespace LeftBound172343
def owner : Owner := ⟨.program ⟨257⟩, ⟨56060⟩⟩
def transferEvent : Nat := 172343
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172341 .coefficient, .predecessor 1 172342 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172341 .coefficient)
      LeftBound172338.bound (LeftBound172338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172342 .coefficient)
      LeftBound169906.bound (LeftBound169906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events663.exact169910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound169906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound169906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172338.bound, LeftBound169906.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172338.bound, LeftBound169906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172338.actual selector witness, LeftBound169906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172343

namespace LeftBound172344
def owner : Owner := ⟨.program ⟨257⟩, ⟨56060⟩⟩
def transferEvent : Nat := 172344
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172340 .summary, .result 169910 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172340 .summary)
      LeftBound172339.bound (LeftBound172339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53080⟩⟩) (rawTerms := some (Proof.Events673.exact172340RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 169910 .summary)
      LeftBound169909.bound (LeftBound169909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56059⟩⟩) (rawTerms := some (Proof.Events663.exact169910RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound169909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172339.bound, LeftBound169909.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172339.bound, LeftBound169909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172339.actual selector witness, LeftBound169909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172344

namespace LeftBound172348
def owner : Owner := ⟨.program ⟨257⟩, ⟨59040⟩⟩
def transferEvent : Nat := 172348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172346 .coefficient, .predecessor 1 172347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172346 .coefficient)
      LeftBound172343.bound (LeftBound172343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172347 .coefficient)
      LeftBound169424.bound (LeftBound169424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events661.exact169428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound169424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound169424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172343.bound, LeftBound169424.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172343.bound, LeftBound169424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172343.actual selector witness, LeftBound169424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172348

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
