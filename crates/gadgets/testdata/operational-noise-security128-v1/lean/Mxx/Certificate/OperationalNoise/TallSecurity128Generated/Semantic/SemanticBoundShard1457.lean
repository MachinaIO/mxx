import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1416
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1420
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1423
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1427
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1431
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1434
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1438
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1442
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1456

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound216213
def owner : Owner := ⟨.program ⟨257⟩, ⟨52956⟩⟩
def transferEvent : Nat := 216213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216211 .coefficient, .predecessor 1 216212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216211 .coefficient)
      LeftBound216208.bound (LeftBound216208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216212 .coefficient)
      LeftBound214263.bound (LeftBound214263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216208.bound, LeftBound214263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216208.bound, LeftBound214263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216208.actual selector witness, LeftBound214263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216213

namespace LeftBound216214
def owner : Owner := ⟨.program ⟨257⟩, ⟨52956⟩⟩
def transferEvent : Nat := 216214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216210 .summary, .result 214267 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216210 .summary)
      LeftBound216209.bound (LeftBound216209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33896⟩⟩) (rawTerms := some (Proof.Events844.exact216210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 214267 .summary)
      LeftBound214266.bound (LeftBound214266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52955⟩⟩) (rawTerms := some (Proof.Events836.exact214267RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound214266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216209.bound, LeftBound214266.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216209.bound, LeftBound214266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216209.actual selector witness, LeftBound214266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216214

namespace LeftBound216218
def owner : Owner := ⟨.program ⟨257⟩, ⟨55936⟩⟩
def transferEvent : Nat := 216218
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216216 .coefficient, .predecessor 1 216217 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216216 .coefficient)
      LeftBound216213.bound (LeftBound216213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216217 .coefficient)
      LeftBound213781.bound (LeftBound213781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216213.bound, LeftBound213781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216213.bound, LeftBound213781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216213.actual selector witness, LeftBound213781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216218

namespace LeftBound216219
def owner : Owner := ⟨.program ⟨257⟩, ⟨55936⟩⟩
def transferEvent : Nat := 216219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216215 .summary, .result 213785 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216215 .summary)
      LeftBound216214.bound (LeftBound216214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52956⟩⟩) (rawTerms := some (Proof.Events844.exact216215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213785 .summary)
      LeftBound213784.bound (LeftBound213784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55935⟩⟩) (rawTerms := some (Proof.Events835.exact213785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216214.bound, LeftBound213784.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216214.bound, LeftBound213784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216214.actual selector witness, LeftBound213784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216219

namespace LeftBound216223
def owner : Owner := ⟨.program ⟨257⟩, ⟨58916⟩⟩
def transferEvent : Nat := 216223
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216221 .coefficient, .predecessor 1 216222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216221 .coefficient)
      LeftBound216218.bound (LeftBound216218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216222 .coefficient)
      LeftBound213299.bound (LeftBound213299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events833.exact213303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216218.bound, LeftBound213299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216218.bound, LeftBound213299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216218.actual selector witness, LeftBound213299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216223

namespace LeftBound216224
def owner : Owner := ⟨.program ⟨257⟩, ⟨58916⟩⟩
def transferEvent : Nat := 216224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216220 .summary, .result 213303 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216220 .summary)
      LeftBound216219.bound (LeftBound216219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55936⟩⟩) (rawTerms := some (Proof.Events844.exact216220RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213303 .summary)
      LeftBound213302.bound (LeftBound213302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58915⟩⟩) (rawTerms := some (Proof.Events833.exact213303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216219.bound, LeftBound213302.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216219.bound, LeftBound213302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216219.actual selector witness, LeftBound213302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216224

namespace LeftBound216228
def owner : Owner := ⟨.program ⟨257⟩, ⟨61896⟩⟩
def transferEvent : Nat := 216228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216226 .coefficient, .predecessor 1 216227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216226 .coefficient)
      LeftBound216223.bound (LeftBound216223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216227 .coefficient)
      LeftBound212817.bound (LeftBound212817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events831.exact212821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216223.bound, LeftBound212817.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216223.bound, LeftBound212817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216223.actual selector witness, LeftBound212817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216228

namespace LeftBound216229
def owner : Owner := ⟨.program ⟨257⟩, ⟨61896⟩⟩
def transferEvent : Nat := 216229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216225 .summary, .result 212821 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216225 .summary)
      LeftBound216224.bound (LeftBound216224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58916⟩⟩) (rawTerms := some (Proof.Events844.exact216225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212821 .summary)
      LeftBound212820.bound (LeftBound212820.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61895⟩⟩) (rawTerms := some (Proof.Events831.exact212821RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound212820.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216224.bound, LeftBound212820.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216224.bound, LeftBound212820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216224.actual selector witness, LeftBound212820.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216229

namespace LeftBound216233
def owner : Owner := ⟨.program ⟨257⟩, ⟨64876⟩⟩
def transferEvent : Nat := 216233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216231 .coefficient, .predecessor 1 216232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216231 .coefficient)
      LeftBound216228.bound (LeftBound216228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216232 .coefficient)
      LeftBound212335.bound (LeftBound212335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events829.exact212339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound212335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound212335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216228.bound, LeftBound212335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216228.bound, LeftBound212335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216228.actual selector witness, LeftBound212335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216233

namespace LeftBound216234
def owner : Owner := ⟨.program ⟨257⟩, ⟨64876⟩⟩
def transferEvent : Nat := 216234
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216230 .summary, .result 212339 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216230 .summary)
      LeftBound216229.bound (LeftBound216229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61896⟩⟩) (rawTerms := some (Proof.Events844.exact216230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 212339 .summary)
      LeftBound212338.bound (LeftBound212338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64875⟩⟩) (rawTerms := some (Proof.Events829.exact212339RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound212338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216229.bound, LeftBound212338.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216229.bound, LeftBound212338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216229.actual selector witness, LeftBound212338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216234

namespace LeftBound216238
def owner : Owner := ⟨.program ⟨257⟩, ⟨70181⟩⟩
def transferEvent : Nat := 216238
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216236 .coefficient, .predecessor 1 216237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216236 .coefficient)
      LeftBound216233.bound (LeftBound216233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216237 .coefficient)
      LeftBound211853.bound (LeftBound211853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events827.exact211857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216233.bound, LeftBound211853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216233.bound, LeftBound211853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216233.actual selector witness, LeftBound211853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216238

namespace LeftBound216239
def owner : Owner := ⟨.program ⟨257⟩, ⟨70181⟩⟩
def transferEvent : Nat := 216239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216235 .summary, .result 211857 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216235 .summary)
      LeftBound216234.bound (LeftBound216234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64876⟩⟩) (rawTerms := some (Proof.Events844.exact216235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211857 .summary)
      LeftBound211856.bound (LeftBound211856.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70180⟩⟩) (rawTerms := some (Proof.Events827.exact211857RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216234.bound, LeftBound211856.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216234.bound, LeftBound211856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216234.actual selector witness, LeftBound211856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216239

namespace LeftBound216243
def owner : Owner := ⟨.program ⟨257⟩, ⟨70182⟩⟩
def transferEvent : Nat := 216243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216241 .coefficient, .predecessor 1 216242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216241 .coefficient)
      LeftBound216238.bound (LeftBound216238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216242 .coefficient)
      LeftBound211371.bound (LeftBound211371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216238.bound, LeftBound211371.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216238.bound, LeftBound211371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216238.actual selector witness, LeftBound211371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216243

namespace LeftBound216244
def owner : Owner := ⟨.program ⟨257⟩, ⟨70182⟩⟩
def transferEvent : Nat := 216244
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216240 .summary, .result 211375 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216240 .summary)
      LeftBound216239.bound (LeftBound216239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70181⟩⟩) (rawTerms := some (Proof.Events844.exact216240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211375 .summary)
      LeftBound211374.bound (LeftBound211374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28292⟩⟩) (rawTerms := some (Proof.Events825.exact211375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216239.bound, LeftBound211374.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216239.bound, LeftBound211374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216239.actual selector witness, LeftBound211374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216244

namespace LeftBound216248
def owner : Owner := ⟨.program ⟨257⟩, ⟨70183⟩⟩
def transferEvent : Nat := 216248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216246 .coefficient, .predecessor 1 216247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216246 .coefficient)
      LeftBound216243.bound (LeftBound216243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216247 .coefficient)
      LeftBound210889.bound (LeftBound210889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events823.exact210893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound210889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound210889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216243.bound, LeftBound210889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216243.bound, LeftBound210889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216243.actual selector witness, LeftBound210889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216248

namespace LeftBound216249
def owner : Owner := ⟨.program ⟨257⟩, ⟨70183⟩⟩
def transferEvent : Nat := 216249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216245 .summary, .result 210893 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216245 .summary)
      LeftBound216244.bound (LeftBound216244.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70182⟩⟩) (rawTerms := some (Proof.Events844.exact216245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 210893 .summary)
      LeftBound210892.bound (LeftBound210892.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30972⟩⟩) (rawTerms := some (Proof.Events823.exact210893RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound210892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216244.bound, LeftBound210892.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216244.bound, LeftBound210892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216244.actual selector witness, LeftBound210892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216249

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
