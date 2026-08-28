import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard789
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard793
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard797
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard800
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard804
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard808
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard811
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard815
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard848

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound128488
def owner : Owner := ⟨.program ⟨257⟩, ⟨69865⟩⟩
def transferEvent : Nat := 128488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128486 .coefficient, .predecessor 1 128487 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128486 .coefficient)
      LeftBound128483.bound (LeftBound128483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128487 .coefficient)
      LeftBound124103.bound (LeftBound124103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events484.exact124107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound124103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound124103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128483.bound, LeftBound124103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128483.bound, LeftBound124103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128483.actual selector witness, LeftBound124103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128488

namespace LeftBound128489
def owner : Owner := ⟨.program ⟨257⟩, ⟨69865⟩⟩
def transferEvent : Nat := 128489
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128485 .summary, .result 124107 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128485 .summary)
      LeftBound128484.bound (LeftBound128484.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64752⟩⟩) (rawTerms := some (Proof.Events501.exact128485RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 124107 .summary)
      LeftBound124106.bound (LeftBound124106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69864⟩⟩) (rawTerms := some (Proof.Events484.exact124107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound124106.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128484.bound, LeftBound124106.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128484.bound, LeftBound124106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128484.actual selector witness, LeftBound124106.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128489

namespace LeftBound128493
def owner : Owner := ⟨.program ⟨257⟩, ⟨69866⟩⟩
def transferEvent : Nat := 128493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128491 .coefficient, .predecessor 1 128492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128491 .coefficient)
      LeftBound128488.bound (LeftBound128488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128492 .coefficient)
      LeftBound123621.bound (LeftBound123621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128488.bound, LeftBound123621.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128488.bound, LeftBound123621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128488.actual selector witness, LeftBound123621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128493

namespace LeftBound128494
def owner : Owner := ⟨.program ⟨257⟩, ⟨69866⟩⟩
def transferEvent : Nat := 128494
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128490 .summary, .result 123625 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128490 .summary)
      LeftBound128489.bound (LeftBound128489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69865⟩⟩) (rawTerms := some (Proof.Events501.exact128490RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123625 .summary)
      LeftBound123624.bound (LeftBound123624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28192⟩⟩) (rawTerms := some (Proof.Events482.exact123625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128489.bound, LeftBound123624.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128489.bound, LeftBound123624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128489.actual selector witness, LeftBound123624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128494

namespace LeftBound128498
def owner : Owner := ⟨.program ⟨257⟩, ⟨69867⟩⟩
def transferEvent : Nat := 128498
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128496 .coefficient, .predecessor 1 128497 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128496 .coefficient)
      LeftBound128493.bound (LeftBound128493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128497 .coefficient)
      LeftBound123139.bound (LeftBound123139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events481.exact123143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128493.bound, LeftBound123139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128493.bound, LeftBound123139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128493.actual selector witness, LeftBound123139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128498

namespace LeftBound128499
def owner : Owner := ⟨.program ⟨257⟩, ⟨69867⟩⟩
def transferEvent : Nat := 128499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128495 .summary, .result 123143 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128495 .summary)
      LeftBound128494.bound (LeftBound128494.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69866⟩⟩) (rawTerms := some (Proof.Events501.exact128495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123143 .summary)
      LeftBound123142.bound (LeftBound123142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30872⟩⟩) (rawTerms := some (Proof.Events481.exact123143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128494.bound, LeftBound123142.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128494.bound, LeftBound123142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128494.actual selector witness, LeftBound123142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128499

namespace LeftBound128503
def owner : Owner := ⟨.program ⟨257⟩, ⟨69868⟩⟩
def transferEvent : Nat := 128503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128501 .coefficient, .predecessor 1 128502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128501 .coefficient)
      LeftBound128498.bound (LeftBound128498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128502 .coefficient)
      LeftBound122657.bound (LeftBound122657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events479.exact122661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound122657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound122657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128498.bound, LeftBound122657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128498.bound, LeftBound122657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128498.actual selector witness, LeftBound122657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128503

namespace LeftBound128504
def owner : Owner := ⟨.program ⟨257⟩, ⟨69868⟩⟩
def transferEvent : Nat := 128504
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128500 .summary, .result 122661 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128500 .summary)
      LeftBound128499.bound (LeftBound128499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69867⟩⟩) (rawTerms := some (Proof.Events501.exact128500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 122661 .summary)
      LeftBound122660.bound (LeftBound122660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36532⟩⟩) (rawTerms := some (Proof.Events479.exact122661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound122660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128499.bound, LeftBound122660.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128499.bound, LeftBound122660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128499.actual selector witness, LeftBound122660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128504

namespace LeftBound128508
def owner : Owner := ⟨.program ⟨257⟩, ⟨69869⟩⟩
def transferEvent : Nat := 128508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128506 .coefficient, .predecessor 1 128507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128506 .coefficient)
      LeftBound128503.bound (LeftBound128503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128507 .coefficient)
      LeftBound122175.bound (LeftBound122175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events477.exact122179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound122175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound122175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128503.bound, LeftBound122175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128503.bound, LeftBound122175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128503.actual selector witness, LeftBound122175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128508

namespace LeftBound128509
def owner : Owner := ⟨.program ⟨257⟩, ⟨69869⟩⟩
def transferEvent : Nat := 128509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128505 .summary, .result 122179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128505 .summary)
      LeftBound128504.bound (LeftBound128504.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69868⟩⟩) (rawTerms := some (Proof.Events501.exact128505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 122179 .summary)
      LeftBound122178.bound (LeftBound122178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39212⟩⟩) (rawTerms := some (Proof.Events477.exact122179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound122178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128504.bound, LeftBound122178.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128504.bound, LeftBound122178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128504.actual selector witness, LeftBound122178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128509

namespace LeftBound128513
def owner : Owner := ⟨.program ⟨257⟩, ⟨69870⟩⟩
def transferEvent : Nat := 128513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128511 .coefficient, .predecessor 1 128512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128511 .coefficient)
      LeftBound128508.bound (LeftBound128508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events501.exact128510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128512 .coefficient)
      LeftBound121693.bound (LeftBound121693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events475.exact121697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128508.bound, LeftBound121693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128508.bound, LeftBound121693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128508.actual selector witness, LeftBound121693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128513

namespace LeftBound128514
def owner : Owner := ⟨.program ⟨257⟩, ⟨69870⟩⟩
def transferEvent : Nat := 128514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128510 .summary, .result 121697 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128510 .summary)
      LeftBound128509.bound (LeftBound128509.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69869⟩⟩) (rawTerms := some (Proof.Events501.exact128510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121697 .summary)
      LeftBound121696.bound (LeftBound121696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41892⟩⟩) (rawTerms := some (Proof.Events475.exact121697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128509.bound, LeftBound121696.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128509.bound, LeftBound121696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128509.actual selector witness, LeftBound121696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128514

namespace LeftBound128518
def owner : Owner := ⟨.program ⟨257⟩, ⟨69871⟩⟩
def transferEvent : Nat := 128518
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128516 .coefficient, .predecessor 1 128517 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128516 .coefficient)
      LeftBound128513.bound (LeftBound128513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events502.exact128515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128517 .coefficient)
      LeftBound121211.bound (LeftBound121211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events473.exact121215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128513.bound, LeftBound121211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128513.bound, LeftBound121211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128513.actual selector witness, LeftBound121211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128518

namespace LeftBound128519
def owner : Owner := ⟨.program ⟨257⟩, ⟨69871⟩⟩
def transferEvent : Nat := 128519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128515 .summary, .result 121215 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128515 .summary)
      LeftBound128514.bound (LeftBound128514.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69870⟩⟩) (rawTerms := some (Proof.Events502.exact128515RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121215 .summary)
      LeftBound121214.bound (LeftBound121214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44572⟩⟩) (rawTerms := some (Proof.Events473.exact121215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128514.bound, LeftBound121214.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128514.bound, LeftBound121214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128514.actual selector witness, LeftBound121214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128519

namespace LeftBound128523
def owner : Owner := ⟨.program ⟨257⟩, ⟨69872⟩⟩
def transferEvent : Nat := 128523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 128521 .coefficient, .predecessor 1 128522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 128521 .coefficient)
      LeftBound128518.bound (LeftBound128518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events502.exact128520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound128518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound128518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 128522 .coefficient)
      LeftBound120729.bound (LeftBound120729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events471.exact120733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128518.bound, LeftBound120729.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128518.bound, LeftBound120729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128518.actual selector witness, LeftBound120729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128523

namespace LeftBound128524
def owner : Owner := ⟨.program ⟨257⟩, ⟨69872⟩⟩
def transferEvent : Nat := 128524
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 128520 .summary, .result 120733 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 128520 .summary)
      LeftBound128519.bound (LeftBound128519.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69871⟩⟩) (rawTerms := some (Proof.Events502.exact128520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound128519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 120733 .summary)
      LeftBound120732.bound (LeftBound120732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47252⟩⟩) (rawTerms := some (Proof.Events471.exact120733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound120732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound128519.bound, LeftBound120732.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound128519.bound, LeftBound120732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound128519.actual selector witness, LeftBound120732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound128524

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
