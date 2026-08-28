import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1815
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1818
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1822
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1826
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1829
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1833
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1836
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1840
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1862

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound274723
def owner : Owner := ⟨.program ⟨257⟩, ⟨58659⟩⟩
def transferEvent : Nat := 274723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274721 .coefficient, .predecessor 1 274722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274721 .coefficient)
      LeftBound274718.bound (LeftBound274718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274722 .coefficient)
      LeftBound271799.bound (LeftBound271799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1061.exact271803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound271799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound271799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274718.bound, LeftBound271799.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274718.bound, LeftBound271799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274718.actual selector witness, LeftBound271799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274723

namespace LeftBound274724
def owner : Owner := ⟨.program ⟨257⟩, ⟨58659⟩⟩
def transferEvent : Nat := 274724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274720 .summary, .result 271803 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274720 .summary)
      LeftBound274719.bound (LeftBound274719.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55679⟩⟩) (rawTerms := some (Proof.Events1073.exact274720RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 271803 .summary)
      LeftBound271802.bound (LeftBound271802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58658⟩⟩) (rawTerms := some (Proof.Events1061.exact271803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound271802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274719.bound, LeftBound271802.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274719.bound, LeftBound271802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274719.actual selector witness, LeftBound271802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274724

namespace LeftBound274728
def owner : Owner := ⟨.program ⟨257⟩, ⟨61639⟩⟩
def transferEvent : Nat := 274728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274726 .coefficient, .predecessor 1 274727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274726 .coefficient)
      LeftBound274723.bound (LeftBound274723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274727 .coefficient)
      LeftBound271317.bound (LeftBound271317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1059.exact271321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound271317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound271317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274723.bound, LeftBound271317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274723.bound, LeftBound271317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274723.actual selector witness, LeftBound271317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274728

namespace LeftBound274729
def owner : Owner := ⟨.program ⟨257⟩, ⟨61639⟩⟩
def transferEvent : Nat := 274729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274725 .summary, .result 271321 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274725 .summary)
      LeftBound274724.bound (LeftBound274724.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58659⟩⟩) (rawTerms := some (Proof.Events1073.exact274725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 271321 .summary)
      LeftBound271320.bound (LeftBound271320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61638⟩⟩) (rawTerms := some (Proof.Events1059.exact271321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound271320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274724.bound, LeftBound271320.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274724.bound, LeftBound271320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274724.actual selector witness, LeftBound271320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274729

namespace LeftBound274733
def owner : Owner := ⟨.program ⟨257⟩, ⟨64619⟩⟩
def transferEvent : Nat := 274733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274731 .coefficient, .predecessor 1 274732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274731 .coefficient)
      LeftBound274728.bound (LeftBound274728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274732 .coefficient)
      LeftBound270835.bound (LeftBound270835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1057.exact270839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274728.bound, LeftBound270835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274728.bound, LeftBound270835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274728.actual selector witness, LeftBound270835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274733

namespace LeftBound274734
def owner : Owner := ⟨.program ⟨257⟩, ⟨64619⟩⟩
def transferEvent : Nat := 274734
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274730 .summary, .result 270839 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274730 .summary)
      LeftBound274729.bound (LeftBound274729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61639⟩⟩) (rawTerms := some (Proof.Events1073.exact274730RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 270839 .summary)
      LeftBound270838.bound (LeftBound270838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64618⟩⟩) (rawTerms := some (Proof.Events1057.exact270839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound270838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274729.bound, LeftBound270838.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274729.bound, LeftBound270838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274729.actual selector witness, LeftBound270838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274734

namespace LeftBound274738
def owner : Owner := ⟨.program ⟨257⟩, ⟨69524⟩⟩
def transferEvent : Nat := 274738
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274736 .coefficient, .predecessor 1 274737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274736 .coefficient)
      LeftBound274733.bound (LeftBound274733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274737 .coefficient)
      LeftBound270353.bound (LeftBound270353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1056.exact270357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound270353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound270353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274733.bound, LeftBound270353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274733.bound, LeftBound270353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274733.actual selector witness, LeftBound270353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274738

namespace LeftBound274739
def owner : Owner := ⟨.program ⟨257⟩, ⟨69524⟩⟩
def transferEvent : Nat := 274739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274735 .summary, .result 270357 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274735 .summary)
      LeftBound274734.bound (LeftBound274734.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64619⟩⟩) (rawTerms := some (Proof.Events1073.exact274735RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 270357 .summary)
      LeftBound270356.bound (LeftBound270356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69523⟩⟩) (rawTerms := some (Proof.Events1056.exact270357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound270356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274734.bound, LeftBound270356.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274734.bound, LeftBound270356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274734.actual selector witness, LeftBound270356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274739

namespace LeftBound274743
def owner : Owner := ⟨.program ⟨257⟩, ⟨69525⟩⟩
def transferEvent : Nat := 274743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274741 .coefficient, .predecessor 1 274742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274741 .coefficient)
      LeftBound274738.bound (LeftBound274738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274742 .coefficient)
      LeftBound269871.bound (LeftBound269871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1054.exact269875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274738.bound, LeftBound269871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274738.bound, LeftBound269871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274738.actual selector witness, LeftBound269871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274743

namespace LeftBound274744
def owner : Owner := ⟨.program ⟨257⟩, ⟨69525⟩⟩
def transferEvent : Nat := 274744
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274740 .summary, .result 269875 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274740 .summary)
      LeftBound274739.bound (LeftBound274739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69524⟩⟩) (rawTerms := some (Proof.Events1073.exact274740RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269875 .summary)
      LeftBound269874.bound (LeftBound269874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28085⟩⟩) (rawTerms := some (Proof.Events1054.exact269875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound269874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274739.bound, LeftBound269874.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274739.bound, LeftBound269874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274739.actual selector witness, LeftBound269874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274744

namespace LeftBound274748
def owner : Owner := ⟨.program ⟨257⟩, ⟨69526⟩⟩
def transferEvent : Nat := 274748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274746 .coefficient, .predecessor 1 274747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274746 .coefficient)
      LeftBound274743.bound (LeftBound274743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274747 .coefficient)
      LeftBound269389.bound (LeftBound269389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274743.bound, LeftBound269389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274743.bound, LeftBound269389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274743.actual selector witness, LeftBound269389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274748

namespace LeftBound274749
def owner : Owner := ⟨.program ⟨257⟩, ⟨69526⟩⟩
def transferEvent : Nat := 274749
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274745 .summary, .result 269393 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274745 .summary)
      LeftBound274744.bound (LeftBound274744.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69525⟩⟩) (rawTerms := some (Proof.Events1073.exact274745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269393 .summary)
      LeftBound269392.bound (LeftBound269392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30765⟩⟩) (rawTerms := some (Proof.Events1052.exact269393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound269392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274744.bound, LeftBound269392.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274744.bound, LeftBound269392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274744.actual selector witness, LeftBound269392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274749

namespace LeftBound274753
def owner : Owner := ⟨.program ⟨257⟩, ⟨69527⟩⟩
def transferEvent : Nat := 274753
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274751 .coefficient, .predecessor 1 274752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274751 .coefficient)
      LeftBound274748.bound (LeftBound274748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274752 .coefficient)
      LeftBound268907.bound (LeftBound268907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1050.exact268911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274748.bound, LeftBound268907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274748.bound, LeftBound268907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274748.actual selector witness, LeftBound268907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274753

namespace LeftBound274754
def owner : Owner := ⟨.program ⟨257⟩, ⟨69527⟩⟩
def transferEvent : Nat := 274754
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274750 .summary, .result 268911 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274750 .summary)
      LeftBound274749.bound (LeftBound274749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69526⟩⟩) (rawTerms := some (Proof.Events1073.exact274750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268911 .summary)
      LeftBound268910.bound (LeftBound268910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36425⟩⟩) (rawTerms := some (Proof.Events1050.exact268911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274749.bound, LeftBound268910.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274749.bound, LeftBound268910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274749.actual selector witness, LeftBound268910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274754

namespace LeftBound274758
def owner : Owner := ⟨.program ⟨257⟩, ⟨69528⟩⟩
def transferEvent : Nat := 274758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274756 .coefficient, .predecessor 1 274757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274756 .coefficient)
      LeftBound274753.bound (LeftBound274753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274757 .coefficient)
      LeftBound268425.bound (LeftBound268425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1048.exact268429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274753.bound, LeftBound268425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274753.bound, LeftBound268425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274753.actual selector witness, LeftBound268425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274758

namespace LeftBound274759
def owner : Owner := ⟨.program ⟨257⟩, ⟨69528⟩⟩
def transferEvent : Nat := 274759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274755 .summary, .result 268429 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274755 .summary)
      LeftBound274754.bound (LeftBound274754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69527⟩⟩) (rawTerms := some (Proof.Events1073.exact274755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268429 .summary)
      LeftBound268428.bound (LeftBound268428.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39105⟩⟩) (rawTerms := some (Proof.Events1048.exact268429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274754.bound, LeftBound268428.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274754.bound, LeftBound268428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274754.actual selector witness, LeftBound268428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274759

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
