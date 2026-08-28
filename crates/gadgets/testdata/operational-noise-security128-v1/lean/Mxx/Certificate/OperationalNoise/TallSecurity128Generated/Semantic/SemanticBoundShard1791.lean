import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1775
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1776
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1778
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1780
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1782
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1783
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1786
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1790

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound265562
def owner : Owner := ⟨.program ⟨257⟩, ⟨33735⟩⟩
def transferEvent : Nat := 265562
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265558 .summary, .result 264880 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265558 .summary)
      LeftBound265557.bound (LeftBound265557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23715⟩⟩) (rawTerms := some (Proof.Events1037.exact265558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 264880 .summary)
      LeftBound264875.bound (LeftBound264875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33734⟩⟩) (rawTerms := some (Proof.Events1034.exact264880RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound264875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265557.bound, LeftBound264875.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265557.bound, LeftBound264875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265557.actual selector witness, LeftBound264875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265562

namespace LeftBound265566
def owner : Owner := ⟨.program ⟨257⟩, ⟨52795⟩⟩
def transferEvent : Nat := 265566
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265564 .coefficient, .predecessor 1 265565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265564 .coefficient)
      LeftBound265561.bound (LeftBound265561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265565 .coefficient)
      LeftBound264661.bound (LeftBound264661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1033.exact264668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound264661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound264661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265561.bound, LeftBound264661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265561.bound, LeftBound264661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265561.actual selector witness, LeftBound264661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265566

namespace LeftBound265567
def owner : Owner := ⟨.program ⟨257⟩, ⟨52795⟩⟩
def transferEvent : Nat := 265567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265563 .summary, .result 264668 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265563 .summary)
      LeftBound265562.bound (LeftBound265562.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33735⟩⟩) (rawTerms := some (Proof.Events1037.exact265563RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 264668 .summary)
      LeftBound264663.bound (LeftBound264663.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52794⟩⟩) (rawTerms := some (Proof.Events1033.exact264668RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound264663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265562.bound, LeftBound264663.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265562.bound, LeftBound264663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265562.actual selector witness, LeftBound264663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265567

namespace LeftBound265571
def owner : Owner := ⟨.program ⟨257⟩, ⟨55775⟩⟩
def transferEvent : Nat := 265571
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265569 .coefficient, .predecessor 1 265570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265569 .coefficient)
      LeftBound265566.bound (LeftBound265566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265570 .coefficient)
      LeftBound264449.bound (LeftBound264449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1033.exact264456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound264449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound264449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265566.bound, LeftBound264449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265566.bound, LeftBound264449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265566.actual selector witness, LeftBound264449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265571

namespace LeftBound265572
def owner : Owner := ⟨.program ⟨257⟩, ⟨55775⟩⟩
def transferEvent : Nat := 265572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265568 .summary, .result 264456 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265568 .summary)
      LeftBound265567.bound (LeftBound265567.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52795⟩⟩) (rawTerms := some (Proof.Events1037.exact265568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 264456 .summary)
      LeftBound264451.bound (LeftBound264451.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55774⟩⟩) (rawTerms := some (Proof.Events1033.exact264456RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound264451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265567.bound, LeftBound264451.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265567.bound, LeftBound264451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265567.actual selector witness, LeftBound264451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265572

namespace LeftBound265576
def owner : Owner := ⟨.program ⟨257⟩, ⟨58755⟩⟩
def transferEvent : Nat := 265576
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265574 .coefficient, .predecessor 1 265575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265574 .coefficient)
      LeftBound265571.bound (LeftBound265571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265575 .coefficient)
      LeftBound264237.bound (LeftBound264237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1032.exact264244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound264237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound264237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265571.bound, LeftBound264237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265571.bound, LeftBound264237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265571.actual selector witness, LeftBound264237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265576

namespace LeftBound265577
def owner : Owner := ⟨.program ⟨257⟩, ⟨58755⟩⟩
def transferEvent : Nat := 265577
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265573 .summary, .result 264244 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265573 .summary)
      LeftBound265572.bound (LeftBound265572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55775⟩⟩) (rawTerms := some (Proof.Events1037.exact265573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 264244 .summary)
      LeftBound264239.bound (LeftBound264239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58754⟩⟩) (rawTerms := some (Proof.Events1032.exact264244RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound264239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265572.bound, LeftBound264239.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265572.bound, LeftBound264239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265572.actual selector witness, LeftBound264239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265577

namespace LeftBound265581
def owner : Owner := ⟨.program ⟨257⟩, ⟨61735⟩⟩
def transferEvent : Nat := 265581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265579 .coefficient, .predecessor 1 265580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265579 .coefficient)
      LeftBound265576.bound (LeftBound265576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265580 .coefficient)
      LeftBound264025.bound (LeftBound264025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1031.exact264032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound264025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound264025.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265576.bound, LeftBound264025.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265576.bound, LeftBound264025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265576.actual selector witness, LeftBound264025.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265581

namespace LeftBound265582
def owner : Owner := ⟨.program ⟨257⟩, ⟨61735⟩⟩
def transferEvent : Nat := 265582
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265578 .summary, .result 264032 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265578 .summary)
      LeftBound265577.bound (LeftBound265577.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58755⟩⟩) (rawTerms := some (Proof.Events1037.exact265578RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 264032 .summary)
      LeftBound264027.bound (LeftBound264027.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61734⟩⟩) (rawTerms := some (Proof.Events1031.exact264032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound264027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265577.bound, LeftBound264027.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265577.bound, LeftBound264027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265577.actual selector witness, LeftBound264027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265582

namespace LeftBound265586
def owner : Owner := ⟨.program ⟨257⟩, ⟨64715⟩⟩
def transferEvent : Nat := 265586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265584 .coefficient, .predecessor 1 265585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265584 .coefficient)
      LeftBound265581.bound (LeftBound265581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265585 .coefficient)
      LeftBound263813.bound (LeftBound263813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1030.exact263820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound263813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound263813.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265581.bound, LeftBound263813.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265581.bound, LeftBound263813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265581.actual selector witness, LeftBound263813.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265586

namespace LeftBound265587
def owner : Owner := ⟨.program ⟨257⟩, ⟨64715⟩⟩
def transferEvent : Nat := 265587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265583 .summary, .result 263820 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265583 .summary)
      LeftBound265582.bound (LeftBound265582.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61735⟩⟩) (rawTerms := some (Proof.Events1037.exact265583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 263820 .summary)
      LeftBound263815.bound (LeftBound263815.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64714⟩⟩) (rawTerms := some (Proof.Events1030.exact263820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound263815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265582.bound, LeftBound263815.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265582.bound, LeftBound263815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265582.actual selector witness, LeftBound263815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265587

namespace LeftBound265591
def owner : Owner := ⟨.program ⟨257⟩, ⟨69772⟩⟩
def transferEvent : Nat := 265591
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265589 .coefficient, .predecessor 1 265590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265589 .coefficient)
      LeftBound265586.bound (LeftBound265586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265590 .coefficient)
      LeftBound263601.bound (LeftBound263601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1029.exact263608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound263601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound263601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265586.bound, LeftBound263601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265586.bound, LeftBound263601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265586.actual selector witness, LeftBound263601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265591

namespace LeftBound265592
def owner : Owner := ⟨.program ⟨257⟩, ⟨69772⟩⟩
def transferEvent : Nat := 265592
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265588 .summary, .result 263608 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265588 .summary)
      LeftBound265587.bound (LeftBound265587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64715⟩⟩) (rawTerms := some (Proof.Events1037.exact265588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 263608 .summary)
      LeftBound263603.bound (LeftBound263603.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69771⟩⟩) (rawTerms := some (Proof.Events1029.exact263608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound263603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265587.bound, LeftBound263603.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265587.bound, LeftBound263603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265587.actual selector witness, LeftBound263603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265592

namespace LeftBound265596
def owner : Owner := ⟨.program ⟨257⟩, ⟨69773⟩⟩
def transferEvent : Nat := 265596
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265594 .coefficient, .predecessor 1 265595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265594 .coefficient)
      LeftBound265591.bound (LeftBound265591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265595 .coefficient)
      LeftBound263389.bound (LeftBound263389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1028.exact263396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound263389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound263389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265591.bound, LeftBound263389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265591.bound, LeftBound263389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265591.actual selector witness, LeftBound263389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265596

namespace LeftBound265597
def owner : Owner := ⟨.program ⟨257⟩, ⟨69773⟩⟩
def transferEvent : Nat := 265597
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265593 .summary, .result 263396 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265593 .summary)
      LeftBound265592.bound (LeftBound265592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69772⟩⟩) (rawTerms := some (Proof.Events1037.exact265593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 263396 .summary)
      LeftBound263391.bound (LeftBound263391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28162⟩⟩) (rawTerms := some (Proof.Events1028.exact263396RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound263391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265592.bound, LeftBound263391.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265592.bound, LeftBound263391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265592.actual selector witness, LeftBound263391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265597

namespace LeftBound265601
def owner : Owner := ⟨.program ⟨257⟩, ⟨69774⟩⟩
def transferEvent : Nat := 265601
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265599 .coefficient, .predecessor 1 265600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265599 .coefficient)
      LeftBound265596.bound (LeftBound265596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265600 .coefficient)
      LeftBound263177.bound (LeftBound263177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1028.exact263184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound263177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound263177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265596.bound, LeftBound263177.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265596.bound, LeftBound263177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265596.actual selector witness, LeftBound263177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265601

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
