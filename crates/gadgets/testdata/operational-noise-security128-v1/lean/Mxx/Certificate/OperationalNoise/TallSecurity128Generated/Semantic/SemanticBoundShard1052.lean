import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard989
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard992
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard996
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1003
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1007
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1010
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1014
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1051

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound157743
def owner : Owner := ⟨.program ⟨257⟩, ⟨69945⟩⟩
def transferEvent : Nat := 157743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157741 .coefficient, .predecessor 1 157742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157741 .coefficient)
      LeftBound157738.bound (LeftBound157738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157742 .coefficient)
      LeftBound152871.bound (LeftBound152871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events597.exact152875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157738.bound, LeftBound152871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157738.bound, LeftBound152871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157738.actual selector witness, LeftBound152871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157743

namespace LeftBound157744
def owner : Owner := ⟨.program ⟨257⟩, ⟨69945⟩⟩
def transferEvent : Nat := 157744
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157740 .summary, .result 152875 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157740 .summary)
      LeftBound157739.bound (LeftBound157739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69944⟩⟩) (rawTerms := some (Proof.Events616.exact157740RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152875 .summary)
      LeftBound152874.bound (LeftBound152874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28217⟩⟩) (rawTerms := some (Proof.Events597.exact152875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157739.bound, LeftBound152874.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157739.bound, LeftBound152874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157739.actual selector witness, LeftBound152874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157744

namespace LeftBound157748
def owner : Owner := ⟨.program ⟨257⟩, ⟨69946⟩⟩
def transferEvent : Nat := 157748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157746 .coefficient, .predecessor 1 157747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157746 .coefficient)
      LeftBound157743.bound (LeftBound157743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157747 .coefficient)
      LeftBound152389.bound (LeftBound152389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157743.bound, LeftBound152389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157743.bound, LeftBound152389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157743.actual selector witness, LeftBound152389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157748

namespace LeftBound157749
def owner : Owner := ⟨.program ⟨257⟩, ⟨69946⟩⟩
def transferEvent : Nat := 157749
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157745 .summary, .result 152393 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157745 .summary)
      LeftBound157744.bound (LeftBound157744.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69945⟩⟩) (rawTerms := some (Proof.Events616.exact157745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152393 .summary)
      LeftBound152392.bound (LeftBound152392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30897⟩⟩) (rawTerms := some (Proof.Events595.exact152393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157744.bound, LeftBound152392.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157744.bound, LeftBound152392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157744.actual selector witness, LeftBound152392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157749

namespace LeftBound157753
def owner : Owner := ⟨.program ⟨257⟩, ⟨69947⟩⟩
def transferEvent : Nat := 157753
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157751 .coefficient, .predecessor 1 157752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157751 .coefficient)
      LeftBound157748.bound (LeftBound157748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157752 .coefficient)
      LeftBound151907.bound (LeftBound151907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157748.bound, LeftBound151907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157748.bound, LeftBound151907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157748.actual selector witness, LeftBound151907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157753

namespace LeftBound157754
def owner : Owner := ⟨.program ⟨257⟩, ⟨69947⟩⟩
def transferEvent : Nat := 157754
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157750 .summary, .result 151911 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157750 .summary)
      LeftBound157749.bound (LeftBound157749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69946⟩⟩) (rawTerms := some (Proof.Events616.exact157750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151911 .summary)
      LeftBound151910.bound (LeftBound151910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36557⟩⟩) (rawTerms := some (Proof.Events593.exact151911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157749.bound, LeftBound151910.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157749.bound, LeftBound151910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157749.actual selector witness, LeftBound151910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157754

namespace LeftBound157758
def owner : Owner := ⟨.program ⟨257⟩, ⟨69948⟩⟩
def transferEvent : Nat := 157758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157756 .coefficient, .predecessor 1 157757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157756 .coefficient)
      LeftBound157753.bound (LeftBound157753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157757 .coefficient)
      LeftBound151425.bound (LeftBound151425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events591.exact151429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157753.bound, LeftBound151425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157753.bound, LeftBound151425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157753.actual selector witness, LeftBound151425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157758

namespace LeftBound157759
def owner : Owner := ⟨.program ⟨257⟩, ⟨69948⟩⟩
def transferEvent : Nat := 157759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157755 .summary, .result 151429 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157755 .summary)
      LeftBound157754.bound (LeftBound157754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69947⟩⟩) (rawTerms := some (Proof.Events616.exact157755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151429 .summary)
      LeftBound151428.bound (LeftBound151428.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39237⟩⟩) (rawTerms := some (Proof.Events591.exact151429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157754.bound, LeftBound151428.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157754.bound, LeftBound151428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157754.actual selector witness, LeftBound151428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157759

namespace LeftBound157763
def owner : Owner := ⟨.program ⟨257⟩, ⟨69949⟩⟩
def transferEvent : Nat := 157763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157761 .coefficient, .predecessor 1 157762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157761 .coefficient)
      LeftBound157758.bound (LeftBound157758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157762 .coefficient)
      LeftBound150943.bound (LeftBound150943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events589.exact150947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound150943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound150943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157758.bound, LeftBound150943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157758.bound, LeftBound150943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157758.actual selector witness, LeftBound150943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157763

namespace LeftBound157764
def owner : Owner := ⟨.program ⟨257⟩, ⟨69949⟩⟩
def transferEvent : Nat := 157764
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157760 .summary, .result 150947 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157760 .summary)
      LeftBound157759.bound (LeftBound157759.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69948⟩⟩) (rawTerms := some (Proof.Events616.exact157760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 150947 .summary)
      LeftBound150946.bound (LeftBound150946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41917⟩⟩) (rawTerms := some (Proof.Events589.exact150947RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound150946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157759.bound, LeftBound150946.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157759.bound, LeftBound150946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157759.actual selector witness, LeftBound150946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157764

namespace LeftBound157768
def owner : Owner := ⟨.program ⟨257⟩, ⟨69950⟩⟩
def transferEvent : Nat := 157768
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157766 .coefficient, .predecessor 1 157767 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157766 .coefficient)
      LeftBound157763.bound (LeftBound157763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157767 .coefficient)
      LeftBound150461.bound (LeftBound150461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events587.exact150465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound150461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound150461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157763.bound, LeftBound150461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157763.bound, LeftBound150461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157763.actual selector witness, LeftBound150461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157768

namespace LeftBound157769
def owner : Owner := ⟨.program ⟨257⟩, ⟨69950⟩⟩
def transferEvent : Nat := 157769
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157765 .summary, .result 150465 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157765 .summary)
      LeftBound157764.bound (LeftBound157764.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69949⟩⟩) (rawTerms := some (Proof.Events616.exact157765RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 150465 .summary)
      LeftBound150464.bound (LeftBound150464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44597⟩⟩) (rawTerms := some (Proof.Events587.exact150465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound150464.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157764.bound, LeftBound150464.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157764.bound, LeftBound150464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157764.actual selector witness, LeftBound150464.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157769

namespace LeftBound157773
def owner : Owner := ⟨.program ⟨257⟩, ⟨69951⟩⟩
def transferEvent : Nat := 157773
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157771 .coefficient, .predecessor 1 157772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157771 .coefficient)
      LeftBound157768.bound (LeftBound157768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157772 .coefficient)
      LeftBound149979.bound (LeftBound149979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events585.exact149983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157768.bound, LeftBound149979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157768.bound, LeftBound149979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157768.actual selector witness, LeftBound149979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157773

namespace LeftBound157774
def owner : Owner := ⟨.program ⟨257⟩, ⟨69951⟩⟩
def transferEvent : Nat := 157774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157770 .summary, .result 149983 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157770 .summary)
      LeftBound157769.bound (LeftBound157769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69950⟩⟩) (rawTerms := some (Proof.Events616.exact157770RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149983 .summary)
      LeftBound149982.bound (LeftBound149982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47277⟩⟩) (rawTerms := some (Proof.Events585.exact149983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157769.bound, LeftBound149982.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157769.bound, LeftBound149982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157769.actual selector witness, LeftBound149982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157774

namespace LeftBound157778
def owner : Owner := ⟨.program ⟨257⟩, ⟨69952⟩⟩
def transferEvent : Nat := 157778
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157776 .coefficient, .predecessor 1 157777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157776 .coefficient)
      LeftBound157773.bound (LeftBound157773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157777 .coefficient)
      LeftBound149497.bound (LeftBound149497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events583.exact149501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157773.bound, LeftBound149497.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157773.bound, LeftBound149497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157773.actual selector witness, LeftBound149497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157778

namespace LeftBound157779
def owner : Owner := ⟨.program ⟨257⟩, ⟨69952⟩⟩
def transferEvent : Nat := 157779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157775 .summary, .result 149501 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157775 .summary)
      LeftBound157774.bound (LeftBound157774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69951⟩⟩) (rawTerms := some (Proof.Events616.exact157775RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149501 .summary)
      LeftBound149500.bound (LeftBound149500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49957⟩⟩) (rawTerms := some (Proof.Events583.exact149501RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157774.bound, LeftBound149500.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157774.bound, LeftBound149500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157774.actual selector witness, LeftBound149500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157779

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
