import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard354
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard355
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard356
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard358
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard359
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard360
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard361
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard362
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard363
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard364
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard370

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60817
def owner : Owner := ⟨.program ⟨257⟩, ⟨53198⟩⟩
def transferEvent : Nat := 60817
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60813 .summary, .result 59918 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60813 .summary)
      LeftBound60812.bound (LeftBound60812.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34138⟩⟩) (rawTerms := some (Proof.Events237.exact60813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59918 .summary)
      LeftBound59913.bound (LeftBound59913.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53197⟩⟩) (rawTerms := some (Proof.Events234.exact59918RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60812.bound, LeftBound59913.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60812.bound, LeftBound59913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60812.actual selector witness, LeftBound59913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60817

namespace LeftBound60821
def owner : Owner := ⟨.program ⟨257⟩, ⟨56178⟩⟩
def transferEvent : Nat := 60821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60819 .coefficient, .predecessor 1 60820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60819 .coefficient)
      LeftBound60816.bound (LeftBound60816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60820 .coefficient)
      LeftBound59699.bound (LeftBound59699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60816.bound, LeftBound59699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60816.bound, LeftBound59699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60816.actual selector witness, LeftBound59699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60821

namespace LeftBound60822
def owner : Owner := ⟨.program ⟨257⟩, ⟨56178⟩⟩
def transferEvent : Nat := 60822
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60818 .summary, .result 59706 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60818 .summary)
      LeftBound60817.bound (LeftBound60817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53198⟩⟩) (rawTerms := some (Proof.Events237.exact60818RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59706 .summary)
      LeftBound59701.bound (LeftBound59701.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56177⟩⟩) (rawTerms := some (Proof.Events233.exact59706RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60817.bound, LeftBound59701.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60817.bound, LeftBound59701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60817.actual selector witness, LeftBound59701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60822

namespace LeftBound60826
def owner : Owner := ⟨.program ⟨257⟩, ⟨59158⟩⟩
def transferEvent : Nat := 60826
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60824 .coefficient, .predecessor 1 60825 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60824 .coefficient)
      LeftBound60821.bound (LeftBound60821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60825 .coefficient)
      LeftBound59487.bound (LeftBound59487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60821.bound, LeftBound59487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60821.bound, LeftBound59487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60821.actual selector witness, LeftBound59487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60826

namespace LeftBound60827
def owner : Owner := ⟨.program ⟨257⟩, ⟨59158⟩⟩
def transferEvent : Nat := 60827
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60823 .summary, .result 59494 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60823 .summary)
      LeftBound60822.bound (LeftBound60822.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56178⟩⟩) (rawTerms := some (Proof.Events237.exact60823RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59494 .summary)
      LeftBound59489.bound (LeftBound59489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59157⟩⟩) (rawTerms := some (Proof.Events232.exact59494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60822.bound, LeftBound59489.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60822.bound, LeftBound59489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60822.actual selector witness, LeftBound59489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60827

namespace LeftBound60831
def owner : Owner := ⟨.program ⟨257⟩, ⟨62138⟩⟩
def transferEvent : Nat := 60831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60829 .coefficient, .predecessor 1 60830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60829 .coefficient)
      LeftBound60826.bound (LeftBound60826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60830 .coefficient)
      LeftBound59275.bound (LeftBound59275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60826.bound, LeftBound59275.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60826.bound, LeftBound59275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60826.actual selector witness, LeftBound59275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60831

namespace LeftBound60832
def owner : Owner := ⟨.program ⟨257⟩, ⟨62138⟩⟩
def transferEvent : Nat := 60832
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60828 .summary, .result 59282 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60828 .summary)
      LeftBound60827.bound (LeftBound60827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59158⟩⟩) (rawTerms := some (Proof.Events237.exact60828RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59282 .summary)
      LeftBound59277.bound (LeftBound59277.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62137⟩⟩) (rawTerms := some (Proof.Events231.exact59282RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59277.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60827.bound, LeftBound59277.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60827.bound, LeftBound59277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60827.actual selector witness, LeftBound59277.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60832

namespace LeftBound60836
def owner : Owner := ⟨.program ⟨257⟩, ⟨65118⟩⟩
def transferEvent : Nat := 60836
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60834 .coefficient, .predecessor 1 60835 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60834 .coefficient)
      LeftBound60831.bound (LeftBound60831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60835 .coefficient)
      LeftBound59063.bound (LeftBound59063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact59070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59063.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60831.bound, LeftBound59063.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60831.bound, LeftBound59063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60831.actual selector witness, LeftBound59063.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60836

namespace LeftBound60837
def owner : Owner := ⟨.program ⟨257⟩, ⟨65118⟩⟩
def transferEvent : Nat := 60837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60833 .summary, .result 59070 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60833 .summary)
      LeftBound60832.bound (LeftBound60832.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62138⟩⟩) (rawTerms := some (Proof.Events237.exact60833RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59070 .summary)
      LeftBound59065.bound (LeftBound59065.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65117⟩⟩) (rawTerms := some (Proof.Events230.exact59070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59065.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60832.bound, LeftBound59065.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60832.bound, LeftBound59065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60832.actual selector witness, LeftBound59065.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60837

namespace LeftBound60841
def owner : Owner := ⟨.program ⟨257⟩, ⟨70799⟩⟩
def transferEvent : Nat := 60841
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60839 .coefficient, .predecessor 1 60840 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60839 .coefficient)
      LeftBound60836.bound (LeftBound60836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60840 .coefficient)
      LeftBound58851.bound (LeftBound58851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58851.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60836.bound, LeftBound58851.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60836.bound, LeftBound58851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60836.actual selector witness, LeftBound58851.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60841

namespace LeftBound60842
def owner : Owner := ⟨.program ⟨257⟩, ⟨70799⟩⟩
def transferEvent : Nat := 60842
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60838 .summary, .result 58858 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60838 .summary)
      LeftBound60837.bound (LeftBound60837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65118⟩⟩) (rawTerms := some (Proof.Events237.exact60838RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58858 .summary)
      LeftBound58853.bound (LeftBound58853.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70798⟩⟩) (rawTerms := some (Proof.Events229.exact58858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60837.bound, LeftBound58853.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60837.bound, LeftBound58853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60837.actual selector witness, LeftBound58853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60842

namespace LeftBound60846
def owner : Owner := ⟨.program ⟨257⟩, ⟨70800⟩⟩
def transferEvent : Nat := 60846
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60844 .coefficient, .predecessor 1 60845 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60844 .coefficient)
      LeftBound60841.bound (LeftBound60841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60845 .coefficient)
      LeftBound58639.bound (LeftBound58639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60841.bound, LeftBound58639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60841.bound, LeftBound58639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60841.actual selector witness, LeftBound58639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60846

namespace LeftBound60847
def owner : Owner := ⟨.program ⟨257⟩, ⟨70800⟩⟩
def transferEvent : Nat := 60847
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60843 .summary, .result 58646 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60843 .summary)
      LeftBound60842.bound (LeftBound60842.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70799⟩⟩) (rawTerms := some (Proof.Events237.exact60843RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58646 .summary)
      LeftBound58641.bound (LeftBound58641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28487⟩⟩) (rawTerms := some (Proof.Events229.exact58646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60842.bound, LeftBound58641.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60842.bound, LeftBound58641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60842.actual selector witness, LeftBound58641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60847

namespace LeftBound60851
def owner : Owner := ⟨.program ⟨257⟩, ⟨70801⟩⟩
def transferEvent : Nat := 60851
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60849 .coefficient, .predecessor 1 60850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60849 .coefficient)
      LeftBound60846.bound (LeftBound60846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60850 .coefficient)
      LeftBound58427.bound (LeftBound58427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60846.bound, LeftBound58427.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60846.bound, LeftBound58427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60846.actual selector witness, LeftBound58427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60851

namespace LeftBound60852
def owner : Owner := ⟨.program ⟨257⟩, ⟨70801⟩⟩
def transferEvent : Nat := 60852
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60848 .summary, .result 58434 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60848 .summary)
      LeftBound60847.bound (LeftBound60847.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70800⟩⟩) (rawTerms := some (Proof.Events237.exact60848RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58434 .summary)
      LeftBound58429.bound (LeftBound58429.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31167⟩⟩) (rawTerms := some (Proof.Events228.exact58434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60847.bound, LeftBound58429.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60847.bound, LeftBound58429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60847.actual selector witness, LeftBound58429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60852

namespace LeftBound60856
def owner : Owner := ⟨.program ⟨257⟩, ⟨70802⟩⟩
def transferEvent : Nat := 60856
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60854 .coefficient, .predecessor 1 60855 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60854 .coefficient)
      LeftBound60851.bound (LeftBound60851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60855 .coefficient)
      LeftBound58215.bound (LeftBound58215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events227.exact58222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60851.bound, LeftBound58215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60851.bound, LeftBound58215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60851.actual selector witness, LeftBound58215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60856

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
