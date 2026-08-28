import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1844
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1847
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1851
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1855
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1858
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1861

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound274655
def owner : Owner := ⟨.program ⟨257⟩, ⟨17530⟩⟩
def transferEvent : Nat := 274655
def frameStart : Nat := 274578
def rule : BoundRule := .product (.predecessor 0 274653 .coefficient) (.predecessor 1 274654 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274653 .coefficient)
      LeftBound274651.bound (LeftBound274651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274654 .coefficient)
      LeftAuthority274628.bound (LeftAuthority274628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound274651.bound LeftAuthority274628.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274651.bound, LeftAuthority274628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound274651.actual selector witness) * (LeftAuthority274628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274655

namespace LeftBound274666
def owner : Owner := ⟨.program ⟨257⟩, ⟨15904⟩⟩
def transferEvent : Nat := 274666
def frameStart : Nat := 274578
def rule : BoundRule := .product (.predecessor 0 274664 .coefficient) (.predecessor 1 274665 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274664 .coefficient)
      LeftAuthority274639.bound (LeftAuthority274639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274665 .coefficient)
      LeftAuthority274662.bound (LeftAuthority274662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority274639.bound LeftAuthority274662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274639.bound, LeftAuthority274662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority274639.actual selector witness) * (LeftAuthority274662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274666

namespace LeftBound274674
def owner : Owner := ⟨.program ⟨257⟩, ⟨15905⟩⟩
def transferEvent : Nat := 274674
def frameStart : Nat := 274578
def rule : BoundRule := .sum [.predecessor 0 274672 .coefficient, .predecessor 1 274673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274672 .coefficient)
      LeftAuthority274670.bound (LeftAuthority274670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274673 .coefficient)
      LeftBound274666.bound (LeftBound274666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority274670.bound, LeftBound274666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274670.bound, LeftBound274666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority274670.actual selector witness, LeftBound274666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274674

namespace LeftBound274678
def owner : Owner := ⟨.program ⟨257⟩, ⟨17533⟩⟩
def transferEvent : Nat := 274678
def frameStart : Nat := 274578
def rule : BoundRule := .sum [.predecessor 0 274676 .coefficient, .predecessor 1 274677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274676 .coefficient)
      LeftBound274674.bound (LeftBound274674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274677 .coefficient)
      LeftBound274655.bound (LeftBound274655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274674.bound, LeftBound274655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274674.bound, LeftBound274655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274674.actual selector witness, LeftBound274655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274678

namespace LeftBound274691
def owner : Owner := ⟨.program ⟨257⟩, ⟨17532⟩⟩
def transferEvent : Nat := 274691
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274689 .coefficient, .predecessor 1 274690 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274689 .coefficient)
      LeftBound274520.bound (LeftBound274520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274690 .coefficient)
      LeftBound274503.bound (LeftBound274503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1072.exact274510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274520.bound, LeftBound274503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274520.bound, LeftBound274503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274520.actual selector witness, LeftBound274503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274691

namespace LeftBound274694
def owner : Owner := ⟨.program ⟨257⟩, ⟨17532⟩⟩
def transferEvent : Nat := 274694
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274688 .summary, .result 274510 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274688 .summary)
      LeftBound274522.bound (LeftBound274522.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16433⟩⟩) (rawTerms := some (Proof.Events1073.exact274688RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274510 .summary)
      LeftBound274505.bound (LeftBound274505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17531⟩⟩) (rawTerms := some (Proof.Events1072.exact274510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274522.bound, LeftBound274505.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274522.bound, LeftBound274505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274522.actual selector witness, LeftBound274505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274694

namespace LeftBound274698
def owner : Owner := ⟨.program ⟨257⟩, ⟨20399⟩⟩
def transferEvent : Nat := 274698
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274696 .coefficient, .predecessor 1 274697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274696 .coefficient)
      LeftBound274691.bound (LeftBound274691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274697 .coefficient)
      LeftBound274209.bound (LeftBound274209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274691.bound, LeftBound274209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274691.bound, LeftBound274209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274691.actual selector witness, LeftBound274209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274698

namespace LeftBound274699
def owner : Owner := ⟨.program ⟨257⟩, ⟨20399⟩⟩
def transferEvent : Nat := 274699
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274695 .summary, .result 274213 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274695 .summary)
      LeftBound274694.bound (LeftBound274694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17532⟩⟩) (rawTerms := some (Proof.Events1073.exact274695RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274213 .summary)
      LeftBound274212.bound (LeftBound274212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20398⟩⟩) (rawTerms := some (Proof.Events1071.exact274213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274212.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274694.bound, LeftBound274212.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274694.bound, LeftBound274212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274694.actual selector witness, LeftBound274212.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274699

namespace LeftBound274703
def owner : Owner := ⟨.program ⟨257⟩, ⟨23619⟩⟩
def transferEvent : Nat := 274703
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274701 .coefficient, .predecessor 1 274702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274701 .coefficient)
      LeftBound274698.bound (LeftBound274698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274702 .coefficient)
      LeftBound273727.bound (LeftBound273727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274698.bound, LeftBound273727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274698.bound, LeftBound273727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274698.actual selector witness, LeftBound273727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274703

namespace LeftBound274704
def owner : Owner := ⟨.program ⟨257⟩, ⟨23619⟩⟩
def transferEvent : Nat := 274704
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274700 .summary, .result 273731 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274700 .summary)
      LeftBound274699.bound (LeftBound274699.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20399⟩⟩) (rawTerms := some (Proof.Events1073.exact274700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273731 .summary)
      LeftBound273730.bound (LeftBound273730.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23618⟩⟩) (rawTerms := some (Proof.Events1069.exact273731RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274699.bound, LeftBound273730.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274699.bound, LeftBound273730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274699.actual selector witness, LeftBound273730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274704

namespace LeftBound274708
def owner : Owner := ⟨.program ⟨257⟩, ⟨33639⟩⟩
def transferEvent : Nat := 274708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274706 .coefficient, .predecessor 1 274707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274706 .coefficient)
      LeftBound274703.bound (LeftBound274703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274707 .coefficient)
      LeftBound273245.bound (LeftBound273245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274703.bound, LeftBound273245.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274703.bound, LeftBound273245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274703.actual selector witness, LeftBound273245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274708

namespace LeftBound274709
def owner : Owner := ⟨.program ⟨257⟩, ⟨33639⟩⟩
def transferEvent : Nat := 274709
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274705 .summary, .result 273249 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274705 .summary)
      LeftBound274704.bound (LeftBound274704.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23619⟩⟩) (rawTerms := some (Proof.Events1073.exact274705RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273249 .summary)
      LeftBound273248.bound (LeftBound273248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33638⟩⟩) (rawTerms := some (Proof.Events1067.exact273249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274704.bound, LeftBound273248.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274704.bound, LeftBound273248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274704.actual selector witness, LeftBound273248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274709

namespace LeftBound274713
def owner : Owner := ⟨.program ⟨257⟩, ⟨52699⟩⟩
def transferEvent : Nat := 274713
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274711 .coefficient, .predecessor 1 274712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274711 .coefficient)
      LeftBound274708.bound (LeftBound274708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274712 .coefficient)
      LeftBound272763.bound (LeftBound272763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1065.exact272767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272763.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274708.bound, LeftBound272763.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274708.bound, LeftBound272763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274708.actual selector witness, LeftBound272763.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274713

namespace LeftBound274714
def owner : Owner := ⟨.program ⟨257⟩, ⟨52699⟩⟩
def transferEvent : Nat := 274714
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274710 .summary, .result 272767 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274710 .summary)
      LeftBound274709.bound (LeftBound274709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33639⟩⟩) (rawTerms := some (Proof.Events1073.exact274710RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272767 .summary)
      LeftBound272766.bound (LeftBound272766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52698⟩⟩) (rawTerms := some (Proof.Events1065.exact272767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound272766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274709.bound, LeftBound272766.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274709.bound, LeftBound272766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274709.actual selector witness, LeftBound272766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274714

namespace LeftBound274718
def owner : Owner := ⟨.program ⟨257⟩, ⟨55679⟩⟩
def transferEvent : Nat := 274718
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274716 .coefficient, .predecessor 1 274717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274716 .coefficient)
      LeftBound274713.bound (LeftBound274713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1073.exact274715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274717 .coefficient)
      LeftBound272281.bound (LeftBound272281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1063.exact272285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274713.bound, LeftBound272281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274713.bound, LeftBound272281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274713.actual selector witness, LeftBound272281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274718

namespace LeftBound274719
def owner : Owner := ⟨.program ⟨257⟩, ⟨55679⟩⟩
def transferEvent : Nat := 274719
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274715 .summary, .result 272285 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274715 .summary)
      LeftBound274714.bound (LeftBound274714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52699⟩⟩) (rawTerms := some (Proof.Events1073.exact274715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272285 .summary)
      LeftBound272284.bound (LeftBound272284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55678⟩⟩) (rawTerms := some (Proof.Events1063.exact272285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound272284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274714.bound, LeftBound272284.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274714.bound, LeftBound272284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274714.actual selector witness, LeftBound272284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274719

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
