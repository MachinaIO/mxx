import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard030

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9731
def owner : Owner := ⟨.program ⟨257⟩, ⟨57165⟩⟩
def transferEvent : Nat := 9731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9729 .coefficient, .predecessor 1 9730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9729 .coefficient)
      LeftBound9727.bound (LeftBound9727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9730 .coefficient)
      LeftBound9654.bound (LeftBound9654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9727.bound, LeftBound9654.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9727.bound, LeftBound9654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9727.actual selector witness, LeftBound9654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9731

namespace LeftBound9735
def owner : Owner := ⟨.program ⟨257⟩, ⟨60145⟩⟩
def transferEvent : Nat := 9735
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9733 .coefficient, .predecessor 1 9734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9733 .coefficient)
      LeftBound9731.bound (LeftBound9731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9734 .coefficient)
      LeftBound9646.bound (LeftBound9646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9731.bound, LeftBound9646.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9731.bound, LeftBound9646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9731.actual selector witness, LeftBound9646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9735

namespace LeftBound9739
def owner : Owner := ⟨.program ⟨257⟩, ⟨63125⟩⟩
def transferEvent : Nat := 9739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9737 .coefficient, .predecessor 1 9738 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9737 .coefficient)
      LeftBound9735.bound (LeftBound9735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9738 .coefficient)
      LeftBound9638.bound (LeftBound9638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9638.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9735.bound, LeftBound9638.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9735.bound, LeftBound9638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9735.actual selector witness, LeftBound9638.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9739

namespace LeftBound9743
def owner : Owner := ⟨.program ⟨257⟩, ⟨66730⟩⟩
def transferEvent : Nat := 9743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9741 .coefficient, .predecessor 1 9742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9741 .coefficient)
      LeftBound9739.bound (LeftBound9739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9742 .coefficient)
      LeftBound9630.bound (LeftBound9630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9739.bound, LeftBound9630.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9739.bound, LeftBound9630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9739.actual selector witness, LeftBound9630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9743

namespace LeftBound9747
def owner : Owner := ⟨.program ⟨257⟩, ⟨66731⟩⟩
def transferEvent : Nat := 9747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9745 .coefficient, .predecessor 1 9746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9745 .coefficient)
      LeftBound9743.bound (LeftBound9743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9746 .coefficient)
      LeftBound9622.bound (LeftBound9622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9622.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9743.bound, LeftBound9622.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9743.bound, LeftBound9622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9743.actual selector witness, LeftBound9622.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9747

namespace LeftBound9751
def owner : Owner := ⟨.program ⟨257⟩, ⟨66732⟩⟩
def transferEvent : Nat := 9751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9749 .coefficient, .predecessor 1 9750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9749 .coefficient)
      LeftBound9747.bound (LeftBound9747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9750 .coefficient)
      LeftBound9614.bound (LeftBound9614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9747.bound, LeftBound9614.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9747.bound, LeftBound9614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9747.actual selector witness, LeftBound9614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9751

namespace LeftBound9755
def owner : Owner := ⟨.program ⟨257⟩, ⟨66733⟩⟩
def transferEvent : Nat := 9755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9753 .coefficient, .predecessor 1 9754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9753 .coefficient)
      LeftBound9751.bound (LeftBound9751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9754 .coefficient)
      LeftBound9606.bound (LeftBound9606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9751.bound, LeftBound9606.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9751.bound, LeftBound9606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9751.actual selector witness, LeftBound9606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9755

namespace LeftBound9759
def owner : Owner := ⟨.program ⟨257⟩, ⟨66734⟩⟩
def transferEvent : Nat := 9759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9757 .coefficient, .predecessor 1 9758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9757 .coefficient)
      LeftBound9755.bound (LeftBound9755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9758 .coefficient)
      LeftBound9598.bound (LeftBound9598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9755.bound, LeftBound9598.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9755.bound, LeftBound9598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9755.actual selector witness, LeftBound9598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9759

namespace LeftBound9763
def owner : Owner := ⟨.program ⟨257⟩, ⟨66735⟩⟩
def transferEvent : Nat := 9763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9761 .coefficient, .predecessor 1 9762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9761 .coefficient)
      LeftBound9759.bound (LeftBound9759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9762 .coefficient)
      LeftBound9590.bound (LeftBound9590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9759.bound, LeftBound9590.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9759.bound, LeftBound9590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9759.actual selector witness, LeftBound9590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9763

namespace LeftBound9767
def owner : Owner := ⟨.program ⟨257⟩, ⟨66736⟩⟩
def transferEvent : Nat := 9767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9765 .coefficient, .predecessor 1 9766 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9765 .coefficient)
      LeftBound9763.bound (LeftBound9763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9766 .coefficient)
      LeftBound9582.bound (LeftBound9582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9763.bound, LeftBound9582.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9763.bound, LeftBound9582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9763.actual selector witness, LeftBound9582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9767

namespace LeftBound9771
def owner : Owner := ⟨.program ⟨257⟩, ⟨66737⟩⟩
def transferEvent : Nat := 9771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9769 .coefficient, .predecessor 1 9770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9769 .coefficient)
      LeftBound9767.bound (LeftBound9767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9770 .coefficient)
      LeftBound9574.bound (LeftBound9574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9767.bound, LeftBound9574.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9767.bound, LeftBound9574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9767.actual selector witness, LeftBound9574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9771

namespace LeftBound9775
def owner : Owner := ⟨.program ⟨257⟩, ⟨66738⟩⟩
def transferEvent : Nat := 9775
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9773 .coefficient, .predecessor 1 9774 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9773 .coefficient)
      LeftBound9771.bound (LeftBound9771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9774 .coefficient)
      LeftBound9566.bound (LeftBound9566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9771.bound, LeftBound9566.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9771.bound, LeftBound9566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9771.actual selector witness, LeftBound9566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9775

namespace LeftBound9779
def owner : Owner := ⟨.program ⟨257⟩, ⟨67497⟩⟩
def transferEvent : Nat := 9779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9777 .coefficient, .predecessor 1 9778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9777 .coefficient)
      LeftBound9775.bound (LeftBound9775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9778 .coefficient)
      LeftBound9558.bound (LeftBound9558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9775.bound, LeftBound9558.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9775.bound, LeftBound9558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9775.actual selector witness, LeftBound9558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9779

namespace LeftBound9783
def owner : Owner := ⟨.program ⟨257⟩, ⟨67498⟩⟩
def transferEvent : Nat := 9783
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9781 .coefficient) (.predecessor 1 9782 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9781 .coefficient)
      LeftBound9779.bound (LeftBound9779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9782 .coefficient)
      LeftAuthority9056.bound (LeftAuthority9056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound9779.bound LeftAuthority9056.bound
def bound : CoeffClass := .finite ⟨252130354449600011142383231213844742340338711334031687757681552282355491398470735629574685714783558467219994723547350190887250934531537049642405180046306868947676244249088944969144134170710039527424, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9779.bound, LeftAuthority9056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound9779.actual selector witness) * (LeftAuthority9056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9783

namespace LeftBound10306
def owner : Owner := ⟨.program ⟨257⟩, ⟨67458⟩⟩
def transferEvent : Nat := 10306
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10304 .coefficient) (.predecessor 1 10305 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10304 .coefficient)
      LeftAuthority10302.bound (LeftAuthority10302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10305 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10302.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10302.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority10302.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10306

namespace LeftBound10314
def owner : Owner := ⟨.program ⟨257⟩, ⟨48360⟩⟩
def transferEvent : Nat := 10314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10312 .coefficient) (.predecessor 1 10313 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10312 .coefficient)
      LeftAuthority10310.bound (LeftAuthority10310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10313 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10310.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10310.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority10310.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10314

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
