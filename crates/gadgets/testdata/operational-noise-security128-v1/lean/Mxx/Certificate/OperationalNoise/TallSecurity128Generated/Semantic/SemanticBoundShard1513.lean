import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1494
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1512

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound224797
def owner : Owner := ⟨.program ⟨257⟩, ⟨36025⟩⟩
def transferEvent : Nat := 224797
def frameStart : Nat := 224710
def rule : BoundRule := .sum [.predecessor 0 224795 .coefficient, .predecessor 1 224796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224795 .coefficient)
      LeftBound224792.bound (LeftBound224792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224796 .coefficient)
      LeftBound224769.bound (LeftBound224769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224769.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224792.bound, LeftBound224769.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224792.bound, LeftBound224769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224792.actual selector witness, LeftBound224769.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224797

namespace LeftBound224801
def owner : Owner := ⟨.program ⟨257⟩, ⟨36251⟩⟩
def transferEvent : Nat := 224801
def frameStart : Nat := 224710
def rule : BoundRule := .product (.predecessor 0 224799 .coefficient) (.predecessor 1 224800 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224799 .coefficient)
      LeftBound224797.bound (LeftBound224797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224800 .coefficient)
      LeftAuthority224754.bound (LeftAuthority224754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events877.exact224755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224797.bound LeftAuthority224754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224797.bound, LeftAuthority224754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224797.actual selector witness) * (LeftAuthority224754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224801

namespace LeftBound224812
def owner : Owner := ⟨.program ⟨257⟩, ⟨34742⟩⟩
def transferEvent : Nat := 224812
def frameStart : Nat := 224710
def rule : BoundRule := .product (.predecessor 0 224810 .coefficient) (.predecessor 1 224811 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224810 .coefficient)
      LeftAuthority224765.bound (LeftAuthority224765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events877.exact224766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224811 .coefficient)
      LeftAuthority224808.bound (LeftAuthority224808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority224765.bound LeftAuthority224808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224765.bound, LeftAuthority224808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority224765.actual selector witness) * (LeftAuthority224808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224812

namespace LeftBound224820
def owner : Owner := ⟨.program ⟨257⟩, ⟨34743⟩⟩
def transferEvent : Nat := 224820
def frameStart : Nat := 224710
def rule : BoundRule := .sum [.predecessor 0 224818 .coefficient, .predecessor 1 224819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224818 .coefficient)
      LeftAuthority224816.bound (LeftAuthority224816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224819 .coefficient)
      LeftBound224812.bound (LeftBound224812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority224816.bound, LeftBound224812.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224816.bound, LeftBound224812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority224816.actual selector witness, LeftBound224812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224820

namespace LeftBound224824
def owner : Owner := ⟨.program ⟨257⟩, ⟨36252⟩⟩
def transferEvent : Nat := 224824
def frameStart : Nat := 224710
def rule : BoundRule := .sum [.predecessor 0 224822 .coefficient, .predecessor 1 224823 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224822 .coefficient)
      LeftBound224820.bound (LeftBound224820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224823 .coefficient)
      LeftBound224801.bound (LeftBound224801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224801.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224820.bound, LeftBound224801.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224820.bound, LeftBound224801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224820.actual selector witness, LeftBound224801.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224824

namespace LeftBound224837
def owner : Owner := ⟨.program ⟨257⟩, ⟨36250⟩⟩
def transferEvent : Nat := 224837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 224835 .coefficient, .predecessor 1 224836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224835 .coefficient)
      LeftBound224658.bound (LeftBound224658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224836 .coefficient)
      LeftBound224641.bound (LeftBound224641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events877.exact224648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224658.bound, LeftBound224641.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224658.bound, LeftBound224641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224658.actual selector witness, LeftBound224641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224837

namespace LeftBound224840
def owner : Owner := ⟨.program ⟨257⟩, ⟨36250⟩⟩
def transferEvent : Nat := 224840
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 224834 .summary, .result 224648 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224834 .summary)
      LeftBound224660.bound (LeftBound224660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35182⟩⟩) (rawTerms := some (Proof.Events878.exact224834RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224648 .summary)
      LeftBound224643.bound (LeftBound224643.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36249⟩⟩) (rawTerms := some (Proof.Events877.exact224648RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224660.bound, LeftBound224643.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224660.bound, LeftBound224643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224660.actual selector witness, LeftBound224643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224840

namespace LeftBound224844
def owner : Owner := ⟨.program ⟨257⟩, ⟨36606⟩⟩
def transferEvent : Nat := 224844
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 224842 .coefficient) (.predecessor 1 224843 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224842 .coefficient)
      LeftBound224837.bound (LeftBound224837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224843 .coefficient)
      LeftAuthority224563.bound (LeftAuthority224563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events877.exact224564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224563.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224837.bound LeftAuthority224563.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224837.bound, LeftAuthority224563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224837.actual selector witness) * (LeftAuthority224563.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224844

namespace LeftBound224845
def owner : Owner := ⟨.program ⟨257⟩, ⟨36606⟩⟩
def transferEvent : Nat := 224845
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36604⟩⟩]⟩ [⟨.result 224564 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224564 .coefficient)
      LeftAuthority224563.bound (LeftAuthority224563.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36604⟩⟩) (rawTerms := some (Proof.Events877.exact224564RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224563.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority224563.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority224563.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound224845

namespace LeftBound224846
def owner : Owner := ⟨.program ⟨257⟩, ⟨36606⟩⟩
def transferEvent : Nat := 224846
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 224841 .summary) (.transfer 224845) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224841 .summary)
      LeftBound224840.bound (LeftBound224840.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36250⟩⟩) (rawTerms := some (Proof.Events878.exact224841RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 224845)
      LeftBound224845.bound (LeftBound224845.actual selector witness) := by
  exact .transfer (LeftBound224845.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound224840.bound LeftBound224845.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224840.bound, LeftBound224845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound224840.actual selector witness) * (LeftBound224845.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224846

namespace LeftBound224857
def owner : Owner := ⟨.program ⟨257⟩, ⟨35478⟩⟩
def transferEvent : Nat := 224857
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 224855 .coefficient) (.value (.predecessor 1 224856 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224855 .coefficient)
      LeftAuthority224853.bound (LeftAuthority224853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224856 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority224853.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224853.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority224853.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound224857

namespace LeftBound224861
def owner : Owner := ⟨.program ⟨257⟩, ⟨35479⟩⟩
def transferEvent : Nat := 224861
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 224859 .coefficient) (.predecessor 1 224860 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224859 .coefficient)
      LeftBound222242.bound (LeftBound222242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224860 .coefficient)
      LeftBound224857.bound (LeftBound224857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224857.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222242.bound LeftBound224857.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222242.bound, LeftBound224857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222242.actual selector witness) * (LeftBound224857.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224861

namespace LeftBound224862
def owner : Owner := ⟨.program ⟨257⟩, ⟨35479⟩⟩
def transferEvent : Nat := 224862
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35476⟩⟩]⟩ [⟨.result 224854 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224854 .coefficient)
      LeftAuthority224853.bound (LeftAuthority224853.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35476⟩⟩) (rawTerms := some (Proof.Events878.exact224854RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224853.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority224853.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority224853.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound224862

namespace LeftBound224863
def owner : Owner := ⟨.program ⟨257⟩, ⟨35479⟩⟩
def transferEvent : Nat := 224863
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 222245 .summary) (.transfer 224862) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222245 .summary)
      LeftBound222243.bound (LeftBound222243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5581⟩⟩) (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 224862)
      LeftBound224862.bound (LeftBound224862.actual selector witness) := by
  exact .transfer (LeftBound224862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222243.bound LeftBound224862.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222243.bound, LeftBound224862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222243.actual selector witness) * (LeftBound224862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound224863

namespace LeftBound224958
def owner : Owner := ⟨.program ⟨257⟩, ⟨34741⟩⟩
def transferEvent : Nat := 224958
def frameStart : Nat := 224919
def rule : BoundRule := .identity (.predecessor 0 224957 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224957 .coefficient)
      LeftAuthority224955.bound (LeftAuthority224955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events878.exact224956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority224955.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority224955.derived selector witness)

def rawBound : CoeffClass := LeftAuthority224955.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority224955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority224955.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound224958

namespace LeftBound224975
def owner : Owner := ⟨.program ⟨257⟩, ⟨36102⟩⟩
def transferEvent : Nat := 224975
def frameStart : Nat := 224919
def rule : BoundRule := .sum [.predecessor 0 224973 .coefficient, .predecessor 1 224974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 224973 .coefficient)
      LeftBound224958.bound (LeftBound224958.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound224958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 224974 .coefficient)
      LeftAuthority224971.bound (LeftAuthority224971.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority224971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound224958.bound, LeftAuthority224971.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound224958.bound, LeftAuthority224971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound224958.actual selector witness, LeftAuthority224971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound224975

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
