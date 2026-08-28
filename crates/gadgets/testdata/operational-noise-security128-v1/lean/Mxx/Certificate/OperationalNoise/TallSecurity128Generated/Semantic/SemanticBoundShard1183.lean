import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1160
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1161
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1162
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1163
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1164
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1165
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1166
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1168
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1169
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1170
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1182

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound177837
def owner : Owner := ⟨.program ⟨257⟩, ⟨64994⟩⟩
def transferEvent : Nat := 177837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177833 .summary, .result 176070 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177833 .summary)
      LeftBound177832.bound (LeftBound177832.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62014⟩⟩) (rawTerms := some (Proof.Events694.exact177833RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 176070 .summary)
      LeftBound176065.bound (LeftBound176065.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64993⟩⟩) (rawTerms := some (Proof.Events687.exact176070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound176065.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177832.bound, LeftBound176065.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177832.bound, LeftBound176065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177832.actual selector witness, LeftBound176065.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177837

namespace LeftBound177841
def owner : Owner := ⟨.program ⟨257⟩, ⟨70483⟩⟩
def transferEvent : Nat := 177841
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177839 .coefficient, .predecessor 1 177840 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177839 .coefficient)
      LeftBound177836.bound (LeftBound177836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177840 .coefficient)
      LeftBound175851.bound (LeftBound175851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events686.exact175858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound175851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound175851.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177836.bound, LeftBound175851.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177836.bound, LeftBound175851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177836.actual selector witness, LeftBound175851.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177841

namespace LeftBound177842
def owner : Owner := ⟨.program ⟨257⟩, ⟨70483⟩⟩
def transferEvent : Nat := 177842
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177838 .summary, .result 175858 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177838 .summary)
      LeftBound177837.bound (LeftBound177837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64994⟩⟩) (rawTerms := some (Proof.Events694.exact177838RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 175858 .summary)
      LeftBound175853.bound (LeftBound175853.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70482⟩⟩) (rawTerms := some (Proof.Events686.exact175858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound175853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177837.bound, LeftBound175853.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177837.bound, LeftBound175853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177837.actual selector witness, LeftBound175853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177842

namespace LeftBound177846
def owner : Owner := ⟨.program ⟨257⟩, ⟨70484⟩⟩
def transferEvent : Nat := 177846
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177844 .coefficient, .predecessor 1 177845 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177844 .coefficient)
      LeftBound177841.bound (LeftBound177841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177845 .coefficient)
      LeftBound175639.bound (LeftBound175639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events686.exact175646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound175639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound175639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177841.bound, LeftBound175639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177841.bound, LeftBound175639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177841.actual selector witness, LeftBound175639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177846

namespace LeftBound177847
def owner : Owner := ⟨.program ⟨257⟩, ⟨70484⟩⟩
def transferEvent : Nat := 177847
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177843 .summary, .result 175646 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177843 .summary)
      LeftBound177842.bound (LeftBound177842.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70483⟩⟩) (rawTerms := some (Proof.Events694.exact177843RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 175646 .summary)
      LeftBound175641.bound (LeftBound175641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28387⟩⟩) (rawTerms := some (Proof.Events686.exact175646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound175641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177842.bound, LeftBound175641.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177842.bound, LeftBound175641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177842.actual selector witness, LeftBound175641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177847

namespace LeftBound177851
def owner : Owner := ⟨.program ⟨257⟩, ⟨70485⟩⟩
def transferEvent : Nat := 177851
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177849 .coefficient, .predecessor 1 177850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177849 .coefficient)
      LeftBound177846.bound (LeftBound177846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177850 .coefficient)
      LeftBound175427.bound (LeftBound175427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events685.exact175434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound175427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound175427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177846.bound, LeftBound175427.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177846.bound, LeftBound175427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177846.actual selector witness, LeftBound175427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177851

namespace LeftBound177852
def owner : Owner := ⟨.program ⟨257⟩, ⟨70485⟩⟩
def transferEvent : Nat := 177852
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177848 .summary, .result 175434 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177848 .summary)
      LeftBound177847.bound (LeftBound177847.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70484⟩⟩) (rawTerms := some (Proof.Events694.exact177848RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 175434 .summary)
      LeftBound175429.bound (LeftBound175429.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31067⟩⟩) (rawTerms := some (Proof.Events685.exact175434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound175429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177847.bound, LeftBound175429.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177847.bound, LeftBound175429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177847.actual selector witness, LeftBound175429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177852

namespace LeftBound177856
def owner : Owner := ⟨.program ⟨257⟩, ⟨70486⟩⟩
def transferEvent : Nat := 177856
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177854 .coefficient, .predecessor 1 177855 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177854 .coefficient)
      LeftBound177851.bound (LeftBound177851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177855 .coefficient)
      LeftBound175215.bound (LeftBound175215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events684.exact175222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound175215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound175215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177851.bound, LeftBound175215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177851.bound, LeftBound175215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177851.actual selector witness, LeftBound175215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177856

namespace LeftBound177857
def owner : Owner := ⟨.program ⟨257⟩, ⟨70486⟩⟩
def transferEvent : Nat := 177857
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177853 .summary, .result 175222 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177853 .summary)
      LeftBound177852.bound (LeftBound177852.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70485⟩⟩) (rawTerms := some (Proof.Events694.exact177853RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 175222 .summary)
      LeftBound175217.bound (LeftBound175217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36727⟩⟩) (rawTerms := some (Proof.Events684.exact175222RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound175217.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177852.bound, LeftBound175217.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177852.bound, LeftBound175217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177852.actual selector witness, LeftBound175217.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177857

namespace LeftBound177861
def owner : Owner := ⟨.program ⟨257⟩, ⟨70487⟩⟩
def transferEvent : Nat := 177861
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177859 .coefficient, .predecessor 1 177860 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177859 .coefficient)
      LeftBound177856.bound (LeftBound177856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177860 .coefficient)
      LeftBound175003.bound (LeftBound175003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events683.exact175010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound175003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound175003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177856.bound, LeftBound175003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177856.bound, LeftBound175003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177856.actual selector witness, LeftBound175003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177861

namespace LeftBound177862
def owner : Owner := ⟨.program ⟨257⟩, ⟨70487⟩⟩
def transferEvent : Nat := 177862
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177858 .summary, .result 175010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177858 .summary)
      LeftBound177857.bound (LeftBound177857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70486⟩⟩) (rawTerms := some (Proof.Events694.exact177858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 175010 .summary)
      LeftBound175005.bound (LeftBound175005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39407⟩⟩) (rawTerms := some (Proof.Events683.exact175010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound175005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177857.bound, LeftBound175005.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177857.bound, LeftBound175005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177857.actual selector witness, LeftBound175005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177862

namespace LeftBound177866
def owner : Owner := ⟨.program ⟨257⟩, ⟨70488⟩⟩
def transferEvent : Nat := 177866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177864 .coefficient, .predecessor 1 177865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177864 .coefficient)
      LeftBound177861.bound (LeftBound177861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177865 .coefficient)
      LeftBound174791.bound (LeftBound174791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events682.exact174798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177861.bound, LeftBound174791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177861.bound, LeftBound174791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177861.actual selector witness, LeftBound174791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177866

namespace LeftBound177867
def owner : Owner := ⟨.program ⟨257⟩, ⟨70488⟩⟩
def transferEvent : Nat := 177867
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177863 .summary, .result 174798 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177863 .summary)
      LeftBound177862.bound (LeftBound177862.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70487⟩⟩) (rawTerms := some (Proof.Events694.exact177863RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174798 .summary)
      LeftBound174793.bound (LeftBound174793.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42087⟩⟩) (rawTerms := some (Proof.Events682.exact174798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound174793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177862.bound, LeftBound174793.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177862.bound, LeftBound174793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177862.actual selector witness, LeftBound174793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177867

namespace LeftBound177871
def owner : Owner := ⟨.program ⟨257⟩, ⟨70489⟩⟩
def transferEvent : Nat := 177871
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177869 .coefficient, .predecessor 1 177870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177869 .coefficient)
      LeftBound177866.bound (LeftBound177866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177870 .coefficient)
      LeftBound174579.bound (LeftBound174579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177866.bound, LeftBound174579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177866.bound, LeftBound174579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177866.actual selector witness, LeftBound174579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177871

namespace LeftBound177872
def owner : Owner := ⟨.program ⟨257⟩, ⟨70489⟩⟩
def transferEvent : Nat := 177872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177868 .summary, .result 174586 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177868 .summary)
      LeftBound177867.bound (LeftBound177867.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70488⟩⟩) (rawTerms := some (Proof.Events694.exact177868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174586 .summary)
      LeftBound174581.bound (LeftBound174581.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44767⟩⟩) (rawTerms := some (Proof.Events681.exact174586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound174581.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177867.bound, LeftBound174581.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177867.bound, LeftBound174581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177867.actual selector witness, LeftBound174581.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177872

namespace LeftBound177876
def owner : Owner := ⟨.program ⟨257⟩, ⟨70490⟩⟩
def transferEvent : Nat := 177876
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177874 .coefficient, .predecessor 1 177875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177874 .coefficient)
      LeftBound177871.bound (LeftBound177871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177875 .coefficient)
      LeftBound174367.bound (LeftBound174367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174367.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177871.bound, LeftBound174367.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177871.bound, LeftBound174367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177871.actual selector witness, LeftBound174367.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177876

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
