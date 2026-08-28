import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard579
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard582

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90949
def owner : Owner := ⟨.program ⟨257⟩, ⟨49528⟩⟩
def transferEvent : Nat := 90949
def frameStart : Nat := 90884
def rule : BoundRule := .product (.predecessor 0 90947 .coefficient) (.predecessor 1 90948 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90947 .coefficient)
      LeftAuthority90945.bound (LeftAuthority90945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90948 .coefficient)
      LeftBound90943.bound (LeftBound90943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority90945.bound LeftBound90943.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90945.bound, LeftBound90943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority90945.actual selector witness) * (LeftBound90943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90949

namespace LeftBound90957
def owner : Owner := ⟨.program ⟨257⟩, ⟨49529⟩⟩
def transferEvent : Nat := 90957
def frameStart : Nat := 90884
def rule : BoundRule := .sum [.predecessor 0 90955 .coefficient, .predecessor 1 90956 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90955 .coefficient)
      LeftAuthority90953.bound (LeftAuthority90953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90956 .coefficient)
      LeftBound90949.bound (LeftBound90949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90953.bound, LeftBound90949.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90953.bound, LeftBound90949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority90953.actual selector witness, LeftBound90949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90957

namespace LeftBound90961
def owner : Owner := ⟨.program ⟨257⟩, ⟨50155⟩⟩
def transferEvent : Nat := 90961
def frameStart : Nat := 90884
def rule : BoundRule := .product (.predecessor 0 90959 .coefficient) (.predecessor 1 90960 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90959 .coefficient)
      LeftBound90957.bound (LeftBound90957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90960 .coefficient)
      LeftAuthority90934.bound (LeftAuthority90934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90934.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90957.bound LeftAuthority90934.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90957.bound, LeftAuthority90934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90957.actual selector witness) * (LeftAuthority90934.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90961

namespace LeftBound90972
def owner : Owner := ⟨.program ⟨257⟩, ⟨48429⟩⟩
def transferEvent : Nat := 90972
def frameStart : Nat := 90884
def rule : BoundRule := .product (.predecessor 0 90970 .coefficient) (.predecessor 1 90971 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90970 .coefficient)
      LeftAuthority90945.bound (LeftAuthority90945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90971 .coefficient)
      LeftAuthority90968.bound (LeftAuthority90968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority90945.bound LeftAuthority90968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90945.bound, LeftAuthority90968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority90945.actual selector witness) * (LeftAuthority90968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90972

namespace LeftBound90980
def owner : Owner := ⟨.program ⟨257⟩, ⟨48430⟩⟩
def transferEvent : Nat := 90980
def frameStart : Nat := 90884
def rule : BoundRule := .sum [.predecessor 0 90978 .coefficient, .predecessor 1 90979 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90978 .coefficient)
      LeftAuthority90976.bound (LeftAuthority90976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90979 .coefficient)
      LeftBound90972.bound (LeftBound90972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90976.bound, LeftBound90972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90976.bound, LeftBound90972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority90976.actual selector witness, LeftBound90972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90980

namespace LeftBound90984
def owner : Owner := ⟨.program ⟨257⟩, ⟨50158⟩⟩
def transferEvent : Nat := 90984
def frameStart : Nat := 90884
def rule : BoundRule := .sum [.predecessor 0 90982 .coefficient, .predecessor 1 90983 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90982 .coefficient)
      LeftBound90980.bound (LeftBound90980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90983 .coefficient)
      LeftBound90961.bound (LeftBound90961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90980.bound, LeftBound90961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90980.bound, LeftBound90961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90980.actual selector witness, LeftBound90961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90984

namespace LeftBound90997
def owner : Owner := ⟨.program ⟨257⟩, ⟨50157⟩⟩
def transferEvent : Nat := 90997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90995 .coefficient, .predecessor 1 90996 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90995 .coefficient)
      LeftBound90826.bound (LeftBound90826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact90994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90996 .coefficient)
      LeftBound90809.bound (LeftBound90809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90809.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90826.bound, LeftBound90809.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90826.bound, LeftBound90809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90826.actual selector witness, LeftBound90809.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90997

namespace LeftBound91000
def owner : Owner := ⟨.program ⟨257⟩, ⟨50157⟩⟩
def transferEvent : Nat := 91000
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90994 .summary, .result 90816 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90994 .summary)
      LeftBound90828.bound (LeftBound90828.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48999⟩⟩) (rawTerms := some (Proof.Events355.exact90994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90816 .summary)
      LeftBound90811.bound (LeftBound90811.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50156⟩⟩) (rawTerms := some (Proof.Events354.exact90816RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90811.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90828.bound, LeftBound90811.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90828.bound, LeftBound90811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90828.actual selector witness, LeftBound90811.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91000

namespace LeftBound91024
def owner : Owner := ⟨.program ⟨257⟩, ⟨45277⟩⟩
def transferEvent : Nat := 91024
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 91022 .coefficient) (.predecessor 1 91023 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91022 .coefficient)
      LeftAuthority3856.bound (LeftAuthority3856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91023 .coefficient)
      LeftBound90526.bound (LeftBound90526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3856.bound LeftBound90526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3856.bound, LeftBound90526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3856.actual selector witness) * (LeftBound90526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91024

namespace LeftBound91029
def owner : Owner := ⟨.program ⟨257⟩, ⟨9918⟩⟩
def transferEvent : Nat := 91029
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91027 .coefficient) (.predecessor 1 91028 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91027 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91028 .coefficient)
      LeftBound17580.bound (LeftBound17580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftBound17580.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftBound17580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftBound17580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91029

namespace LeftBound91034
def owner : Owner := ⟨.program ⟨257⟩, ⟨45278⟩⟩
def transferEvent : Nat := 91034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91032 .coefficient, .predecessor 1 91033 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91032 .coefficient)
      LeftBound91029.bound (LeftBound91029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91033 .coefficient)
      LeftBound91024.bound (LeftBound91024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91029.bound, LeftBound91024.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91029.bound, LeftBound91024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91029.actual selector witness, LeftBound91024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91034

namespace LeftBound91038
def owner : Owner := ⟨.program ⟨257⟩, ⟨45279⟩⟩
def transferEvent : Nat := 91038
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91036 .coefficient, .predecessor 1 91037 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91036 .coefficient)
      LeftBound91034.bound (LeftBound91034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91037 .coefficient)
      LeftBound17572.bound (LeftBound17572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91034.bound, LeftBound17572.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91034.bound, LeftBound17572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91034.actual selector witness, LeftBound17572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91038

namespace LeftBound91039
def owner : Owner := ⟨.program ⟨257⟩, ⟨45279⟩⟩
def transferEvent : Nat := 91039
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨110⟩⟩]⟩ [⟨.result 17573 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17573 .coefficient)
      LeftBound17572.bound (LeftBound17572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨110⟩⟩) (rawTerms := some (Proof.Events068.exact17573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17572.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17572.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17572.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91039

namespace LeftBound91044
def owner : Owner := ⟨.program ⟨257⟩, ⟨45280⟩⟩
def transferEvent : Nat := 91044
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91042 .coefficient) (.predecessor 1 91043 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91042 .coefficient)
      LeftBound91038.bound (LeftBound91038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events355.exact91041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91043 .coefficient)
      LeftAuthority3859.bound (LeftAuthority3859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3859.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound91038.bound LeftAuthority3859.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91038.bound, LeftAuthority3859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound91038.actual selector witness) * (LeftAuthority3859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91044

namespace LeftBound91045
def owner : Owner := ⟨.program ⟨257⟩, ⟨45280⟩⟩
def transferEvent : Nat := 91045
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14856⟩⟩], []⟩ [⟨.result 3860 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3860 .coefficient)
      LeftAuthority3859.bound (LeftAuthority3859.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14856⟩⟩) (rawTerms := some (Proof.Events015.exact3860RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3859.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3859.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority3859.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91045

namespace LeftBound91046
def owner : Owner := ⟨.program ⟨257⟩, ⟨45280⟩⟩
def transferEvent : Nat := 91046
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91041 .summary) (.transfer 91045) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91041 .summary)
      LeftBound91039.bound (LeftBound91039.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45279⟩⟩) (rawTerms := some (Proof.Events355.exact91041RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 91045)
      LeftBound91045.bound (LeftBound91045.actual selector witness) := by
  exact .transfer (LeftBound91045.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound91039.bound LeftBound91045.bound
def bound : CoeffClass := .finite ⟨49414144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91039.bound, LeftBound91045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound91039.actual selector witness) * (LeftBound91045.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91046

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
