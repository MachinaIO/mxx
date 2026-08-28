import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1439

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound213891
def owner : Owner := ⟨.program ⟨257⟩, ⟨51452⟩⟩
def transferEvent : Nat := 213891
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 213890) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 213890)
      LeftBound213890.bound (LeftBound213890.actual selector witness) := by
  exact .transfer (LeftBound213890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound213890.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound213890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound213890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213891

namespace LeftBound213970
def owner : Owner := ⟨.program ⟨257⟩, ⟨50546⟩⟩
def transferEvent : Nat := 213970
def frameStart : Nat := 213941
def rule : BoundRule := .product (.predecessor 0 213968 .coefficient) (.predecessor 1 213969 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213968 .coefficient)
      LeftAuthority213966.bound (LeftAuthority213966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213969 .coefficient)
      LeftAuthority213963.bound (LeftAuthority213963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority213966.bound LeftAuthority213963.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213966.bound, LeftAuthority213963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority213966.actual selector witness) * (LeftAuthority213963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213970

namespace LeftBound213974
def owner : Owner := ⟨.program ⟨257⟩, ⟨50547⟩⟩
def transferEvent : Nat := 213974
def frameStart : Nat := 213941
def rule : BoundRule := .identity (.predecessor 0 213973 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213973 .coefficient)
      LeftBound213970.bound (LeftBound213970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213970.derived selector witness)

def rawBound : CoeffClass := LeftBound213970.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound213970.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound213974

namespace LeftBound213991
def owner : Owner := ⟨.program ⟨257⟩, ⟨52286⟩⟩
def transferEvent : Nat := 213991
def frameStart : Nat := 213941
def rule : BoundRule := .sum [.predecessor 0 213989 .coefficient, .predecessor 1 213990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213989 .coefficient)
      LeftBound213974.bound (LeftBound213974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound213974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213990 .coefficient)
      LeftAuthority213987.bound (LeftAuthority213987.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority213987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213974.bound, LeftAuthority213987.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213974.bound, LeftAuthority213987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213974.actual selector witness, LeftAuthority213987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213991

namespace LeftBound213994
def owner : Owner := ⟨.program ⟨257⟩, ⟨52287⟩⟩
def transferEvent : Nat := 213994
def frameStart : Nat := 213941
def rule : BoundRule := .identity (.predecessor 0 213993 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213993 .coefficient)
      LeftBound213991.bound (LeftBound213991.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound213991.derived selector witness)

def rawBound : CoeffClass := LeftBound213991.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound213991.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound213994

namespace LeftBound214000
def owner : Owner := ⟨.program ⟨257⟩, ⟨52288⟩⟩
def transferEvent : Nat := 214000
def frameStart : Nat := 213941
def rule : BoundRule := .product (.predecessor 0 213998 .coefficient) (.predecessor 1 213999 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213998 .coefficient)
      LeftAuthority213996.bound (LeftAuthority213996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213999 .coefficient)
      LeftBound213994.bound (LeftBound213994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213994.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority213996.bound LeftBound213994.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213996.bound, LeftBound213994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority213996.actual selector witness) * (LeftBound213994.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound214000

namespace LeftBound214016
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 214016
def frameStart : Nat := 213941
def rule : BoundRule := .scale (.predecessor 0 214014 .coefficient) (.value (.predecessor 1 214015 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214014 .coefficient)
      LeftAuthority214012.bound (LeftAuthority214012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact214013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214015 .coefficient)
      LeftAuthority214003.bound (LeftAuthority214003.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority214003.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority214012.bound LeftAuthority214003.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority214012.bound, LeftAuthority214003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority214012.actual selector witness) * (LeftAuthority214003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound214016

namespace LeftBound214019
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 214019
def frameStart : Nat := 213941
def rule : BoundRule := .identity (.predecessor 0 214018 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214018 .coefficient)
      LeftAuthority214006.bound (LeftAuthority214006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact214007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214006.derived selector witness)

def rawBound : CoeffClass := LeftAuthority214006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority214006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority214006.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound214019

namespace LeftBound214023
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 214023
def frameStart : Nat := 213941
def rule : BoundRule := .product (.predecessor 0 214021 .coefficient) (.predecessor 1 214022 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214021 .coefficient)
      LeftBound214019.bound (LeftBound214019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214022 .coefficient)
      LeftBound214016.bound (LeftBound214016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound214019.bound LeftBound214016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound214019.bound, LeftBound214016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound214019.actual selector witness) * (LeftBound214016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound214023

namespace LeftBound214028
def owner : Owner := ⟨.program ⟨257⟩, ⟨52289⟩⟩
def transferEvent : Nat := 214028
def frameStart : Nat := 213941
def rule : BoundRule := .sum [.predecessor 0 214026 .coefficient, .predecessor 1 214027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214026 .coefficient)
      LeftBound214023.bound (LeftBound214023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214027 .coefficient)
      LeftBound214000.bound (LeftBound214000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact214002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound214023.bound, LeftBound214000.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound214023.bound, LeftBound214000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound214023.actual selector witness, LeftBound214000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound214028

namespace LeftBound214032
def owner : Owner := ⟨.program ⟨257⟩, ⟨52522⟩⟩
def transferEvent : Nat := 214032
def frameStart : Nat := 213941
def rule : BoundRule := .product (.predecessor 0 214030 .coefficient) (.predecessor 1 214031 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214030 .coefficient)
      LeftBound214028.bound (LeftBound214028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214031 .coefficient)
      LeftAuthority213985.bound (LeftAuthority213985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213985.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213985.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound214028.bound LeftAuthority213985.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound214028.bound, LeftAuthority213985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound214028.actual selector witness) * (LeftAuthority213985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound214032

namespace LeftBound214043
def owner : Owner := ⟨.program ⟨257⟩, ⟨50890⟩⟩
def transferEvent : Nat := 214043
def frameStart : Nat := 213941
def rule : BoundRule := .product (.predecessor 0 214041 .coefficient) (.predecessor 1 214042 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214041 .coefficient)
      LeftAuthority213996.bound (LeftAuthority213996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214042 .coefficient)
      LeftAuthority214039.bound (LeftAuthority214039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority213996.bound LeftAuthority214039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213996.bound, LeftAuthority214039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority213996.actual selector witness) * (LeftAuthority214039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound214043

namespace LeftBound214051
def owner : Owner := ⟨.program ⟨257⟩, ⟨50891⟩⟩
def transferEvent : Nat := 214051
def frameStart : Nat := 213941
def rule : BoundRule := .sum [.predecessor 0 214049 .coefficient, .predecessor 1 214050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214049 .coefficient)
      LeftAuthority214047.bound (LeftAuthority214047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214050 .coefficient)
      LeftBound214043.bound (LeftBound214043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority214047.bound, LeftBound214043.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority214047.bound, LeftBound214043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority214047.actual selector witness, LeftBound214043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound214051

namespace LeftBound214055
def owner : Owner := ⟨.program ⟨257⟩, ⟨52523⟩⟩
def transferEvent : Nat := 214055
def frameStart : Nat := 213941
def rule : BoundRule := .sum [.predecessor 0 214053 .coefficient, .predecessor 1 214054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214053 .coefficient)
      LeftBound214051.bound (LeftBound214051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214054 .coefficient)
      LeftBound214032.bound (LeftBound214032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound214051.bound, LeftBound214032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound214051.bound, LeftBound214032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound214051.actual selector witness, LeftBound214032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound214055

namespace LeftBound214068
def owner : Owner := ⟨.program ⟨257⟩, ⟨52521⟩⟩
def transferEvent : Nat := 214068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 214066 .coefficient, .predecessor 1 214067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 214066 .coefficient)
      LeftBound213889.bound (LeftBound213889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events836.exact214065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 214067 .coefficient)
      LeftBound213872.bound (LeftBound213872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213889.bound, LeftBound213872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213889.bound, LeftBound213872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213889.actual selector witness, LeftBound213872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound214068

namespace LeftBound214071
def owner : Owner := ⟨.program ⟨257⟩, ⟨52521⟩⟩
def transferEvent : Nat := 214071
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 214065 .summary, .result 213879 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 214065 .summary)
      LeftBound213891.bound (LeftBound213891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51452⟩⟩) (rawTerms := some (Proof.Events836.exact214065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213879 .summary)
      LeftBound213874.bound (LeftBound213874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52520⟩⟩) (rawTerms := some (Proof.Events835.exact213879RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213891.bound, LeftBound213874.bound]
def bound : CoeffClass := .finite ⟨2997889464187086962688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213891.bound, LeftBound213874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213891.actual selector witness, LeftBound213874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound214071

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
