import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1364

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound203864
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def transferEvent : Nat := 203864
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨40892⟩⟩]⟩ [⟨.result 203856 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 203856 .coefficient)
      LeftAuthority203855.bound (LeftAuthority203855.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨40892⟩⟩) (rawTerms := some (Proof.Events796.exact203856RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203855.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority203855.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority203855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority203855.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound203864

namespace LeftBound203865
def owner : Owner := ⟨.program ⟨257⟩, ⟨40895⟩⟩
def transferEvent : Nat := 203865
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 203864) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 203864)
      LeftBound203864.bound (LeftBound203864.actual selector witness) := by
  exact .transfer (LeftBound203864.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound203864.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound203864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound203864.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound203865

namespace LeftBound203960
def owner : Owner := ⟨.program ⟨257⟩, ⟨40125⟩⟩
def transferEvent : Nat := 203960
def frameStart : Nat := 203921
def rule : BoundRule := .identity (.predecessor 0 203959 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203959 .coefficient)
      LeftAuthority203957.bound (LeftAuthority203957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203957.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203957.derived selector witness)

def rawBound : CoeffClass := LeftAuthority203957.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority203957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority203957.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound203960

namespace LeftBound203977
def owner : Owner := ⟨.program ⟨257⟩, ⟨41474⟩⟩
def transferEvent : Nat := 203977
def frameStart : Nat := 203921
def rule : BoundRule := .sum [.predecessor 0 203975 .coefficient, .predecessor 1 203976 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203975 .coefficient)
      LeftBound203960.bound (LeftBound203960.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound203960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 203976 .coefficient)
      LeftAuthority203973.bound (LeftAuthority203973.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority203973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound203960.bound, LeftAuthority203973.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound203960.bound, LeftAuthority203973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound203960.actual selector witness, LeftAuthority203973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound203977

namespace LeftBound203980
def owner : Owner := ⟨.program ⟨257⟩, ⟨41475⟩⟩
def transferEvent : Nat := 203980
def frameStart : Nat := 203921
def rule : BoundRule := .identity (.predecessor 0 203979 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203979 .coefficient)
      LeftBound203977.bound (LeftBound203977.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound203977.derived selector witness)

def rawBound : CoeffClass := LeftBound203977.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound203977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound203977.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound203980

namespace LeftBound203986
def owner : Owner := ⟨.program ⟨257⟩, ⟨41476⟩⟩
def transferEvent : Nat := 203986
def frameStart : Nat := 203921
def rule : BoundRule := .product (.predecessor 0 203984 .coefficient) (.predecessor 1 203985 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203984 .coefficient)
      LeftAuthority203982.bound (LeftAuthority203982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 203985 .coefficient)
      LeftBound203980.bound (LeftBound203980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority203982.bound LeftBound203980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority203982.bound, LeftBound203980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority203982.actual selector witness) * (LeftBound203980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound203986

namespace LeftBound203994
def owner : Owner := ⟨.program ⟨257⟩, ⟨41477⟩⟩
def transferEvent : Nat := 203994
def frameStart : Nat := 203921
def rule : BoundRule := .sum [.predecessor 0 203992 .coefficient, .predecessor 1 203993 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203992 .coefficient)
      LeftAuthority203990.bound (LeftAuthority203990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 203993 .coefficient)
      LeftBound203986.bound (LeftBound203986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203986.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority203990.bound, LeftBound203986.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority203990.bound, LeftBound203986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority203990.actual selector witness, LeftBound203986.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound203994

namespace LeftBound203998
def owner : Owner := ⟨.program ⟨257⟩, ⟨42034⟩⟩
def transferEvent : Nat := 203998
def frameStart : Nat := 203921
def rule : BoundRule := .product (.predecessor 0 203996 .coefficient) (.predecessor 1 203997 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 203996 .coefficient)
      LeftBound203994.bound (LeftBound203994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 203997 .coefficient)
      LeftAuthority203971.bound (LeftAuthority203971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203971.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound203994.bound LeftAuthority203971.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound203994.bound, LeftAuthority203971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound203994.actual selector witness) * (LeftAuthority203971.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound203998

namespace LeftBound204009
def owner : Owner := ⟨.program ⟨257⟩, ⟨40350⟩⟩
def transferEvent : Nat := 204009
def frameStart : Nat := 203921
def rule : BoundRule := .product (.predecessor 0 204007 .coefficient) (.predecessor 1 204008 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204007 .coefficient)
      LeftAuthority203982.bound (LeftAuthority203982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority203982.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority203982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204008 .coefficient)
      LeftAuthority204005.bound (LeftAuthority204005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204005.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority203982.bound LeftAuthority204005.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority203982.bound, LeftAuthority204005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority203982.actual selector witness) * (LeftAuthority204005.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204009

namespace LeftBound204017
def owner : Owner := ⟨.program ⟨257⟩, ⟨40351⟩⟩
def transferEvent : Nat := 204017
def frameStart : Nat := 203921
def rule : BoundRule := .sum [.predecessor 0 204015 .coefficient, .predecessor 1 204016 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204015 .coefficient)
      LeftAuthority204013.bound (LeftAuthority204013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204013.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204013.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204016 .coefficient)
      LeftBound204009.bound (LeftBound204009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority204013.bound, LeftBound204009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204013.bound, LeftBound204009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority204013.actual selector witness, LeftBound204009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204017

namespace LeftBound204021
def owner : Owner := ⟨.program ⟨257⟩, ⟨42038⟩⟩
def transferEvent : Nat := 204021
def frameStart : Nat := 203921
def rule : BoundRule := .sum [.predecessor 0 204019 .coefficient, .predecessor 1 204020 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204019 .coefficient)
      LeftBound204017.bound (LeftBound204017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204017.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204020 .coefficient)
      LeftBound203998.bound (LeftBound203998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound204017.bound, LeftBound203998.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204017.bound, LeftBound203998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound204017.actual selector witness, LeftBound203998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204021

namespace LeftBound204034
def owner : Owner := ⟨.program ⟨257⟩, ⟨42036⟩⟩
def transferEvent : Nat := 204034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 204032 .coefficient, .predecessor 1 204033 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204032 .coefficient)
      LeftBound203863.bound (LeftBound203863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact204031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204033 .coefficient)
      LeftBound203846.bound (LeftBound203846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound203863.bound, LeftBound203846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound203863.bound, LeftBound203846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound203863.actual selector witness, LeftBound203846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204034

namespace LeftBound204037
def owner : Owner := ⟨.program ⟨257⟩, ⟨42036⟩⟩
def transferEvent : Nat := 204037
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 204031 .summary, .result 203853 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204031 .summary)
      LeftBound203865.bound (LeftBound203865.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40895⟩⟩) (rawTerms := some (Proof.Events796.exact204031RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound203865.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 203853 .summary)
      LeftBound203848.bound (LeftBound203848.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42035⟩⟩) (rawTerms := some (Proof.Events796.exact203853RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound203848.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound203865.bound, LeftBound203848.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound203865.bound, LeftBound203848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound203865.actual selector witness, LeftBound203848.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204037

namespace LeftBound204041
def owner : Owner := ⟨.program ⟨257⟩, ⟨42037⟩⟩
def transferEvent : Nat := 204041
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 204039 .coefficient) (.predecessor 1 204040 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204039 .coefficient)
      LeftBound204034.bound (LeftBound204034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204040 .coefficient)
      LeftBound15601.bound (LeftBound15601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound204034.bound LeftBound15601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204034.bound, LeftBound15601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound204034.actual selector witness) * (LeftBound15601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204041

namespace LeftBound204042
def owner : Owner := ⟨.program ⟨257⟩, ⟨42037⟩⟩
def transferEvent : Nat := 204042
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩ [⟨.result 15598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15598 .coefficient)
      LeftAuthority15597.bound (LeftAuthority15597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7159⟩⟩) (rawTerms := some (Proof.Events060.exact15598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound204042

namespace LeftBound204043
def owner : Owner := ⟨.program ⟨257⟩, ⟨42037⟩⟩
def transferEvent : Nat := 204043
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 204038 .summary) (.transfer 204042) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204038 .summary)
      LeftBound204037.bound (LeftBound204037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42036⟩⟩) (rawTerms := some (Proof.Events797.exact204038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 204042)
      LeftBound204042.bound (LeftBound204042.actual selector witness) := by
  exact .transfer (LeftBound204042.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound204037.bound LeftBound204042.bound
def bound : CoeffClass := .finite ⟨345671091840339265080175045977281837137920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204037.bound, LeftBound204042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound204037.actual selector witness) * (LeftBound204042.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204043

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
