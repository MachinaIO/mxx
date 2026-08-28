import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard537

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83791
def owner : Owner := ⟨.program ⟨257⟩, ⟨18419⟩⟩
def transferEvent : Nat := 83791
def frameStart : Nat := 83762
def rule : BoundRule := .product (.predecessor 0 83789 .coefficient) (.predecessor 1 83790 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83789 .coefficient)
      LeftAuthority83787.bound (LeftAuthority83787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83790 .coefficient)
      LeftAuthority83784.bound (LeftAuthority83784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83784.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83784.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83787.bound LeftAuthority83784.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83787.bound, LeftAuthority83784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority83787.actual selector witness) * (LeftAuthority83784.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83791

namespace LeftBound83795
def owner : Owner := ⟨.program ⟨257⟩, ⟨18420⟩⟩
def transferEvent : Nat := 83795
def frameStart : Nat := 83762
def rule : BoundRule := .identity (.predecessor 0 83794 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83794 .coefficient)
      LeftBound83791.bound (LeftBound83791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83791.derived selector witness)

def rawBound : CoeffClass := LeftBound83791.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound83791.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83795

namespace LeftBound83812
def owner : Owner := ⟨.program ⟨257⟩, ⟨20010⟩⟩
def transferEvent : Nat := 83812
def frameStart : Nat := 83762
def rule : BoundRule := .sum [.predecessor 0 83810 .coefficient, .predecessor 1 83811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83810 .coefficient)
      LeftBound83795.bound (LeftBound83795.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83811 .coefficient)
      LeftAuthority83808.bound (LeftAuthority83808.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83795.bound, LeftAuthority83808.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83795.bound, LeftAuthority83808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83795.actual selector witness, LeftAuthority83808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83812

namespace LeftBound83815
def owner : Owner := ⟨.program ⟨257⟩, ⟨20011⟩⟩
def transferEvent : Nat := 83815
def frameStart : Nat := 83762
def rule : BoundRule := .identity (.predecessor 0 83814 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83814 .coefficient)
      LeftBound83812.bound (LeftBound83812.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83812.derived selector witness)

def rawBound : CoeffClass := LeftBound83812.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound83812.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83815

namespace LeftBound83821
def owner : Owner := ⟨.program ⟨257⟩, ⟨20012⟩⟩
def transferEvent : Nat := 83821
def frameStart : Nat := 83762
def rule : BoundRule := .product (.predecessor 0 83819 .coefficient) (.predecessor 1 83820 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83819 .coefficient)
      LeftAuthority83817.bound (LeftAuthority83817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83820 .coefficient)
      LeftBound83815.bound (LeftBound83815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83815.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority83817.bound LeftBound83815.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83817.bound, LeftBound83815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority83817.actual selector witness) * (LeftBound83815.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83821

namespace LeftBound83837
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 83837
def frameStart : Nat := 83762
def rule : BoundRule := .scale (.predecessor 0 83835 .coefficient) (.value (.predecessor 1 83836 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83835 .coefficient)
      LeftAuthority83833.bound (LeftAuthority83833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83833.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83836 .coefficient)
      LeftAuthority83824.bound (LeftAuthority83824.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83824.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83833.bound LeftAuthority83824.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83833.bound, LeftAuthority83824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority83833.actual selector witness) * (LeftAuthority83824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83837

namespace LeftBound83840
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 83840
def frameStart : Nat := 83762
def rule : BoundRule := .identity (.predecessor 0 83839 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83839 .coefficient)
      LeftAuthority83827.bound (LeftAuthority83827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83827.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83827.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83827.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority83827.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83840

namespace LeftBound83844
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 83844
def frameStart : Nat := 83762
def rule : BoundRule := .product (.predecessor 0 83842 .coefficient) (.predecessor 1 83843 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83842 .coefficient)
      LeftBound83840.bound (LeftBound83840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83843 .coefficient)
      LeftBound83837.bound (LeftBound83837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83837.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound83840.bound LeftBound83837.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83840.bound, LeftBound83837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound83840.actual selector witness) * (LeftBound83837.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83844

namespace LeftBound83849
def owner : Owner := ⟨.program ⟨257⟩, ⟨20013⟩⟩
def transferEvent : Nat := 83849
def frameStart : Nat := 83762
def rule : BoundRule := .sum [.predecessor 0 83847 .coefficient, .predecessor 1 83848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83847 .coefficient)
      LeftBound83844.bound (LeftBound83844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83848 .coefficient)
      LeftBound83821.bound (LeftBound83821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83844.bound, LeftBound83821.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83844.bound, LeftBound83821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83844.actual selector witness, LeftBound83821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83849

namespace LeftBound83853
def owner : Owner := ⟨.program ⟨257⟩, ⟨20288⟩⟩
def transferEvent : Nat := 83853
def frameStart : Nat := 83762
def rule : BoundRule := .product (.predecessor 0 83851 .coefficient) (.predecessor 1 83852 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83851 .coefficient)
      LeftBound83849.bound (LeftBound83849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83852 .coefficient)
      LeftAuthority83806.bound (LeftAuthority83806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83806.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound83849.bound LeftAuthority83806.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83849.bound, LeftAuthority83806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound83849.actual selector witness) * (LeftAuthority83806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83853

namespace LeftBound83864
def owner : Owner := ⟨.program ⟨257⟩, ⟨18638⟩⟩
def transferEvent : Nat := 83864
def frameStart : Nat := 83762
def rule : BoundRule := .product (.predecessor 0 83862 .coefficient) (.predecessor 1 83863 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83862 .coefficient)
      LeftAuthority83817.bound (LeftAuthority83817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83863 .coefficient)
      LeftAuthority83860.bound (LeftAuthority83860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83817.bound LeftAuthority83860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83817.bound, LeftAuthority83860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority83817.actual selector witness) * (LeftAuthority83860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83864

namespace LeftBound83872
def owner : Owner := ⟨.program ⟨257⟩, ⟨18639⟩⟩
def transferEvent : Nat := 83872
def frameStart : Nat := 83762
def rule : BoundRule := .sum [.predecessor 0 83870 .coefficient, .predecessor 1 83871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83870 .coefficient)
      LeftAuthority83868.bound (LeftAuthority83868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83871 .coefficient)
      LeftBound83864.bound (LeftBound83864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83868.bound, LeftBound83864.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83868.bound, LeftBound83864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority83868.actual selector witness, LeftBound83864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83872

namespace LeftBound83876
def owner : Owner := ⟨.program ⟨257⟩, ⟨20289⟩⟩
def transferEvent : Nat := 83876
def frameStart : Nat := 83762
def rule : BoundRule := .sum [.predecessor 0 83874 .coefficient, .predecessor 1 83875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83874 .coefficient)
      LeftBound83872.bound (LeftBound83872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83875 .coefficient)
      LeftBound83853.bound (LeftBound83853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83872.bound, LeftBound83853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83872.bound, LeftBound83853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83872.actual selector witness, LeftBound83853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83876

namespace LeftBound83889
def owner : Owner := ⟨.program ⟨257⟩, ⟨20287⟩⟩
def transferEvent : Nat := 83889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83887 .coefficient, .predecessor 1 83888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83887 .coefficient)
      LeftBound83710.bound (LeftBound83710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83888 .coefficient)
      LeftBound83693.bound (LeftBound83693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83710.bound, LeftBound83693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83710.bound, LeftBound83693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83710.actual selector witness, LeftBound83693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83889

namespace LeftBound83892
def owner : Owner := ⟨.program ⟨257⟩, ⟨20287⟩⟩
def transferEvent : Nat := 83892
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83886 .summary, .result 83700 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83886 .summary)
      LeftBound83712.bound (LeftBound83712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19212⟩⟩) (rawTerms := some (Proof.Events327.exact83886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83700 .summary)
      LeftBound83695.bound (LeftBound83695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20286⟩⟩) (rawTerms := some (Proof.Events326.exact83700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83712.bound, LeftBound83695.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83712.bound, LeftBound83695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound83712.actual selector witness, LeftBound83695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83892

namespace LeftBound83896
def owner : Owner := ⟨.program ⟨257⟩, ⟨20840⟩⟩
def transferEvent : Nat := 83896
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83894 .coefficient) (.predecessor 1 83895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 83894 .coefficient)
      LeftBound83889.bound (LeftBound83889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 83895 .coefficient)
      LeftAuthority83615.bound (LeftAuthority83615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83615.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound83889.bound LeftAuthority83615.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83889.bound, LeftAuthority83615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound83889.actual selector witness) * (LeftAuthority83615.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83896

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
