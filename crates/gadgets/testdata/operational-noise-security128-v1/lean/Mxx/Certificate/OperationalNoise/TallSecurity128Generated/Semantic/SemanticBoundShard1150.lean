import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1088

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound171938
def owner : Owner := ⟨.program ⟨257⟩, ⟨16331⟩⟩
def transferEvent : Nat := 171938
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 171936 .coefficient) (.value (.predecessor 1 171937 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171936 .coefficient)
      LeftAuthority171934.bound (LeftAuthority171934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact171935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171937 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority171934.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171934.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority171934.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound171938

namespace LeftBound171942
def owner : Owner := ⟨.program ⟨257⟩, ⟨16332⟩⟩
def transferEvent : Nat := 171942
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 171940 .coefficient) (.predecessor 1 171941 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171940 .coefficient)
      LeftBound163742.bound (LeftBound163742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171941 .coefficient)
      LeftBound171938.bound (LeftBound171938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact171939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171938.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163742.bound LeftBound171938.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163742.bound, LeftBound171938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163742.actual selector witness) * (LeftBound171938.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171942

namespace LeftBound171943
def owner : Owner := ⟨.program ⟨257⟩, ⟨16332⟩⟩
def transferEvent : Nat := 171943
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨16329⟩⟩]⟩ [⟨.result 171935 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171935 .coefficient)
      LeftAuthority171934.bound (LeftAuthority171934.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨16329⟩⟩) (rawTerms := some (Proof.Events671.exact171935RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171934.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority171934.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority171934.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound171943

namespace LeftBound171944
def owner : Owner := ⟨.program ⟨257⟩, ⟨16332⟩⟩
def transferEvent : Nat := 171944
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163745 .summary) (.transfer 171943) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163745 .summary)
      LeftBound163743.bound (LeftBound163743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6466⟩⟩) (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 171943)
      LeftBound171943.bound (LeftBound171943.actual selector witness) := by
  exact .transfer (LeftBound171943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163743.bound LeftBound171943.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163743.bound, LeftBound171943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163743.actual selector witness) * (LeftBound171943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171944

namespace LeftBound172023
def owner : Owner := ⟨.program ⟨257⟩, ⟨15571⟩⟩
def transferEvent : Nat := 172023
def frameStart : Nat := 171994
def rule : BoundRule := .product (.predecessor 0 172021 .coefficient) (.predecessor 1 172022 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172021 .coefficient)
      LeftAuthority172019.bound (LeftAuthority172019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact172020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172019.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172022 .coefficient)
      LeftAuthority172016.bound (LeftAuthority172016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact172017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority172019.bound LeftAuthority172016.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172019.bound, LeftAuthority172016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority172019.actual selector witness) * (LeftAuthority172016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172023

namespace LeftBound172027
def owner : Owner := ⟨.program ⟨257⟩, ⟨15572⟩⟩
def transferEvent : Nat := 172027
def frameStart : Nat := 171994
def rule : BoundRule := .identity (.predecessor 0 172026 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172026 .coefficient)
      LeftBound172023.bound (LeftBound172023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events671.exact172025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172023.derived selector witness)

def rawBound : CoeffClass := LeftBound172023.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound172023.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound172027

namespace LeftBound172044
def owner : Owner := ⟨.program ⟨257⟩, ⟨17142⟩⟩
def transferEvent : Nat := 172044
def frameStart : Nat := 171994
def rule : BoundRule := .sum [.predecessor 0 172042 .coefficient, .predecessor 1 172043 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172042 .coefficient)
      LeftBound172027.bound (LeftBound172027.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound172027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172043 .coefficient)
      LeftAuthority172040.bound (LeftAuthority172040.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority172040.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172027.bound, LeftAuthority172040.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172027.bound, LeftAuthority172040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172027.actual selector witness, LeftAuthority172040.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172044

namespace LeftBound172047
def owner : Owner := ⟨.program ⟨257⟩, ⟨17143⟩⟩
def transferEvent : Nat := 172047
def frameStart : Nat := 171994
def rule : BoundRule := .identity (.predecessor 0 172046 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172046 .coefficient)
      LeftBound172044.bound (LeftBound172044.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound172044.derived selector witness)

def rawBound : CoeffClass := LeftBound172044.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound172044.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound172047

namespace LeftBound172053
def owner : Owner := ⟨.program ⟨257⟩, ⟨17144⟩⟩
def transferEvent : Nat := 172053
def frameStart : Nat := 171994
def rule : BoundRule := .product (.predecessor 0 172051 .coefficient) (.predecessor 1 172052 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172051 .coefficient)
      LeftAuthority172049.bound (LeftAuthority172049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172052 .coefficient)
      LeftBound172047.bound (LeftBound172047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority172049.bound LeftBound172047.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172049.bound, LeftBound172047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority172049.actual selector witness) * (LeftBound172047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172053

namespace LeftBound172069
def owner : Owner := ⟨.program ⟨257⟩, ⟨9569⟩⟩
def transferEvent : Nat := 172069
def frameStart : Nat := 171994
def rule : BoundRule := .scale (.predecessor 0 172067 .coefficient) (.value (.predecessor 1 172068 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172067 .coefficient)
      LeftAuthority172065.bound (LeftAuthority172065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172068 .coefficient)
      LeftAuthority172056.bound (LeftAuthority172056.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority172056.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority172065.bound LeftAuthority172056.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172065.bound, LeftAuthority172056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority172065.actual selector witness) * (LeftAuthority172056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound172069

namespace LeftBound172072
def owner : Owner := ⟨.program ⟨257⟩, ⟨7303⟩⟩
def transferEvent : Nat := 172072
def frameStart : Nat := 171994
def rule : BoundRule := .identity (.predecessor 0 172071 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172071 .coefficient)
      LeftAuthority172059.bound (LeftAuthority172059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172059.derived selector witness)

def rawBound : CoeffClass := LeftAuthority172059.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority172059.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound172072

namespace LeftBound172076
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def transferEvent : Nat := 172076
def frameStart : Nat := 171994
def rule : BoundRule := .product (.predecessor 0 172074 .coefficient) (.predecessor 1 172075 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172074 .coefficient)
      LeftBound172072.bound (LeftBound172072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172075 .coefficient)
      LeftBound172069.bound (LeftBound172069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172069.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound172072.bound LeftBound172069.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172072.bound, LeftBound172069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound172072.actual selector witness) * (LeftBound172069.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172076

namespace LeftBound172081
def owner : Owner := ⟨.program ⟨257⟩, ⟨17145⟩⟩
def transferEvent : Nat := 172081
def frameStart : Nat := 171994
def rule : BoundRule := .sum [.predecessor 0 172079 .coefficient, .predecessor 1 172080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172079 .coefficient)
      LeftBound172076.bound (LeftBound172076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172080 .coefficient)
      LeftBound172053.bound (LeftBound172053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172076.bound, LeftBound172053.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172076.bound, LeftBound172053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172076.actual selector witness, LeftBound172053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172081

namespace LeftBound172085
def owner : Owner := ⟨.program ⟨257⟩, ⟨17406⟩⟩
def transferEvent : Nat := 172085
def frameStart : Nat := 171994
def rule : BoundRule := .product (.predecessor 0 172083 .coefficient) (.predecessor 1 172084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172083 .coefficient)
      LeftBound172081.bound (LeftBound172081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172084 .coefficient)
      LeftAuthority172038.bound (LeftAuthority172038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound172081.bound LeftAuthority172038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172081.bound, LeftAuthority172038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound172081.actual selector witness) * (LeftAuthority172038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172085

namespace LeftBound172096
def owner : Owner := ⟨.program ⟨257⟩, ⟨15822⟩⟩
def transferEvent : Nat := 172096
def frameStart : Nat := 171994
def rule : BoundRule := .product (.predecessor 0 172094 .coefficient) (.predecessor 1 172095 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172094 .coefficient)
      LeftAuthority172049.bound (LeftAuthority172049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172095 .coefficient)
      LeftAuthority172092.bound (LeftAuthority172092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority172049.bound LeftAuthority172092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172049.bound, LeftAuthority172092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority172049.actual selector witness) * (LeftAuthority172092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound172096

namespace LeftBound172104
def owner : Owner := ⟨.program ⟨257⟩, ⟨15823⟩⟩
def transferEvent : Nat := 172104
def frameStart : Nat := 171994
def rule : BoundRule := .sum [.predecessor 0 172102 .coefficient, .predecessor 1 172103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172102 .coefficient)
      LeftAuthority172100.bound (LeftAuthority172100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority172100.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority172100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172103 .coefficient)
      LeftBound172096.bound (LeftBound172096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events672.exact172098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority172100.bound, LeftBound172096.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority172100.bound, LeftBound172096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority172100.actual selector witness, LeftBound172096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172104

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
