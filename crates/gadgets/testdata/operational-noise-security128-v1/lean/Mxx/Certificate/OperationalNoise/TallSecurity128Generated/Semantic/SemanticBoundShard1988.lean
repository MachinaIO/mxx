import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1987

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound293909
def owner : Owner := ⟨.program ⟨257⟩, ⟨32575⟩⟩
def transferEvent : Nat := 293909
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 293907 .coefficient) (.predecessor 1 293908 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293907 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293908 .coefficient)
      LeftBound293905.bound (LeftBound293905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact293906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293905.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound293905.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound293905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound293905.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293909

namespace LeftBound293910
def owner : Owner := ⟨.program ⟨257⟩, ⟨32575⟩⟩
def transferEvent : Nat := 293910
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32572⟩⟩]⟩ [⟨.result 293902 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293902 .coefficient)
      LeftAuthority293901.bound (LeftAuthority293901.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32572⟩⟩) (rawTerms := some (Proof.Events1148.exact293902RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293901.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority293901.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority293901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority293901.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound293910

namespace LeftBound293911
def owner : Owner := ⟨.program ⟨257⟩, ⟨32575⟩⟩
def transferEvent : Nat := 293911
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 293910) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 293910)
      LeftBound293910.bound (LeftBound293910.actual selector witness) := by
  exact .transfer (LeftBound293910.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound293910.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound293910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound293910.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293911

namespace LeftBound294006
def owner : Owner := ⟨.program ⟨257⟩, ⟨31781⟩⟩
def transferEvent : Nat := 294006
def frameStart : Nat := 293967
def rule : BoundRule := .identity (.predecessor 0 294005 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294005 .coefficient)
      LeftAuthority294003.bound (LeftAuthority294003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294003.derived selector witness)

def rawBound : CoeffClass := LeftAuthority294003.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority294003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority294003.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound294006

namespace LeftBound294023
def owner : Owner := ⟨.program ⟨257⟩, ⟨33282⟩⟩
def transferEvent : Nat := 294023
def frameStart : Nat := 293967
def rule : BoundRule := .sum [.predecessor 0 294021 .coefficient, .predecessor 1 294022 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294021 .coefficient)
      LeftBound294006.bound (LeftBound294006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound294006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294022 .coefficient)
      LeftAuthority294019.bound (LeftAuthority294019.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority294019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294006.bound, LeftAuthority294019.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294006.bound, LeftAuthority294019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294006.actual selector witness, LeftAuthority294019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294023

namespace LeftBound294026
def owner : Owner := ⟨.program ⟨257⟩, ⟨33283⟩⟩
def transferEvent : Nat := 294026
def frameStart : Nat := 293967
def rule : BoundRule := .identity (.predecessor 0 294025 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294025 .coefficient)
      LeftBound294023.bound (LeftBound294023.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound294023.derived selector witness)

def rawBound : CoeffClass := LeftBound294023.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound294023.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound294026

namespace LeftBound294032
def owner : Owner := ⟨.program ⟨257⟩, ⟨33284⟩⟩
def transferEvent : Nat := 294032
def frameStart : Nat := 293967
def rule : BoundRule := .product (.predecessor 0 294030 .coefficient) (.predecessor 1 294031 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294030 .coefficient)
      LeftAuthority294028.bound (LeftAuthority294028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294031 .coefficient)
      LeftBound294026.bound (LeftBound294026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294026.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority294028.bound LeftBound294026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority294028.bound, LeftBound294026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority294028.actual selector witness) * (LeftBound294026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound294032

namespace LeftBound294040
def owner : Owner := ⟨.program ⟨257⟩, ⟨33285⟩⟩
def transferEvent : Nat := 294040
def frameStart : Nat := 293967
def rule : BoundRule := .sum [.predecessor 0 294038 .coefficient, .predecessor 1 294039 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294038 .coefficient)
      LeftAuthority294036.bound (LeftAuthority294036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294039 .coefficient)
      LeftBound294032.bound (LeftBound294032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority294036.bound, LeftBound294032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority294036.bound, LeftBound294032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority294036.actual selector witness, LeftBound294032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294040

namespace LeftBound294044
def owner : Owner := ⟨.program ⟨257⟩, ⟨33700⟩⟩
def transferEvent : Nat := 294044
def frameStart : Nat := 293967
def rule : BoundRule := .product (.predecessor 0 294042 .coefficient) (.predecessor 1 294043 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294042 .coefficient)
      LeftBound294040.bound (LeftBound294040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294043 .coefficient)
      LeftAuthority294017.bound (LeftAuthority294017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294017.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound294040.bound LeftAuthority294017.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294040.bound, LeftAuthority294017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound294040.actual selector witness) * (LeftAuthority294017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound294044

namespace LeftBound294055
def owner : Owner := ⟨.program ⟨257⟩, ⟨31990⟩⟩
def transferEvent : Nat := 294055
def frameStart : Nat := 293967
def rule : BoundRule := .product (.predecessor 0 294053 .coefficient) (.predecessor 1 294054 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294053 .coefficient)
      LeftAuthority294028.bound (LeftAuthority294028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294054 .coefficient)
      LeftAuthority294051.bound (LeftAuthority294051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority294028.bound LeftAuthority294051.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority294028.bound, LeftAuthority294051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority294028.actual selector witness) * (LeftAuthority294051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound294055

namespace LeftBound294063
def owner : Owner := ⟨.program ⟨257⟩, ⟨31991⟩⟩
def transferEvent : Nat := 294063
def frameStart : Nat := 293967
def rule : BoundRule := .sum [.predecessor 0 294061 .coefficient, .predecessor 1 294062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294061 .coefficient)
      LeftAuthority294059.bound (LeftAuthority294059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294062 .coefficient)
      LeftBound294055.bound (LeftBound294055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294055.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority294059.bound, LeftBound294055.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority294059.bound, LeftBound294055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority294059.actual selector witness, LeftBound294055.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294063

namespace LeftBound294067
def owner : Owner := ⟨.program ⟨257⟩, ⟨33705⟩⟩
def transferEvent : Nat := 294067
def frameStart : Nat := 293967
def rule : BoundRule := .sum [.predecessor 0 294065 .coefficient, .predecessor 1 294066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294065 .coefficient)
      LeftBound294063.bound (LeftBound294063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294066 .coefficient)
      LeftBound294044.bound (LeftBound294044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294044.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294063.bound, LeftBound294044.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294063.bound, LeftBound294044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294063.actual selector witness, LeftBound294044.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294067

namespace LeftBound294080
def owner : Owner := ⟨.program ⟨257⟩, ⟨33702⟩⟩
def transferEvent : Nat := 294080
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294078 .coefficient, .predecessor 1 294079 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294078 .coefficient)
      LeftBound293909.bound (LeftBound293909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294079 .coefficient)
      LeftBound293892.bound (LeftBound293892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact293899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound293909.bound, LeftBound293892.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound293909.bound, LeftBound293892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound293909.actual selector witness, LeftBound293892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294080

namespace LeftBound294083
def owner : Owner := ⟨.program ⟨257⟩, ⟨33702⟩⟩
def transferEvent : Nat := 294083
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294077 .summary, .result 293899 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294077 .summary)
      LeftBound293911.bound (LeftBound293911.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32575⟩⟩) (rawTerms := some (Proof.Events1148.exact294077RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293899 .summary)
      LeftBound293894.bound (LeftBound293894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33701⟩⟩) (rawTerms := some (Proof.Events1148.exact293899RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293894.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound293911.bound, LeftBound293894.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound293911.bound, LeftBound293894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound293911.actual selector witness, LeftBound293894.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294083

namespace LeftBound294087
def owner : Owner := ⟨.program ⟨257⟩, ⟨33703⟩⟩
def transferEvent : Nat := 294087
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 294085 .coefficient) (.predecessor 1 294086 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294085 .coefficient)
      LeftBound294080.bound (LeftBound294080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1148.exact294084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294086 .coefficient)
      LeftBound15821.bound (LeftBound15821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound294080.bound LeftBound15821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294080.bound, LeftBound15821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound294080.actual selector witness) * (LeftBound15821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound294087

namespace LeftBound294088
def owner : Owner := ⟨.program ⟨257⟩, ⟨33703⟩⟩
def transferEvent : Nat := 294088
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩ [⟨.result 15818 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15818 .coefficient)
      LeftAuthority15817.bound (LeftAuthority15817.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7145⟩⟩) (rawTerms := some (Proof.Events061.exact15818RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15817.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15817.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15817.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound294088

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
