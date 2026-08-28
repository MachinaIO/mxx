import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard296
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard351

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57827
def owner : Owner := ⟨.program ⟨257⟩, ⟨38335⟩⟩
def transferEvent : Nat := 57827
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 57826) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 57826)
      LeftBound57826.bound (LeftBound57826.actual selector witness) := by
  exact .transfer (LeftBound57826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound57826.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound57826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound57826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57827

namespace LeftBound57922
def owner : Owner := ⟨.program ⟨257⟩, ⟨37493⟩⟩
def transferEvent : Nat := 57922
def frameStart : Nat := 57883
def rule : BoundRule := .identity (.predecessor 0 57921 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57921 .coefficient)
      LeftAuthority57919.bound (LeftAuthority57919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57919.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57919.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority57919.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57922

namespace LeftBound57939
def owner : Owner := ⟨.program ⟨257⟩, ⟨38818⟩⟩
def transferEvent : Nat := 57939
def frameStart : Nat := 57883
def rule : BoundRule := .sum [.predecessor 0 57937 .coefficient, .predecessor 1 57938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57937 .coefficient)
      LeftBound57922.bound (LeftBound57922.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57938 .coefficient)
      LeftAuthority57935.bound (LeftAuthority57935.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57922.bound, LeftAuthority57935.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57922.bound, LeftAuthority57935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57922.actual selector witness, LeftAuthority57935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57939

namespace LeftBound57942
def owner : Owner := ⟨.program ⟨257⟩, ⟨38819⟩⟩
def transferEvent : Nat := 57942
def frameStart : Nat := 57883
def rule : BoundRule := .identity (.predecessor 0 57941 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57941 .coefficient)
      LeftBound57939.bound (LeftBound57939.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57939.derived selector witness)

def rawBound : CoeffClass := LeftBound57939.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound57939.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57942

namespace LeftBound57948
def owner : Owner := ⟨.program ⟨257⟩, ⟨38820⟩⟩
def transferEvent : Nat := 57948
def frameStart : Nat := 57883
def rule : BoundRule := .product (.predecessor 0 57946 .coefficient) (.predecessor 1 57947 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57946 .coefficient)
      LeftAuthority57944.bound (LeftAuthority57944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57947 .coefficient)
      LeftBound57942.bound (LeftBound57942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57942.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority57944.bound LeftBound57942.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57944.bound, LeftBound57942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority57944.actual selector witness) * (LeftBound57942.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57948

namespace LeftBound57956
def owner : Owner := ⟨.program ⟨257⟩, ⟨38821⟩⟩
def transferEvent : Nat := 57956
def frameStart : Nat := 57883
def rule : BoundRule := .sum [.predecessor 0 57954 .coefficient, .predecessor 1 57955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57954 .coefficient)
      LeftAuthority57952.bound (LeftAuthority57952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57955 .coefficient)
      LeftBound57948.bound (LeftBound57948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57948.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57952.bound, LeftBound57948.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57952.bound, LeftBound57948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority57952.actual selector witness, LeftBound57948.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57956

namespace LeftBound57960
def owner : Owner := ⟨.program ⟨257⟩, ⟨39504⟩⟩
def transferEvent : Nat := 57960
def frameStart : Nat := 57883
def rule : BoundRule := .product (.predecessor 0 57958 .coefficient) (.predecessor 1 57959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57958 .coefficient)
      LeftBound57956.bound (LeftBound57956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57959 .coefficient)
      LeftAuthority57933.bound (LeftAuthority57933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57956.bound LeftAuthority57933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57956.bound, LeftAuthority57933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57956.actual selector witness) * (LeftAuthority57933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57960

namespace LeftBound57971
def owner : Owner := ⟨.program ⟨257⟩, ⟨37745⟩⟩
def transferEvent : Nat := 57971
def frameStart : Nat := 57883
def rule : BoundRule := .product (.predecessor 0 57969 .coefficient) (.predecessor 1 57970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57969 .coefficient)
      LeftAuthority57944.bound (LeftAuthority57944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57970 .coefficient)
      LeftAuthority57967.bound (LeftAuthority57967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57944.bound LeftAuthority57967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57944.bound, LeftAuthority57967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority57944.actual selector witness) * (LeftAuthority57967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57971

namespace LeftBound57979
def owner : Owner := ⟨.program ⟨257⟩, ⟨37746⟩⟩
def transferEvent : Nat := 57979
def frameStart : Nat := 57883
def rule : BoundRule := .sum [.predecessor 0 57977 .coefficient, .predecessor 1 57978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57977 .coefficient)
      LeftAuthority57975.bound (LeftAuthority57975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57978 .coefficient)
      LeftBound57971.bound (LeftBound57971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57975.bound, LeftBound57971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57975.bound, LeftBound57971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority57975.actual selector witness, LeftBound57971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57979

namespace LeftBound57983
def owner : Owner := ⟨.program ⟨257⟩, ⟨39508⟩⟩
def transferEvent : Nat := 57983
def frameStart : Nat := 57883
def rule : BoundRule := .sum [.predecessor 0 57981 .coefficient, .predecessor 1 57982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57981 .coefficient)
      LeftBound57979.bound (LeftBound57979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57982 .coefficient)
      LeftBound57960.bound (LeftBound57960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57979.bound, LeftBound57960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57979.bound, LeftBound57960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57979.actual selector witness, LeftBound57960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57983

namespace LeftBound57996
def owner : Owner := ⟨.program ⟨257⟩, ⟨39506⟩⟩
def transferEvent : Nat := 57996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57994 .coefficient, .predecessor 1 57995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57994 .coefficient)
      LeftBound57825.bound (LeftBound57825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57995 .coefficient)
      LeftBound57808.bound (LeftBound57808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57825.bound, LeftBound57808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57825.bound, LeftBound57808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57825.actual selector witness, LeftBound57808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57996

namespace LeftBound57999
def owner : Owner := ⟨.program ⟨257⟩, ⟨39506⟩⟩
def transferEvent : Nat := 57999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57993 .summary, .result 57815 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57993 .summary)
      LeftBound57827.bound (LeftBound57827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38335⟩⟩) (rawTerms := some (Proof.Events226.exact57993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57815 .summary)
      LeftBound57810.bound (LeftBound57810.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39505⟩⟩) (rawTerms := some (Proof.Events225.exact57815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57827.bound, LeftBound57810.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57827.bound, LeftBound57810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57827.actual selector witness, LeftBound57810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57999

namespace LeftBound58003
def owner : Owner := ⟨.program ⟨257⟩, ⟨39507⟩⟩
def transferEvent : Nat := 58003
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58001 .coefficient) (.predecessor 1 58002 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 58001 .coefficient)
      LeftBound57996.bound (LeftBound57996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 58002 .coefficient)
      LeftBound15621.bound (LeftBound15621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57996.bound LeftBound15621.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57996.bound, LeftBound15621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57996.actual selector witness) * (LeftBound15621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58003

namespace LeftBound58004
def owner : Owner := ⟨.program ⟨257⟩, ⟨39507⟩⟩
def transferEvent : Nat := 58004
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩ [⟨.result 15618 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15618 .coefficient)
      LeftAuthority15617.bound (LeftAuthority15617.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7161⟩⟩) (rawTerms := some (Proof.Events061.exact15618RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15617.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15617.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15617.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58004

namespace LeftBound58005
def owner : Owner := ⟨.program ⟨257⟩, ⟨39507⟩⟩
def transferEvent : Nat := 58005
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58000 .summary) (.transfer 58004) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58000 .summary)
      LeftBound57999.bound (LeftBound57999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39506⟩⟩) (rawTerms := some (Proof.Events226.exact58000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 58004)
      LeftBound58004.bound (LeftBound58004.actual selector witness) := by
  exact .transfer (LeftBound58004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57999.bound LeftBound58004.bound
def bound : CoeffClass := .finite ⟨345666873099141705532726864949014345809920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57999.bound, LeftBound58004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57999.actual selector witness) * (LeftBound58004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58005

namespace LeftBound58020
def owner : Owner := ⟨.program ⟨257⟩, ⟨36825⟩⟩
def transferEvent : Nat := 58020
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58018 .coefficient) (.predecessor 1 58019 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 58018 .coefficient)
      LeftBound49337.bound (LeftBound49337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 58019 .coefficient)
      LeftAuthority58016.bound (LeftAuthority58016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound49337.bound LeftAuthority58016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49337.bound, LeftAuthority58016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound49337.actual selector witness) * (LeftAuthority58016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58020

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
