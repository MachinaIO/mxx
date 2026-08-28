import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1200

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound179904
def owner : Owner := ⟨.program ⟨257⟩, ⟨39868⟩⟩
def transferEvent : Nat := 179904
def frameStart : Nat := 179871
def rule : BoundRule := .identity (.predecessor 0 179903 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179903 .coefficient)
      LeftBound179900.bound (LeftBound179900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179900.derived selector witness)

def rawBound : CoeffClass := LeftBound179900.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound179900.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound179904

namespace LeftBound179921
def owner : Owner := ⟨.program ⟨257⟩, ⟨41398⟩⟩
def transferEvent : Nat := 179921
def frameStart : Nat := 179871
def rule : BoundRule := .sum [.predecessor 0 179919 .coefficient, .predecessor 1 179920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179919 .coefficient)
      LeftBound179904.bound (LeftBound179904.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound179904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179920 .coefficient)
      LeftAuthority179917.bound (LeftAuthority179917.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority179917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound179904.bound, LeftAuthority179917.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179904.bound, LeftAuthority179917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound179904.actual selector witness, LeftAuthority179917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound179921

namespace LeftBound179924
def owner : Owner := ⟨.program ⟨257⟩, ⟨41399⟩⟩
def transferEvent : Nat := 179924
def frameStart : Nat := 179871
def rule : BoundRule := .identity (.predecessor 0 179923 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179923 .coefficient)
      LeftBound179921.bound (LeftBound179921.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound179921.derived selector witness)

def rawBound : CoeffClass := LeftBound179921.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound179921.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound179924

namespace LeftBound179930
def owner : Owner := ⟨.program ⟨257⟩, ⟨41400⟩⟩
def transferEvent : Nat := 179930
def frameStart : Nat := 179871
def rule : BoundRule := .product (.predecessor 0 179928 .coefficient) (.predecessor 1 179929 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179928 .coefficient)
      LeftAuthority179926.bound (LeftAuthority179926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179929 .coefficient)
      LeftBound179924.bound (LeftBound179924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority179926.bound LeftBound179924.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179926.bound, LeftBound179924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority179926.actual selector witness) * (LeftBound179924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound179930

namespace LeftBound179946
def owner : Owner := ⟨.program ⟨257⟩, ⟨9557⟩⟩
def transferEvent : Nat := 179946
def frameStart : Nat := 179871
def rule : BoundRule := .scale (.predecessor 0 179944 .coefficient) (.value (.predecessor 1 179945 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179944 .coefficient)
      LeftAuthority179942.bound (LeftAuthority179942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179945 .coefficient)
      LeftAuthority179933.bound (LeftAuthority179933.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority179933.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority179942.bound LeftAuthority179933.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179942.bound, LeftAuthority179933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority179942.actual selector witness) * (LeftAuthority179933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound179946

namespace LeftBound179949
def owner : Owner := ⟨.program ⟨257⟩, ⟨7299⟩⟩
def transferEvent : Nat := 179949
def frameStart : Nat := 179871
def rule : BoundRule := .identity (.predecessor 0 179948 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179948 .coefficient)
      LeftAuthority179936.bound (LeftAuthority179936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179936.derived selector witness)

def rawBound : CoeffClass := LeftAuthority179936.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority179936.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound179949

namespace LeftBound179953
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def transferEvent : Nat := 179953
def frameStart : Nat := 179871
def rule : BoundRule := .product (.predecessor 0 179951 .coefficient) (.predecessor 1 179952 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179951 .coefficient)
      LeftBound179949.bound (LeftBound179949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179952 .coefficient)
      LeftBound179946.bound (LeftBound179946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound179949.bound LeftBound179946.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179949.bound, LeftBound179946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound179949.actual selector witness) * (LeftBound179946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound179953

namespace LeftBound179958
def owner : Owner := ⟨.program ⟨257⟩, ⟨41401⟩⟩
def transferEvent : Nat := 179958
def frameStart : Nat := 179871
def rule : BoundRule := .sum [.predecessor 0 179956 .coefficient, .predecessor 1 179957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179956 .coefficient)
      LeftBound179953.bound (LeftBound179953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179957 .coefficient)
      LeftBound179930.bound (LeftBound179930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound179953.bound, LeftBound179930.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179953.bound, LeftBound179930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound179953.actual selector witness, LeftBound179930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound179958

namespace LeftBound179962
def owner : Owner := ⟨.program ⟨257⟩, ⟨41655⟩⟩
def transferEvent : Nat := 179962
def frameStart : Nat := 179871
def rule : BoundRule := .product (.predecessor 0 179960 .coefficient) (.predecessor 1 179961 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179960 .coefficient)
      LeftBound179958.bound (LeftBound179958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179961 .coefficient)
      LeftAuthority179915.bound (LeftAuthority179915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound179958.bound LeftAuthority179915.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179958.bound, LeftAuthority179915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound179958.actual selector witness) * (LeftAuthority179915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound179962

namespace LeftBound179973
def owner : Owner := ⟨.program ⟨257⟩, ⟨40134⟩⟩
def transferEvent : Nat := 179973
def frameStart : Nat := 179871
def rule : BoundRule := .product (.predecessor 0 179971 .coefficient) (.predecessor 1 179972 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179971 .coefficient)
      LeftAuthority179926.bound (LeftAuthority179926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179972 .coefficient)
      LeftAuthority179969.bound (LeftAuthority179969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact179970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority179926.bound LeftAuthority179969.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179926.bound, LeftAuthority179969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority179926.actual selector witness) * (LeftAuthority179969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound179973

namespace LeftBound179981
def owner : Owner := ⟨.program ⟨257⟩, ⟨40135⟩⟩
def transferEvent : Nat := 179981
def frameStart : Nat := 179871
def rule : BoundRule := .sum [.predecessor 0 179979 .coefficient, .predecessor 1 179980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179979 .coefficient)
      LeftAuthority179977.bound (LeftAuthority179977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact179978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179980 .coefficient)
      LeftBound179973.bound (LeftBound179973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact179975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority179977.bound, LeftBound179973.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179977.bound, LeftBound179973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority179977.actual selector witness, LeftBound179973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound179981

namespace LeftBound179985
def owner : Owner := ⟨.program ⟨257⟩, ⟨41656⟩⟩
def transferEvent : Nat := 179985
def frameStart : Nat := 179871
def rule : BoundRule := .sum [.predecessor 0 179983 .coefficient, .predecessor 1 179984 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179983 .coefficient)
      LeftBound179981.bound (LeftBound179981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact179982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179984 .coefficient)
      LeftBound179962.bound (LeftBound179962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound179981.bound, LeftBound179962.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179981.bound, LeftBound179962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound179981.actual selector witness, LeftBound179962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound179985

namespace LeftBound179998
def owner : Owner := ⟨.program ⟨257⟩, ⟨41654⟩⟩
def transferEvent : Nat := 179998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 179996 .coefficient, .predecessor 1 179997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 179996 .coefficient)
      LeftBound179819.bound (LeftBound179819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact179995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 179997 .coefficient)
      LeftBound179802.bound (LeftBound179802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound179819.bound, LeftBound179802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179819.bound, LeftBound179802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound179819.actual selector witness, LeftBound179802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound179998

namespace LeftBound180001
def owner : Owner := ⟨.program ⟨257⟩, ⟨41654⟩⟩
def transferEvent : Nat := 180001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 179995 .summary, .result 179809 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 179995 .summary)
      LeftBound179821.bound (LeftBound179821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40582⟩⟩) (rawTerms := some (Proof.Events703.exact179995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound179821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 179809 .summary)
      LeftBound179804.bound (LeftBound179804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41653⟩⟩) (rawTerms := some (Proof.Events702.exact179809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound179804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound179821.bound, LeftBound179804.bound]
def bound : CoeffClass := .finite ⟨2998218789909838430208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179821.bound, LeftBound179804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound179821.actual selector witness, LeftBound179804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound180001

namespace LeftBound180005
def owner : Owner := ⟨.program ⟨257⟩, ⟨42066⟩⟩
def transferEvent : Nat := 180005
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 180003 .coefficient) (.predecessor 1 180004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180003 .coefficient)
      LeftBound179998.bound (LeftBound179998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact180002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180004 .coefficient)
      LeftAuthority179724.bound (LeftAuthority179724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound179998.bound LeftAuthority179724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound179998.bound, LeftAuthority179724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound179998.actual selector witness) * (LeftAuthority179724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180005

namespace LeftBound180006
def owner : Owner := ⟨.program ⟨257⟩, ⟨42066⟩⟩
def transferEvent : Nat := 180006
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨42064⟩⟩]⟩ [⟨.result 179725 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 179725 .coefficient)
      LeftAuthority179724.bound (LeftAuthority179724.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42064⟩⟩) (rawTerms := some (Proof.Events702.exact179725RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority179724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority179724.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority179724.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority179724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority179724.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound180006

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
