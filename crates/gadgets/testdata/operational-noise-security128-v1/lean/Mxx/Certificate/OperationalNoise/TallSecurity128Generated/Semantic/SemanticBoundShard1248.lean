import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1247

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound186087
def owner : Owner := ⟨.program ⟨257⟩, ⟨19182⟩⟩
def transferEvent : Nat := 186087
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 186086) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 186086)
      LeftBound186086.bound (LeftBound186086.actual selector witness) := by
  exact .transfer (LeftBound186086.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound186086.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound186086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound186086.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186087

namespace LeftBound186166
def owner : Owner := ⟨.program ⟨257⟩, ⟨18347⟩⟩
def transferEvent : Nat := 186166
def frameStart : Nat := 186137
def rule : BoundRule := .product (.predecessor 0 186164 .coefficient) (.predecessor 1 186165 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186164 .coefficient)
      LeftAuthority186162.bound (LeftAuthority186162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186165 .coefficient)
      LeftAuthority186159.bound (LeftAuthority186159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186159.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority186162.bound LeftAuthority186159.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186162.bound, LeftAuthority186159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority186162.actual selector witness) * (LeftAuthority186159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186166

namespace LeftBound186170
def owner : Owner := ⟨.program ⟨257⟩, ⟨18348⟩⟩
def transferEvent : Nat := 186170
def frameStart : Nat := 186137
def rule : BoundRule := .identity (.predecessor 0 186169 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186169 .coefficient)
      LeftBound186166.bound (LeftBound186166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186166.derived selector witness)

def rawBound : CoeffClass := LeftBound186166.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound186166.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186170

namespace LeftBound186187
def owner : Owner := ⟨.program ⟨257⟩, ⟨19998⟩⟩
def transferEvent : Nat := 186187
def frameStart : Nat := 186137
def rule : BoundRule := .sum [.predecessor 0 186185 .coefficient, .predecessor 1 186186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186185 .coefficient)
      LeftBound186170.bound (LeftBound186170.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound186170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186186 .coefficient)
      LeftAuthority186183.bound (LeftAuthority186183.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority186183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186170.bound, LeftAuthority186183.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186170.bound, LeftAuthority186183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186170.actual selector witness, LeftAuthority186183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186187

namespace LeftBound186190
def owner : Owner := ⟨.program ⟨257⟩, ⟨19999⟩⟩
def transferEvent : Nat := 186190
def frameStart : Nat := 186137
def rule : BoundRule := .identity (.predecessor 0 186189 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186189 .coefficient)
      LeftBound186187.bound (LeftBound186187.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound186187.derived selector witness)

def rawBound : CoeffClass := LeftBound186187.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound186187.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186190

namespace LeftBound186196
def owner : Owner := ⟨.program ⟨257⟩, ⟨20000⟩⟩
def transferEvent : Nat := 186196
def frameStart : Nat := 186137
def rule : BoundRule := .product (.predecessor 0 186194 .coefficient) (.predecessor 1 186195 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186194 .coefficient)
      LeftAuthority186192.bound (LeftAuthority186192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186195 .coefficient)
      LeftBound186190.bound (LeftBound186190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority186192.bound LeftBound186190.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186192.bound, LeftBound186190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority186192.actual selector witness) * (LeftBound186190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186196

namespace LeftBound186212
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 186212
def frameStart : Nat := 186137
def rule : BoundRule := .scale (.predecessor 0 186210 .coefficient) (.value (.predecessor 1 186211 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186210 .coefficient)
      LeftAuthority186208.bound (LeftAuthority186208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186211 .coefficient)
      LeftAuthority186199.bound (LeftAuthority186199.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority186199.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority186208.bound LeftAuthority186199.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186208.bound, LeftAuthority186199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority186208.actual selector witness) * (LeftAuthority186199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound186212

namespace LeftBound186215
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 186215
def frameStart : Nat := 186137
def rule : BoundRule := .identity (.predecessor 0 186214 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186214 .coefficient)
      LeftAuthority186202.bound (LeftAuthority186202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186202.derived selector witness)

def rawBound : CoeffClass := LeftAuthority186202.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority186202.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186215

namespace LeftBound186219
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 186219
def frameStart : Nat := 186137
def rule : BoundRule := .product (.predecessor 0 186217 .coefficient) (.predecessor 1 186218 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186217 .coefficient)
      LeftBound186215.bound (LeftBound186215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186218 .coefficient)
      LeftBound186212.bound (LeftBound186212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186215.bound LeftBound186212.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186215.bound, LeftBound186212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186215.actual selector witness) * (LeftBound186212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186219

namespace LeftBound186224
def owner : Owner := ⟨.program ⟨257⟩, ⟨20001⟩⟩
def transferEvent : Nat := 186224
def frameStart : Nat := 186137
def rule : BoundRule := .sum [.predecessor 0 186222 .coefficient, .predecessor 1 186223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186222 .coefficient)
      LeftBound186219.bound (LeftBound186219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186223 .coefficient)
      LeftBound186196.bound (LeftBound186196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186219.bound, LeftBound186196.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186219.bound, LeftBound186196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186219.actual selector witness, LeftBound186196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186224

namespace LeftBound186228
def owner : Owner := ⟨.program ⟨257⟩, ⟨20255⟩⟩
def transferEvent : Nat := 186228
def frameStart : Nat := 186137
def rule : BoundRule := .product (.predecessor 0 186226 .coefficient) (.predecessor 1 186227 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186226 .coefficient)
      LeftBound186224.bound (LeftBound186224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186227 .coefficient)
      LeftAuthority186181.bound (LeftAuthority186181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186181.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186224.bound LeftAuthority186181.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186224.bound, LeftAuthority186181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186224.actual selector witness) * (LeftAuthority186181.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186228

namespace LeftBound186239
def owner : Owner := ⟨.program ⟨257⟩, ⟨18614⟩⟩
def transferEvent : Nat := 186239
def frameStart : Nat := 186137
def rule : BoundRule := .product (.predecessor 0 186237 .coefficient) (.predecessor 1 186238 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186237 .coefficient)
      LeftAuthority186192.bound (LeftAuthority186192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186238 .coefficient)
      LeftAuthority186235.bound (LeftAuthority186235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority186192.bound LeftAuthority186235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186192.bound, LeftAuthority186235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority186192.actual selector witness) * (LeftAuthority186235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186239

namespace LeftBound186247
def owner : Owner := ⟨.program ⟨257⟩, ⟨18615⟩⟩
def transferEvent : Nat := 186247
def frameStart : Nat := 186137
def rule : BoundRule := .sum [.predecessor 0 186245 .coefficient, .predecessor 1 186246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186245 .coefficient)
      LeftAuthority186243.bound (LeftAuthority186243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186246 .coefficient)
      LeftBound186239.bound (LeftBound186239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority186243.bound, LeftBound186239.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186243.bound, LeftBound186239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority186243.actual selector witness, LeftBound186239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186247

namespace LeftBound186251
def owner : Owner := ⟨.program ⟨257⟩, ⟨20256⟩⟩
def transferEvent : Nat := 186251
def frameStart : Nat := 186137
def rule : BoundRule := .sum [.predecessor 0 186249 .coefficient, .predecessor 1 186250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186249 .coefficient)
      LeftBound186247.bound (LeftBound186247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186250 .coefficient)
      LeftBound186228.bound (LeftBound186228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186247.bound, LeftBound186228.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186247.bound, LeftBound186228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186247.actual selector witness, LeftBound186228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186251

namespace LeftBound186264
def owner : Owner := ⟨.program ⟨257⟩, ⟨20254⟩⟩
def transferEvent : Nat := 186264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186262 .coefficient, .predecessor 1 186263 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186262 .coefficient)
      LeftBound186085.bound (LeftBound186085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events727.exact186261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186263 .coefficient)
      LeftBound186068.bound (LeftBound186068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact186075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186085.bound, LeftBound186068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186085.bound, LeftBound186068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186085.actual selector witness, LeftBound186068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186264

namespace LeftBound186267
def owner : Owner := ⟨.program ⟨257⟩, ⟨20254⟩⟩
def transferEvent : Nat := 186267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186261 .summary, .result 186075 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186261 .summary)
      LeftBound186087.bound (LeftBound186087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19182⟩⟩) (rawTerms := some (Proof.Events727.exact186261RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186075 .summary)
      LeftBound186070.bound (LeftBound186070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20253⟩⟩) (rawTerms := some (Proof.Events726.exact186075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186087.bound, LeftBound186070.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186087.bound, LeftBound186070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186087.actual selector witness, LeftBound186070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186267

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
