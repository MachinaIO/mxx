import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard347

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57190
def owner : Owner := ⟨.program ⟨257⟩, ⟨46375⟩⟩
def transferEvent : Nat := 57190
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨46372⟩⟩]⟩ [⟨.result 57182 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57182 .coefficient)
      LeftAuthority57181.bound (LeftAuthority57181.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨46372⟩⟩) (rawTerms := some (Proof.Events223.exact57182RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57181.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57181.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority57181.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority57181.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57190

namespace LeftBound57191
def owner : Owner := ⟨.program ⟨257⟩, ⟨46375⟩⟩
def transferEvent : Nat := 57191
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 57190) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 57190)
      LeftBound57190.bound (LeftBound57190.actual selector witness) := by
  exact .transfer (LeftBound57190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound57190.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound57190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound57190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57191

namespace LeftBound57286
def owner : Owner := ⟨.program ⟨257⟩, ⟨45533⟩⟩
def transferEvent : Nat := 57286
def frameStart : Nat := 57247
def rule : BoundRule := .identity (.predecessor 0 57285 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57285 .coefficient)
      LeftAuthority57283.bound (LeftAuthority57283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57283.derived selector witness)

def rawBound : CoeffClass := LeftAuthority57283.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority57283.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57286

namespace LeftBound57303
def owner : Owner := ⟨.program ⟨257⟩, ⟨46858⟩⟩
def transferEvent : Nat := 57303
def frameStart : Nat := 57247
def rule : BoundRule := .sum [.predecessor 0 57301 .coefficient, .predecessor 1 57302 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57301 .coefficient)
      LeftBound57286.bound (LeftBound57286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57302 .coefficient)
      LeftAuthority57299.bound (LeftAuthority57299.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority57299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57286.bound, LeftAuthority57299.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57286.bound, LeftAuthority57299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57286.actual selector witness, LeftAuthority57299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57303

namespace LeftBound57306
def owner : Owner := ⟨.program ⟨257⟩, ⟨46859⟩⟩
def transferEvent : Nat := 57306
def frameStart : Nat := 57247
def rule : BoundRule := .identity (.predecessor 0 57305 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57305 .coefficient)
      LeftBound57303.bound (LeftBound57303.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57303.derived selector witness)

def rawBound : CoeffClass := LeftBound57303.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound57303.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57306

namespace LeftBound57312
def owner : Owner := ⟨.program ⟨257⟩, ⟨46860⟩⟩
def transferEvent : Nat := 57312
def frameStart : Nat := 57247
def rule : BoundRule := .product (.predecessor 0 57310 .coefficient) (.predecessor 1 57311 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57310 .coefficient)
      LeftAuthority57308.bound (LeftAuthority57308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57311 .coefficient)
      LeftBound57306.bound (LeftBound57306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority57308.bound LeftBound57306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57308.bound, LeftBound57306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority57308.actual selector witness) * (LeftBound57306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57312

namespace LeftBound57320
def owner : Owner := ⟨.program ⟨257⟩, ⟨46861⟩⟩
def transferEvent : Nat := 57320
def frameStart : Nat := 57247
def rule : BoundRule := .sum [.predecessor 0 57318 .coefficient, .predecessor 1 57319 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57318 .coefficient)
      LeftAuthority57316.bound (LeftAuthority57316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57319 .coefficient)
      LeftBound57312.bound (LeftBound57312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57316.bound, LeftBound57312.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57316.bound, LeftBound57312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority57316.actual selector witness, LeftBound57312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57320

namespace LeftBound57324
def owner : Owner := ⟨.program ⟨257⟩, ⟨47544⟩⟩
def transferEvent : Nat := 57324
def frameStart : Nat := 57247
def rule : BoundRule := .product (.predecessor 0 57322 .coefficient) (.predecessor 1 57323 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57322 .coefficient)
      LeftBound57320.bound (LeftBound57320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57323 .coefficient)
      LeftAuthority57297.bound (LeftAuthority57297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57297.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57320.bound LeftAuthority57297.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57320.bound, LeftAuthority57297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57320.actual selector witness) * (LeftAuthority57297.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57324

namespace LeftBound57335
def owner : Owner := ⟨.program ⟨257⟩, ⟨45785⟩⟩
def transferEvent : Nat := 57335
def frameStart : Nat := 57247
def rule : BoundRule := .product (.predecessor 0 57333 .coefficient) (.predecessor 1 57334 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57333 .coefficient)
      LeftAuthority57308.bound (LeftAuthority57308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57334 .coefficient)
      LeftAuthority57331.bound (LeftAuthority57331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57331.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57331.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57308.bound LeftAuthority57331.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57308.bound, LeftAuthority57331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority57308.actual selector witness) * (LeftAuthority57331.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57335

namespace LeftBound57343
def owner : Owner := ⟨.program ⟨257⟩, ⟨45786⟩⟩
def transferEvent : Nat := 57343
def frameStart : Nat := 57247
def rule : BoundRule := .sum [.predecessor 0 57341 .coefficient, .predecessor 1 57342 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57341 .coefficient)
      LeftAuthority57339.bound (LeftAuthority57339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57339.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57342 .coefficient)
      LeftBound57335.bound (LeftBound57335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57339.bound, LeftBound57335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57339.bound, LeftBound57335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority57339.actual selector witness, LeftBound57335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57343

namespace LeftBound57347
def owner : Owner := ⟨.program ⟨257⟩, ⟨47548⟩⟩
def transferEvent : Nat := 57347
def frameStart : Nat := 57247
def rule : BoundRule := .sum [.predecessor 0 57345 .coefficient, .predecessor 1 57346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57345 .coefficient)
      LeftBound57343.bound (LeftBound57343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57346 .coefficient)
      LeftBound57324.bound (LeftBound57324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57343.bound, LeftBound57324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57343.bound, LeftBound57324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57343.actual selector witness, LeftBound57324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57347

namespace LeftBound57360
def owner : Owner := ⟨.program ⟨257⟩, ⟨47546⟩⟩
def transferEvent : Nat := 57360
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57358 .coefficient, .predecessor 1 57359 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57358 .coefficient)
      LeftBound57189.bound (LeftBound57189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57359 .coefficient)
      LeftBound57172.bound (LeftBound57172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57172.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57189.bound, LeftBound57172.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57189.bound, LeftBound57172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57189.actual selector witness, LeftBound57172.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57360

namespace LeftBound57363
def owner : Owner := ⟨.program ⟨257⟩, ⟨47546⟩⟩
def transferEvent : Nat := 57363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57357 .summary, .result 57179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57357 .summary)
      LeftBound57191.bound (LeftBound57191.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46375⟩⟩) (rawTerms := some (Proof.Events224.exact57357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57179 .summary)
      LeftBound57174.bound (LeftBound57174.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47545⟩⟩) (rawTerms := some (Proof.Events223.exact57179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57191.bound, LeftBound57174.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57191.bound, LeftBound57174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound57191.actual selector witness, LeftBound57174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57363

namespace LeftBound57367
def owner : Owner := ⟨.program ⟨257⟩, ⟨47547⟩⟩
def transferEvent : Nat := 57367
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57365 .coefficient) (.predecessor 1 57366 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 57365 .coefficient)
      LeftBound57360.bound (LeftBound57360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 57366 .coefficient)
      LeftBound15561.bound (LeftBound15561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15561.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57360.bound LeftBound15561.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57360.bound, LeftBound15561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57360.actual selector witness) * (LeftBound15561.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57367

namespace LeftBound57368
def owner : Owner := ⟨.program ⟨257⟩, ⟨47547⟩⟩
def transferEvent : Nat := 57368
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩ [⟨.result 15558 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15558 .coefficient)
      LeftAuthority15557.bound (LeftAuthority15557.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7151⟩⟩) (rawTerms := some (Proof.Events060.exact15558RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15557.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15557.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15557.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57368

namespace LeftBound57369
def owner : Owner := ⟨.program ⟨257⟩, ⟨47547⟩⟩
def transferEvent : Nat := 57369
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 57364 .summary) (.transfer 57368) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57364 .summary)
      LeftBound57363.bound (LeftBound57363.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47546⟩⟩) (rawTerms := some (Proof.Events224.exact57364RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 57368)
      LeftBound57368.bound (LeftBound57368.actual selector witness) := by
  exact .transfer (LeftBound57368.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound57363.bound LeftBound57368.bound
def bound : CoeffClass := .finite ⟨345683748063931943722519589062084311121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57363.bound, LeftBound57368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound57363.actual selector witness) * (LeftBound57368.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57369

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
