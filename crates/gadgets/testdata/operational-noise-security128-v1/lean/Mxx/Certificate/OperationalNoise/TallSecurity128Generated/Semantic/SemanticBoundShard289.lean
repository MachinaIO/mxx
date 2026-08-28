import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard288

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48381
def owner : Owner := ⟨.program ⟨257⟩, ⟨42191⟩⟩
def transferEvent : Nat := 48381
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨42189⟩⟩]⟩ [⟨.result 48100 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 48100 .coefficient)
      LeftAuthority48099.bound (LeftAuthority48099.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42189⟩⟩) (rawTerms := some (Proof.Events187.exact48100RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48099.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48099.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48099.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority48099.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48381

namespace LeftBound48382
def owner : Owner := ⟨.program ⟨257⟩, ⟨42191⟩⟩
def transferEvent : Nat := 48382
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48377 .summary) (.transfer 48381) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 48377 .summary)
      LeftBound48376.bound (LeftBound48376.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41709⟩⟩) (rawTerms := some (Proof.Events188.exact48377RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 48381)
      LeftBound48381.bound (LeftBound48381.actual selector witness) := by
  exact .transfer (LeftBound48381.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound48376.bound LeftBound48381.bound
def bound : CoeffClass := .finite ⟨32193129122288627115968346193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48376.bound, LeftBound48381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound48376.actual selector witness) * (LeftBound48381.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48382

namespace LeftBound48393
def owner : Owner := ⟨.program ⟨257⟩, ⟨41018⟩⟩
def transferEvent : Nat := 48393
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48391 .coefficient) (.value (.predecessor 1 48392 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48391 .coefficient)
      LeftAuthority48389.bound (LeftAuthority48389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48392 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48389.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48389.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority48389.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48393

namespace LeftBound48397
def owner : Owner := ⟨.program ⟨257⟩, ⟨41019⟩⟩
def transferEvent : Nat := 48397
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48395 .coefficient) (.predecessor 1 48396 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48395 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48396 .coefficient)
      LeftBound48393.bound (LeftBound48393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48393.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound48393.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound48393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound48393.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48397

namespace LeftBound48398
def owner : Owner := ⟨.program ⟨257⟩, ⟨41019⟩⟩
def transferEvent : Nat := 48398
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨41016⟩⟩]⟩ [⟨.result 48390 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 48390 .coefficient)
      LeftAuthority48389.bound (LeftAuthority48389.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨41016⟩⟩) (rawTerms := some (Proof.Events189.exact48390RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48389.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48389.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority48389.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48398

namespace LeftBound48399
def owner : Owner := ⟨.program ⟨257⟩, ⟨41019⟩⟩
def transferEvent : Nat := 48399
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 48398) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 48398)
      LeftBound48398.bound (LeftBound48398.actual selector witness) := by
  exact .transfer (LeftBound48398.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound48398.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound48398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound48398.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48399

namespace LeftBound48494
def owner : Owner := ⟨.program ⟨257⟩, ⟨40173⟩⟩
def transferEvent : Nat := 48494
def frameStart : Nat := 48455
def rule : BoundRule := .identity (.predecessor 0 48493 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48493 .coefficient)
      LeftAuthority48491.bound (LeftAuthority48491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48491.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48491.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority48491.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48494

namespace LeftBound48511
def owner : Owner := ⟨.program ⟨257⟩, ⟨41498⟩⟩
def transferEvent : Nat := 48511
def frameStart : Nat := 48455
def rule : BoundRule := .sum [.predecessor 0 48509 .coefficient, .predecessor 1 48510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48509 .coefficient)
      LeftBound48494.bound (LeftBound48494.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48510 .coefficient)
      LeftAuthority48507.bound (LeftAuthority48507.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48507.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48494.bound, LeftAuthority48507.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48494.bound, LeftAuthority48507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound48494.actual selector witness, LeftAuthority48507.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48511

namespace LeftBound48514
def owner : Owner := ⟨.program ⟨257⟩, ⟨41499⟩⟩
def transferEvent : Nat := 48514
def frameStart : Nat := 48455
def rule : BoundRule := .identity (.predecessor 0 48513 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48513 .coefficient)
      LeftBound48511.bound (LeftBound48511.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48511.derived selector witness)

def rawBound : CoeffClass := LeftBound48511.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound48511.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48514

namespace LeftBound48520
def owner : Owner := ⟨.program ⟨257⟩, ⟨41500⟩⟩
def transferEvent : Nat := 48520
def frameStart : Nat := 48455
def rule : BoundRule := .product (.predecessor 0 48518 .coefficient) (.predecessor 1 48519 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48518 .coefficient)
      LeftAuthority48516.bound (LeftAuthority48516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48519 .coefficient)
      LeftBound48514.bound (LeftBound48514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority48516.bound LeftBound48514.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48516.bound, LeftBound48514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority48516.actual selector witness) * (LeftBound48514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48520

namespace LeftBound48528
def owner : Owner := ⟨.program ⟨257⟩, ⟨41501⟩⟩
def transferEvent : Nat := 48528
def frameStart : Nat := 48455
def rule : BoundRule := .sum [.predecessor 0 48526 .coefficient, .predecessor 1 48527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48526 .coefficient)
      LeftAuthority48524.bound (LeftAuthority48524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48527 .coefficient)
      LeftBound48520.bound (LeftBound48520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48524.bound, LeftBound48520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48524.bound, LeftBound48520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority48524.actual selector witness, LeftBound48520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48528

namespace LeftBound48532
def owner : Owner := ⟨.program ⟨257⟩, ⟨42190⟩⟩
def transferEvent : Nat := 48532
def frameStart : Nat := 48455
def rule : BoundRule := .product (.predecessor 0 48530 .coefficient) (.predecessor 1 48531 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48530 .coefficient)
      LeftBound48528.bound (LeftBound48528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48531 .coefficient)
      LeftAuthority48505.bound (LeftAuthority48505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound48528.bound LeftAuthority48505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48528.bound, LeftAuthority48505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound48528.actual selector witness) * (LeftAuthority48505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48532

namespace LeftBound48543
def owner : Owner := ⟨.program ⟨257⟩, ⟨40424⟩⟩
def transferEvent : Nat := 48543
def frameStart : Nat := 48455
def rule : BoundRule := .product (.predecessor 0 48541 .coefficient) (.predecessor 1 48542 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48541 .coefficient)
      LeftAuthority48516.bound (LeftAuthority48516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48542 .coefficient)
      LeftAuthority48539.bound (LeftAuthority48539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority48516.bound LeftAuthority48539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48516.bound, LeftAuthority48539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority48516.actual selector witness) * (LeftAuthority48539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48543

namespace LeftBound48551
def owner : Owner := ⟨.program ⟨257⟩, ⟨40425⟩⟩
def transferEvent : Nat := 48551
def frameStart : Nat := 48455
def rule : BoundRule := .sum [.predecessor 0 48549 .coefficient, .predecessor 1 48550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48549 .coefficient)
      LeftAuthority48547.bound (LeftAuthority48547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48550 .coefficient)
      LeftBound48543.bound (LeftBound48543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48547.bound, LeftBound48543.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48547.bound, LeftBound48543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority48547.actual selector witness, LeftBound48543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48551

namespace LeftBound48555
def owner : Owner := ⟨.program ⟨257⟩, ⟨42193⟩⟩
def transferEvent : Nat := 48555
def frameStart : Nat := 48455
def rule : BoundRule := .sum [.predecessor 0 48553 .coefficient, .predecessor 1 48554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48553 .coefficient)
      LeftBound48551.bound (LeftBound48551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48554 .coefficient)
      LeftBound48532.bound (LeftBound48532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48551.bound, LeftBound48532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48551.bound, LeftBound48532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound48551.actual selector witness, LeftBound48532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48555

namespace LeftBound48568
def owner : Owner := ⟨.program ⟨257⟩, ⟨42192⟩⟩
def transferEvent : Nat := 48568
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48566 .coefficient, .predecessor 1 48567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 48566 .coefficient)
      LeftBound48397.bound (LeftBound48397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 48567 .coefficient)
      LeftBound48380.bound (LeftBound48380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48397.bound, LeftBound48380.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48397.bound, LeftBound48380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound48397.actual selector witness, LeftBound48380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48568

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
