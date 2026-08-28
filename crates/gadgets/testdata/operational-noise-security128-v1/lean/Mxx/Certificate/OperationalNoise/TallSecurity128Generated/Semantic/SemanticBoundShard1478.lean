import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1437
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1477

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound220380
def owner : Owner := ⟨.program ⟨257⟩, ⟨55927⟩⟩
def transferEvent : Nat := 220380
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨55925⟩⟩]⟩ [⟨.result 220376 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220376 .coefficient)
      LeftAuthority220375.bound (LeftAuthority220375.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨55925⟩⟩) (rawTerms := some (Proof.Events860.exact220376RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220375.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220375.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority220375.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority220375.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound220380

namespace LeftBound220381
def owner : Owner := ⟨.program ⟨257⟩, ⟨55927⟩⟩
def transferEvent : Nat := 220381
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 213590 .summary) (.transfer 220380) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213590 .summary)
      LeftBound213589.bound (LeftBound213589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55501⟩⟩) (rawTerms := some (Proof.Events834.exact213590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 220380)
      LeftBound220380.bound (LeftBound220380.actual selector witness) := by
  exact .transfer (LeftBound220380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound213589.bound LeftBound220380.bound
def bound : CoeffClass := .finite ⟨32189789464711941702873220382720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213589.bound, LeftBound220380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound213589.actual selector witness) * (LeftBound220380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220381

namespace LeftBound220392
def owner : Owner := ⟨.program ⟨257⟩, ⟨54734⟩⟩
def transferEvent : Nat := 220392
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 220390 .coefficient) (.value (.predecessor 1 220391 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220390 .coefficient)
      LeftAuthority220388.bound (LeftAuthority220388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events860.exact220389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220391 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority220388.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220388.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority220388.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound220392

namespace LeftBound220396
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def transferEvent : Nat := 220396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 220394 .coefficient) (.predecessor 1 220395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220394 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220395 .coefficient)
      LeftBound220392.bound (LeftBound220392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events860.exact220393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound220392.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound220392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound220392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220396

namespace LeftBound220397
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def transferEvent : Nat := 220397
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨54732⟩⟩]⟩ [⟨.result 220389 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220389 .coefficient)
      LeftAuthority220388.bound (LeftAuthority220388.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54732⟩⟩) (rawTerms := some (Proof.Events860.exact220389RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220388.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority220388.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority220388.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound220397

namespace LeftBound220398
def owner : Owner := ⟨.program ⟨257⟩, ⟨54735⟩⟩
def transferEvent : Nat := 220398
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 220397) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 220397)
      LeftBound220397.bound (LeftBound220397.actual selector witness) := by
  exact .transfer (LeftBound220397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound220397.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound220397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound220397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220398

namespace LeftBound220493
def owner : Owner := ⟨.program ⟨257⟩, ⟨53869⟩⟩
def transferEvent : Nat := 220493
def frameStart : Nat := 220454
def rule : BoundRule := .identity (.predecessor 0 220492 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220492 .coefficient)
      LeftAuthority220490.bound (LeftAuthority220490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220490.derived selector witness)

def rawBound : CoeffClass := LeftAuthority220490.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority220490.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound220493

namespace LeftBound220510
def owner : Owner := ⟨.program ⟨257⟩, ⟨55346⟩⟩
def transferEvent : Nat := 220510
def frameStart : Nat := 220454
def rule : BoundRule := .sum [.predecessor 0 220508 .coefficient, .predecessor 1 220509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220508 .coefficient)
      LeftBound220493.bound (LeftBound220493.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound220493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220509 .coefficient)
      LeftAuthority220506.bound (LeftAuthority220506.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority220506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220493.bound, LeftAuthority220506.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220493.bound, LeftAuthority220506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220493.actual selector witness, LeftAuthority220506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220510

namespace LeftBound220513
def owner : Owner := ⟨.program ⟨257⟩, ⟨55347⟩⟩
def transferEvent : Nat := 220513
def frameStart : Nat := 220454
def rule : BoundRule := .identity (.predecessor 0 220512 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220512 .coefficient)
      LeftBound220510.bound (LeftBound220510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound220510.derived selector witness)

def rawBound : CoeffClass := LeftBound220510.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound220510.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound220513

namespace LeftBound220519
def owner : Owner := ⟨.program ⟨257⟩, ⟨55348⟩⟩
def transferEvent : Nat := 220519
def frameStart : Nat := 220454
def rule : BoundRule := .product (.predecessor 0 220517 .coefficient) (.predecessor 1 220518 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220517 .coefficient)
      LeftAuthority220515.bound (LeftAuthority220515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220518 .coefficient)
      LeftBound220513.bound (LeftBound220513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority220515.bound LeftBound220513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220515.bound, LeftBound220513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority220515.actual selector witness) * (LeftBound220513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220519

namespace LeftBound220527
def owner : Owner := ⟨.program ⟨257⟩, ⟨55349⟩⟩
def transferEvent : Nat := 220527
def frameStart : Nat := 220454
def rule : BoundRule := .sum [.predecessor 0 220525 .coefficient, .predecessor 1 220526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220525 .coefficient)
      LeftAuthority220523.bound (LeftAuthority220523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220526 .coefficient)
      LeftBound220519.bound (LeftBound220519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220519.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority220523.bound, LeftBound220519.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220523.bound, LeftBound220519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority220523.actual selector witness, LeftBound220519.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220527

namespace LeftBound220531
def owner : Owner := ⟨.program ⟨257⟩, ⟨55926⟩⟩
def transferEvent : Nat := 220531
def frameStart : Nat := 220454
def rule : BoundRule := .product (.predecessor 0 220529 .coefficient) (.predecessor 1 220530 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220529 .coefficient)
      LeftBound220527.bound (LeftBound220527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220530 .coefficient)
      LeftAuthority220504.bound (LeftAuthority220504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound220527.bound LeftAuthority220504.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220527.bound, LeftAuthority220504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound220527.actual selector witness) * (LeftAuthority220504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220531

namespace LeftBound220542
def owner : Owner := ⟨.program ⟨257⟩, ⟨54148⟩⟩
def transferEvent : Nat := 220542
def frameStart : Nat := 220454
def rule : BoundRule := .product (.predecessor 0 220540 .coefficient) (.predecessor 1 220541 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220540 .coefficient)
      LeftAuthority220515.bound (LeftAuthority220515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220541 .coefficient)
      LeftAuthority220538.bound (LeftAuthority220538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220538.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority220515.bound LeftAuthority220538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220515.bound, LeftAuthority220538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority220515.actual selector witness) * (LeftAuthority220538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220542

namespace LeftBound220550
def owner : Owner := ⟨.program ⟨257⟩, ⟨54149⟩⟩
def transferEvent : Nat := 220550
def frameStart : Nat := 220454
def rule : BoundRule := .sum [.predecessor 0 220548 .coefficient, .predecessor 1 220549 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220548 .coefficient)
      LeftAuthority220546.bound (LeftAuthority220546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220546.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220549 .coefficient)
      LeftBound220542.bound (LeftBound220542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220542.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority220546.bound, LeftBound220542.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220546.bound, LeftBound220542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority220546.actual selector witness, LeftBound220542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220550

namespace LeftBound220554
def owner : Owner := ⟨.program ⟨257⟩, ⟨55931⟩⟩
def transferEvent : Nat := 220554
def frameStart : Nat := 220454
def rule : BoundRule := .sum [.predecessor 0 220552 .coefficient, .predecessor 1 220553 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220552 .coefficient)
      LeftBound220550.bound (LeftBound220550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220553 .coefficient)
      LeftBound220531.bound (LeftBound220531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220550.bound, LeftBound220531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220550.bound, LeftBound220531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220550.actual selector witness, LeftBound220531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220554

namespace LeftBound220567
def owner : Owner := ⟨.program ⟨257⟩, ⟨55928⟩⟩
def transferEvent : Nat := 220567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 220565 .coefficient, .predecessor 1 220566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220565 .coefficient)
      LeftBound220396.bound (LeftBound220396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events861.exact220564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220566 .coefficient)
      LeftBound220379.bound (LeftBound220379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events860.exact220386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220379.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220396.bound, LeftBound220379.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220396.bound, LeftBound220379.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220396.actual selector witness, LeftBound220379.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220567

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
