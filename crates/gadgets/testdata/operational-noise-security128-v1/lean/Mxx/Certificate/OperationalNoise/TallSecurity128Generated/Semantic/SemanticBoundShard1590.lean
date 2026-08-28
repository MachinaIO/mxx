import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1490
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1563
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1589

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound236387
def owner : Owner := ⟨.program ⟨257⟩, ⟨71210⟩⟩
def transferEvent : Nat := 236387
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236383 .summary, .result 232450 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236383 .summary)
      LeftBound236382.bound (LeftBound236382.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70096⟩⟩) (rawTerms := some (Proof.Events923.exact236383RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 232450 .summary)
      LeftBound232445.bound (LeftBound232445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71208⟩⟩) (rawTerms := some (Proof.Events908.exact232450RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound232445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236382.bound, LeftBound232445.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236382.bound, LeftBound232445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236382.actual selector witness, LeftBound232445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236387

namespace LeftBound236393
def owner : Owner := ⟨.program ⟨257⟩, ⟨7415⟩⟩
def transferEvent : Nat := 236393
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 236391 .coefficient) (.predecessor 1 236392 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236391 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236392 .coefficient)
      LeftAuthority16546.bound (LeftAuthority16546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16546.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16546.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236393

namespace LeftBound236398
def owner : Owner := ⟨.program ⟨257⟩, ⟨9233⟩⟩
def transferEvent : Nat := 236398
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236396 .coefficient, .predecessor 1 236397 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236396 .coefficient)
      LeftBound236393.bound (LeftBound236393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236397 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236393.bound, LeftBound222151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236393.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236393.actual selector witness, LeftBound222151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236398

namespace LeftBound236402
def owner : Owner := ⟨.program ⟨257⟩, ⟨9234⟩⟩
def transferEvent : Nat := 236402
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236400 .coefficient, .predecessor 1 236401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236400 .coefficient)
      LeftBound236398.bound (LeftBound236398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236401 .coefficient)
      LeftAuthority236389.bound (LeftAuthority236389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236398.bound, LeftAuthority236389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236398.bound, LeftAuthority236389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236398.actual selector witness, LeftAuthority236389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236402

namespace LeftBound236403
def owner : Owner := ⟨.program ⟨257⟩, ⟨9234⟩⟩
def transferEvent : Nat := 236403
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨2⟩⟩]⟩ [⟨.result 236390 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236390 .coefficient)
      LeftAuthority236389.bound (LeftAuthority236389.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨2⟩⟩) (rawTerms := some (Proof.Events923.exact236390RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236389.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority236389.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority236389.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound236403

namespace LeftBound236408
def owner : Owner := ⟨.program ⟨257⟩, ⟨9627⟩⟩
def transferEvent : Nat := 236408
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 236406 .coefficient) (.predecessor 1 236407 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236406 .coefficient)
      LeftBound236402.bound (LeftBound236402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236407 .coefficient)
      LeftBound15983.bound (LeftBound15983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound236402.bound LeftBound15983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236402.bound, LeftBound15983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound236402.actual selector witness) * (LeftBound15983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236408

namespace LeftBound236409
def owner : Owner := ⟨.program ⟨257⟩, ⟨9627⟩⟩
def transferEvent : Nat := 236409
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩ [⟨.result 15980 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15980 .coefficient)
      LeftAuthority15979.bound (LeftAuthority15979.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9583⟩⟩) (rawTerms := some (Proof.Events062.exact15980RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15979.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15979.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15979.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound236409

namespace LeftBound236410
def owner : Owner := ⟨.program ⟨257⟩, ⟨9627⟩⟩
def transferEvent : Nat := 236410
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236405 .summary) (.transfer 236409) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236405 .summary)
      LeftBound236403.bound (LeftBound236403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9234⟩⟩) (rawTerms := some (Proof.Events923.exact236405RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 236409)
      LeftBound236409.bound (LeftBound236409.actual selector witness) := by
  exact .transfer (LeftBound236409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound236403.bound LeftBound236409.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236403.bound, LeftBound236409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound236403.actual selector witness) * (LeftBound236409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236410

namespace LeftBound236436
def owner : Owner := ⟨.program ⟨257⟩, ⟨71211⟩⟩
def transferEvent : Nat := 236436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236434 .coefficient, .predecessor 1 236435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236434 .coefficient)
      LeftBound236408.bound (LeftBound236408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236435 .coefficient)
      LeftBound236386.bound (LeftBound236386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236408.bound, LeftBound236386.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236408.bound, LeftBound236386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236408.actual selector witness, LeftBound236386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236436

namespace LeftBound236456
def owner : Owner := ⟨.program ⟨257⟩, ⟨71211⟩⟩
def transferEvent : Nat := 236456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236433 .summary, .result 236388 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236433 .summary)
      LeftBound236410.bound (LeftBound236410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9627⟩⟩) (rawTerms := some (Proof.Events923.exact236433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236388 .summary)
      LeftBound236387.bound (LeftBound236387.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71210⟩⟩) (rawTerms := some (Proof.Events923.exact236388RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236410.bound, LeftBound236387.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236410.bound, LeftBound236387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236410.actual selector witness, LeftBound236387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236456

namespace LeftBound236460
def owner : Owner := ⟨.program ⟨257⟩, ⟨71212⟩⟩
def transferEvent : Nat := 236460
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 236458 .coefficient) (.predecessor 1 236459 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236458 .coefficient)
      LeftBound236436.bound (LeftBound236436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236459 .coefficient)
      LeftBound16543.bound (LeftBound16543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16543.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound236436.bound LeftBound16543.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236436.bound, LeftBound16543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound236436.actual selector witness) * (LeftBound16543.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236460

namespace LeftBound236461
def owner : Owner := ⟨.program ⟨257⟩, ⟨71212⟩⟩
def transferEvent : Nat := 236461
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9519⟩⟩]⟩ [⟨.result 16540 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16540 .coefficient)
      LeftAuthority16539.bound (LeftAuthority16539.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9519⟩⟩) (rawTerms := some (Proof.Events064.exact16540RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16539.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16539.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16539.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound236461

namespace LeftBound236462
def owner : Owner := ⟨.program ⟨257⟩, ⟨71212⟩⟩
def transferEvent : Nat := 236462
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236457 .summary) (.transfer 236461) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236457 .summary)
      LeftBound236456.bound (LeftBound236456.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71211⟩⟩) (rawTerms := some (Proof.Events923.exact236457RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 236461)
      LeftBound236461.bound (LeftBound236461.actual selector witness) := by
  exact .transfer (LeftBound236461.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound236456.bound LeftBound236461.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167293655866246524336865280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236456.bound, LeftBound236461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound236456.actual selector witness) * (LeftBound236461.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236462

namespace LeftBound236524
def owner : Owner := ⟨.program ⟨257⟩, ⟨71213⟩⟩
def transferEvent : Nat := 236524
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236522 .coefficient, .predecessor 1 236523 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236522 .coefficient)
      LeftBound236460.bound (LeftBound236460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236523 .coefficient)
      LeftBound222041.bound (LeftBound222041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236460.bound, LeftBound222041.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236460.bound, LeftBound222041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236460.actual selector witness, LeftBound222041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236524

namespace LeftBound236544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71213⟩⟩
def transferEvent : Nat := 236544
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236521 .summary, .result 222118 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236521 .summary)
      LeftBound236462.bound (LeftBound236462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71212⟩⟩) (rawTerms := some (Proof.Events923.exact236521RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222118 .summary)
      LeftBound222079.bound (LeftBound222079.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67443⟩⟩) (rawTerms := some (Proof.Events867.exact222118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236462.bound, LeftBound222079.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236462.bound, LeftBound222079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236462.actual selector witness, LeftBound222079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236544

namespace LeftBound236548
def owner : Owner := ⟨.program ⟨257⟩, ⟨71214⟩⟩
def transferEvent : Nat := 236548
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 236546 .coefficient) (.predecessor 1 236547 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236546 .coefficient)
      LeftBound236524.bound (LeftBound236524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236547 .coefficient)
      LeftBound16533.bound (LeftBound16533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound236524.bound LeftBound16533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236524.bound, LeftBound16533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound236524.actual selector witness) * (LeftBound16533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound236548

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
