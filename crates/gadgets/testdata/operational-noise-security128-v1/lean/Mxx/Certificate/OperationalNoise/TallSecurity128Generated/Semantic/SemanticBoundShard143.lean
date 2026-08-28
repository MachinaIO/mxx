import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard140
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard142

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27598
def owner : Owner := ⟨.program ⟨257⟩, ⟨67299⟩⟩
def transferEvent : Nat := 27598
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27596 .coefficient, .predecessor 1 27597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27596 .coefficient)
      LeftAuthority27594.bound (LeftAuthority27594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27597 .coefficient)
      LeftBound27590.bound (LeftBound27590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27594.bound, LeftBound27590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27594.bound, LeftBound27590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority27594.actual selector witness, LeftBound27590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27598

namespace LeftBound27602
def owner : Owner := ⟨.program ⟨257⟩, ⟨70973⟩⟩
def transferEvent : Nat := 27602
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27600 .coefficient, .predecessor 1 27601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27600 .coefficient)
      LeftBound27598.bound (LeftBound27598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27601 .coefficient)
      LeftBound27511.bound (LeftBound27511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27598.bound, LeftBound27511.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27598.bound, LeftBound27511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27598.actual selector witness, LeftBound27511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27602

namespace LeftBound27649
def owner : Owner := ⟨.program ⟨257⟩, ⟨70971⟩⟩
def transferEvent : Nat := 27649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27647 .coefficient, .predecessor 1 27648 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27647 .coefficient)
      LeftBound26240.bound (LeftBound26240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27648 .coefficient)
      LeftBound26155.bound (LeftBound26155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26240.bound, LeftBound26155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26240.bound, LeftBound26155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26240.actual selector witness, LeftBound26155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27649

namespace LeftBound27686
def owner : Owner := ⟨.program ⟨257⟩, ⟨70971⟩⟩
def transferEvent : Nat := 27686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27646 .summary, .result 26230 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27646 .summary)
      LeftBound26242.bound (LeftBound26242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68286⟩⟩) (rawTerms := some (Proof.Events107.exact27646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26230 .summary)
      LeftBound26157.bound (LeftBound26157.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70970⟩⟩) (rawTerms := some (Proof.Events102.exact26230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26242.bound, LeftBound26157.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469506489977540968448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26242.bound, LeftBound26157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26242.actual selector witness, LeftBound26157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27686

namespace LeftBound27690
def owner : Owner := ⟨.program ⟨257⟩, ⟨70972⟩⟩
def transferEvent : Nat := 27690
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27688 .coefficient) (.predecessor 1 27689 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27688 .coefficient)
      LeftBound27649.bound (LeftBound27649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27689 .coefficient)
      LeftBound15521.bound (LeftBound15521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27649.bound LeftBound15521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27649.bound, LeftBound15521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27649.actual selector witness) * (LeftBound15521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27690

namespace LeftBound27691
def owner : Owner := ⟨.program ⟨257⟩, ⟨70972⟩⟩
def transferEvent : Nat := 27691
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩ [⟨.result 15518 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15518 .coefficient)
      LeftAuthority15517.bound (LeftAuthority15517.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7139⟩⟩) (rawTerms := some (Proof.Events060.exact15518RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15517.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15517.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15517.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27691

namespace LeftBound27692
def owner : Owner := ⟨.program ⟨257⟩, ⟨70972⟩⟩
def transferEvent : Nat := 27692
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27687 .summary) (.transfer 27691) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27687 .summary)
      LeftBound27686.bound (LeftBound27686.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70971⟩⟩) (rawTerms := some (Proof.Events108.exact27687RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 27691)
      LeftBound27691.bound (LeftBound27691.actual selector witness) := by
  exact .transfer (LeftBound27691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27686.bound LeftBound27691.bound
def bound : CoeffClass := .finite ⟨66805187221379434678483228029309283225584960819691520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27686.bound, LeftBound27691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27686.actual selector witness) * (LeftBound27691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27692

namespace LeftBound27707
def owner : Owner := ⟨.program ⟨257⟩, ⟨49807⟩⟩
def transferEvent : Nat := 27707
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27705 .coefficient) (.predecessor 1 27706 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27705 .coefficient)
      LeftBound17351.bound (LeftBound17351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27706 .coefficient)
      LeftAuthority27703.bound (LeftAuthority27703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27703.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound17351.bound LeftAuthority27703.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17351.bound, LeftAuthority27703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound17351.actual selector witness) * (LeftAuthority27703.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27707

namespace LeftBound27708
def owner : Owner := ⟨.program ⟨257⟩, ⟨49807⟩⟩
def transferEvent : Nat := 27708
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨49805⟩⟩]⟩ [⟨.result 27704 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27704 .coefficient)
      LeftAuthority27703.bound (LeftAuthority27703.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨49805⟩⟩) (rawTerms := some (Proof.Events108.exact27704RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27703.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27703.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority27703.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27708

namespace LeftBound27709
def owner : Owner := ⟨.program ⟨257⟩, ⟨49807⟩⟩
def transferEvent : Nat := 27709
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17355 .summary) (.transfer 27708) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17355 .summary)
      LeftBound17354.bound (LeftBound17354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49565⟩⟩) (rawTerms := some (Proof.Events067.exact17355RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 27708)
      LeftBound27708.bound (LeftBound27708.actual selector witness) := by
  exact .transfer (LeftBound27708.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound17354.bound LeftBound27708.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17354.bound, LeftBound27708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound17354.actual selector witness) * (LeftBound27708.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27709

namespace LeftBound27720
def owner : Owner := ⟨.program ⟨257⟩, ⟨48720⟩⟩
def transferEvent : Nat := 27720
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27718 .coefficient) (.value (.predecessor 1 27719 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27718 .coefficient)
      LeftAuthority27716.bound (LeftAuthority27716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27719 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27716.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27716.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority27716.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27720

namespace LeftBound27724
def owner : Owner := ⟨.program ⟨257⟩, ⟨48721⟩⟩
def transferEvent : Nat := 27724
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27722 .coefficient) (.predecessor 1 27723 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27722 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27723 .coefficient)
      LeftBound27720.bound (LeftBound27720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound27720.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound27720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound27720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27724

namespace LeftBound27725
def owner : Owner := ⟨.program ⟨257⟩, ⟨48721⟩⟩
def transferEvent : Nat := 27725
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨48718⟩⟩]⟩ [⟨.result 27717 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27717 .coefficient)
      LeftAuthority27716.bound (LeftAuthority27716.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48718⟩⟩) (rawTerms := some (Proof.Events108.exact27717RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27716.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27716.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority27716.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27725

namespace LeftBound27726
def owner : Owner := ⟨.program ⟨257⟩, ⟨48721⟩⟩
def transferEvent : Nat := 27726
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 27725) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 27725)
      LeftBound27725.bound (LeftBound27725.actual selector witness) := by
  exact .transfer (LeftBound27725.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound27725.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound27725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound27725.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27726

namespace LeftBound27821
def owner : Owner := ⟨.program ⟨257⟩, ⟨48079⟩⟩
def transferEvent : Nat := 27821
def frameStart : Nat := 27782
def rule : BoundRule := .identity (.predecessor 0 27820 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27820 .coefficient)
      LeftAuthority27818.bound (LeftAuthority27818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27818.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27818.derived selector witness)

def rawBound : CoeffClass := LeftAuthority27818.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority27818.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27821

namespace LeftBound27838
def owner : Owner := ⟨.program ⟨257⟩, ⟨49470⟩⟩
def transferEvent : Nat := 27838
def frameStart : Nat := 27782
def rule : BoundRule := .sum [.predecessor 0 27836 .coefficient, .predecessor 1 27837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27836 .coefficient)
      LeftBound27821.bound (LeftBound27821.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27837 .coefficient)
      LeftAuthority27834.bound (LeftAuthority27834.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27821.bound, LeftAuthority27834.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27821.bound, LeftAuthority27834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27821.actual selector witness, LeftAuthority27834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27838

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
