import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1696
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1697
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1704

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound252601
def owner : Owner := ⟨.program ⟨257⟩, ⟨44049⟩⟩
def transferEvent : Nat := 252601
def frameStart : Nat := 252514
def rule : BoundRule := .sum [.predecessor 0 252599 .coefficient, .predecessor 1 252600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252599 .coefficient)
      LeftBound252596.bound (LeftBound252596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252600 .coefficient)
      LeftBound252573.bound (LeftBound252573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound252596.bound, LeftBound252573.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252596.bound, LeftBound252573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound252596.actual selector witness, LeftBound252573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252601

namespace LeftBound252605
def owner : Owner := ⟨.program ⟨257⟩, ⟨44247⟩⟩
def transferEvent : Nat := 252605
def frameStart : Nat := 252514
def rule : BoundRule := .product (.predecessor 0 252603 .coefficient) (.predecessor 1 252604 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252603 .coefficient)
      LeftBound252601.bound (LeftBound252601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252604 .coefficient)
      LeftAuthority252558.bound (LeftAuthority252558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound252601.bound LeftAuthority252558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252601.bound, LeftAuthority252558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound252601.actual selector witness) * (LeftAuthority252558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252605

namespace LeftBound252616
def owner : Owner := ⟨.program ⟨257⟩, ⟨42750⟩⟩
def transferEvent : Nat := 252616
def frameStart : Nat := 252514
def rule : BoundRule := .product (.predecessor 0 252614 .coefficient) (.predecessor 1 252615 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252614 .coefficient)
      LeftAuthority252569.bound (LeftAuthority252569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252615 .coefficient)
      LeftAuthority252612.bound (LeftAuthority252612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority252569.bound LeftAuthority252612.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252569.bound, LeftAuthority252612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority252569.actual selector witness) * (LeftAuthority252612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252616

namespace LeftBound252624
def owner : Owner := ⟨.program ⟨257⟩, ⟨42751⟩⟩
def transferEvent : Nat := 252624
def frameStart : Nat := 252514
def rule : BoundRule := .sum [.predecessor 0 252622 .coefficient, .predecessor 1 252623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252622 .coefficient)
      LeftAuthority252620.bound (LeftAuthority252620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252623 .coefficient)
      LeftBound252616.bound (LeftBound252616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252616.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority252620.bound, LeftBound252616.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252620.bound, LeftBound252616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority252620.actual selector witness, LeftBound252616.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252624

namespace LeftBound252628
def owner : Owner := ⟨.program ⟨257⟩, ⟨44248⟩⟩
def transferEvent : Nat := 252628
def frameStart : Nat := 252514
def rule : BoundRule := .sum [.predecessor 0 252626 .coefficient, .predecessor 1 252627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252626 .coefficient)
      LeftBound252624.bound (LeftBound252624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252627 .coefficient)
      LeftBound252605.bound (LeftBound252605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound252624.bound, LeftBound252605.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252624.bound, LeftBound252605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound252624.actual selector witness, LeftBound252605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252628

namespace LeftBound252641
def owner : Owner := ⟨.program ⟨257⟩, ⟨44246⟩⟩
def transferEvent : Nat := 252641
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 252639 .coefficient, .predecessor 1 252640 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252639 .coefficient)
      LeftBound252462.bound (LeftBound252462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252640 .coefficient)
      LeftBound252445.bound (LeftBound252445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound252462.bound, LeftBound252445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252462.bound, LeftBound252445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound252462.actual selector witness, LeftBound252445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252641

namespace LeftBound252644
def owner : Owner := ⟨.program ⟨257⟩, ⟨44246⟩⟩
def transferEvent : Nat := 252644
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 252638 .summary, .result 252452 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252638 .summary)
      LeftBound252464.bound (LeftBound252464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43182⟩⟩) (rawTerms := some (Proof.Events986.exact252638RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252452 .summary)
      LeftBound252447.bound (LeftBound252447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44245⟩⟩) (rawTerms := some (Proof.Events986.exact252452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound252464.bound, LeftBound252447.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252464.bound, LeftBound252447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound252464.actual selector witness, LeftBound252447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252644

namespace LeftBound252648
def owner : Owner := ⟨.program ⟨257⟩, ⟨44546⟩⟩
def transferEvent : Nat := 252648
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 252646 .coefficient) (.predecessor 1 252647 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252646 .coefficient)
      LeftBound252641.bound (LeftBound252641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252647 .coefficient)
      LeftAuthority252367.bound (LeftAuthority252367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events985.exact252368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound252641.bound LeftAuthority252367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252641.bound, LeftAuthority252367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound252641.actual selector witness) * (LeftAuthority252367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252648

namespace LeftBound252649
def owner : Owner := ⟨.program ⟨257⟩, ⟨44546⟩⟩
def transferEvent : Nat := 252649
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44544⟩⟩]⟩ [⟨.result 252368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252368 .coefficient)
      LeftAuthority252367.bound (LeftAuthority252367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44544⟩⟩) (rawTerms := some (Proof.Events985.exact252368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority252367.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority252367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound252649

namespace LeftBound252650
def owner : Owner := ⟨.program ⟨257⟩, ⟨44546⟩⟩
def transferEvent : Nat := 252650
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 252645 .summary) (.transfer 252649) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252645 .summary)
      LeftBound252644.bound (LeftBound252644.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44246⟩⟩) (rawTerms := some (Proof.Events986.exact252645RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 252649)
      LeftBound252649.bound (LeftBound252649.actual selector witness) := by
  exact .transfer (LeftBound252649.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound252644.bound LeftBound252649.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252644.bound, LeftBound252649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound252644.actual selector witness) * (LeftBound252649.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252650

namespace LeftBound252661
def owner : Owner := ⟨.program ⟨257⟩, ⟨43438⟩⟩
def transferEvent : Nat := 252661
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 252659 .coefficient) (.value (.predecessor 1 252660 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252659 .coefficient)
      LeftAuthority252657.bound (LeftAuthority252657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252660 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority252657.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252657.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority252657.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound252661

namespace LeftBound252665
def owner : Owner := ⟨.program ⟨257⟩, ⟨43439⟩⟩
def transferEvent : Nat := 252665
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 252663 .coefficient) (.predecessor 1 252664 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252663 .coefficient)
      LeftBound251492.bound (LeftBound251492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252664 .coefficient)
      LeftBound252661.bound (LeftBound252661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events986.exact252662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252661.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251492.bound LeftBound252661.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251492.bound, LeftBound252661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251492.actual selector witness) * (LeftBound252661.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252665

namespace LeftBound252666
def owner : Owner := ⟨.program ⟨257⟩, ⟨43439⟩⟩
def transferEvent : Nat := 252666
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43436⟩⟩]⟩ [⟨.result 252658 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252658 .coefficient)
      LeftAuthority252657.bound (LeftAuthority252657.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43436⟩⟩) (rawTerms := some (Proof.Events986.exact252658RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252657.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority252657.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority252657.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound252666

namespace LeftBound252667
def owner : Owner := ⟨.program ⟨257⟩, ⟨43439⟩⟩
def transferEvent : Nat := 252667
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 251495 .summary) (.transfer 252666) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251495 .summary)
      LeftBound251493.bound (LeftBound251493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 252666)
      LeftBound252666.bound (LeftBound252666.actual selector witness) := by
  exact .transfer (LeftBound252666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251493.bound LeftBound252666.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251493.bound, LeftBound252666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251493.actual selector witness) * (LeftBound252666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound252667

namespace LeftBound252762
def owner : Owner := ⟨.program ⟨257⟩, ⟨42749⟩⟩
def transferEvent : Nat := 252762
def frameStart : Nat := 252723
def rule : BoundRule := .identity (.predecessor 0 252761 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252761 .coefficient)
      LeftAuthority252759.bound (LeftAuthority252759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events987.exact252760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority252759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority252759.derived selector witness)

def rawBound : CoeffClass := LeftAuthority252759.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority252759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority252759.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound252762

namespace LeftBound252779
def owner : Owner := ⟨.program ⟨257⟩, ⟨44126⟩⟩
def transferEvent : Nat := 252779
def frameStart : Nat := 252723
def rule : BoundRule := .sum [.predecessor 0 252777 .coefficient, .predecessor 1 252778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 252777 .coefficient)
      LeftBound252762.bound (LeftBound252762.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound252762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 252778 .coefficient)
      LeftAuthority252775.bound (LeftAuthority252775.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority252775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound252762.bound, LeftAuthority252775.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252762.bound, LeftAuthority252775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound252762.actual selector witness, LeftAuthority252775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound252779

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
