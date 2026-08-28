import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard192
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard195
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard199
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard203
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard206
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard210
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard213
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard217
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard239

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40723
def owner : Owner := ⟨.program ⟨257⟩, ⟨59195⟩⟩
def transferEvent : Nat := 40723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40721 .coefficient, .predecessor 1 40722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40721 .coefficient)
      LeftBound40718.bound (LeftBound40718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40722 .coefficient)
      LeftBound37799.bound (LeftBound37799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40718.bound, LeftBound37799.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40718.bound, LeftBound37799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40718.actual selector witness, LeftBound37799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40723

namespace LeftBound40724
def owner : Owner := ⟨.program ⟨257⟩, ⟨59195⟩⟩
def transferEvent : Nat := 40724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40720 .summary, .result 37803 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40720 .summary)
      LeftBound40719.bound (LeftBound40719.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56215⟩⟩) (rawTerms := some (Proof.Events159.exact40720RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37803 .summary)
      LeftBound37802.bound (LeftBound37802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59194⟩⟩) (rawTerms := some (Proof.Events147.exact37803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40719.bound, LeftBound37802.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40719.bound, LeftBound37802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40719.actual selector witness, LeftBound37802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40724

namespace LeftBound40728
def owner : Owner := ⟨.program ⟨257⟩, ⟨62175⟩⟩
def transferEvent : Nat := 40728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40726 .coefficient, .predecessor 1 40727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40726 .coefficient)
      LeftBound40723.bound (LeftBound40723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40727 .coefficient)
      LeftBound37317.bound (LeftBound37317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40723.bound, LeftBound37317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40723.bound, LeftBound37317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40723.actual selector witness, LeftBound37317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40728

namespace LeftBound40729
def owner : Owner := ⟨.program ⟨257⟩, ⟨62175⟩⟩
def transferEvent : Nat := 40729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40725 .summary, .result 37321 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40725 .summary)
      LeftBound40724.bound (LeftBound40724.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59195⟩⟩) (rawTerms := some (Proof.Events159.exact40725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 37321 .summary)
      LeftBound37320.bound (LeftBound37320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62174⟩⟩) (rawTerms := some (Proof.Events145.exact37321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40724.bound, LeftBound37320.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40724.bound, LeftBound37320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40724.actual selector witness, LeftBound37320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40729

namespace LeftBound40733
def owner : Owner := ⟨.program ⟨257⟩, ⟨65155⟩⟩
def transferEvent : Nat := 40733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40731 .coefficient, .predecessor 1 40732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40731 .coefficient)
      LeftBound40728.bound (LeftBound40728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40732 .coefficient)
      LeftBound36835.bound (LeftBound36835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40728.bound, LeftBound36835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40728.bound, LeftBound36835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40728.actual selector witness, LeftBound36835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40733

namespace LeftBound40734
def owner : Owner := ⟨.program ⟨257⟩, ⟨65155⟩⟩
def transferEvent : Nat := 40734
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40730 .summary, .result 36839 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40730 .summary)
      LeftBound40729.bound (LeftBound40729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62175⟩⟩) (rawTerms := some (Proof.Events159.exact40730RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36839 .summary)
      LeftBound36838.bound (LeftBound36838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65154⟩⟩) (rawTerms := some (Proof.Events143.exact36839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40729.bound, LeftBound36838.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40729.bound, LeftBound36838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40729.actual selector witness, LeftBound36838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40734

namespace LeftBound40738
def owner : Owner := ⟨.program ⟨257⟩, ⟨70892⟩⟩
def transferEvent : Nat := 40738
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40736 .coefficient, .predecessor 1 40737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40736 .coefficient)
      LeftBound40733.bound (LeftBound40733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40737 .coefficient)
      LeftBound36353.bound (LeftBound36353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40733.bound, LeftBound36353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40733.bound, LeftBound36353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40733.actual selector witness, LeftBound36353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40738

namespace LeftBound40739
def owner : Owner := ⟨.program ⟨257⟩, ⟨70892⟩⟩
def transferEvent : Nat := 40739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40735 .summary, .result 36357 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40735 .summary)
      LeftBound40734.bound (LeftBound40734.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65155⟩⟩) (rawTerms := some (Proof.Events159.exact40735RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36357 .summary)
      LeftBound36356.bound (LeftBound36356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70891⟩⟩) (rawTerms := some (Proof.Events142.exact36357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40734.bound, LeftBound36356.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40734.bound, LeftBound36356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40734.actual selector witness, LeftBound36356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40739

namespace LeftBound40743
def owner : Owner := ⟨.program ⟨257⟩, ⟨70893⟩⟩
def transferEvent : Nat := 40743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40741 .coefficient, .predecessor 1 40742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40741 .coefficient)
      LeftBound40738.bound (LeftBound40738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40742 .coefficient)
      LeftBound35871.bound (LeftBound35871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40738.bound, LeftBound35871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40738.bound, LeftBound35871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40738.actual selector witness, LeftBound35871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40743

namespace LeftBound40744
def owner : Owner := ⟨.program ⟨257⟩, ⟨70893⟩⟩
def transferEvent : Nat := 40744
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40740 .summary, .result 35875 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40740 .summary)
      LeftBound40739.bound (LeftBound40739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70892⟩⟩) (rawTerms := some (Proof.Events159.exact40740RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35875 .summary)
      LeftBound35874.bound (LeftBound35874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28517⟩⟩) (rawTerms := some (Proof.Events140.exact35875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40739.bound, LeftBound35874.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40739.bound, LeftBound35874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40739.actual selector witness, LeftBound35874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40744

namespace LeftBound40748
def owner : Owner := ⟨.program ⟨257⟩, ⟨70894⟩⟩
def transferEvent : Nat := 40748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40746 .coefficient, .predecessor 1 40747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40746 .coefficient)
      LeftBound40743.bound (LeftBound40743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40747 .coefficient)
      LeftBound35389.bound (LeftBound35389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40743.bound, LeftBound35389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40743.bound, LeftBound35389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40743.actual selector witness, LeftBound35389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40748

namespace LeftBound40749
def owner : Owner := ⟨.program ⟨257⟩, ⟨70894⟩⟩
def transferEvent : Nat := 40749
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40745 .summary, .result 35393 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40745 .summary)
      LeftBound40744.bound (LeftBound40744.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70893⟩⟩) (rawTerms := some (Proof.Events159.exact40745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35393 .summary)
      LeftBound35392.bound (LeftBound35392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31197⟩⟩) (rawTerms := some (Proof.Events138.exact35393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40744.bound, LeftBound35392.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40744.bound, LeftBound35392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40744.actual selector witness, LeftBound35392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40749

namespace LeftBound40753
def owner : Owner := ⟨.program ⟨257⟩, ⟨70895⟩⟩
def transferEvent : Nat := 40753
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40751 .coefficient, .predecessor 1 40752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40751 .coefficient)
      LeftBound40748.bound (LeftBound40748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40752 .coefficient)
      LeftBound34907.bound (LeftBound34907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40748.bound, LeftBound34907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40748.bound, LeftBound34907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40748.actual selector witness, LeftBound34907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40753

namespace LeftBound40754
def owner : Owner := ⟨.program ⟨257⟩, ⟨70895⟩⟩
def transferEvent : Nat := 40754
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40750 .summary, .result 34911 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40750 .summary)
      LeftBound40749.bound (LeftBound40749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70894⟩⟩) (rawTerms := some (Proof.Events159.exact40750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34911 .summary)
      LeftBound34910.bound (LeftBound34910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36857⟩⟩) (rawTerms := some (Proof.Events136.exact34911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40749.bound, LeftBound34910.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40749.bound, LeftBound34910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40749.actual selector witness, LeftBound34910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40754

namespace LeftBound40758
def owner : Owner := ⟨.program ⟨257⟩, ⟨70896⟩⟩
def transferEvent : Nat := 40758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40756 .coefficient, .predecessor 1 40757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 40756 .coefficient)
      LeftBound40753.bound (LeftBound40753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 40757 .coefficient)
      LeftBound34425.bound (LeftBound34425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40753.bound, LeftBound34425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40753.bound, LeftBound34425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40753.actual selector witness, LeftBound34425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40758

namespace LeftBound40759
def owner : Owner := ⟨.program ⟨257⟩, ⟨70896⟩⟩
def transferEvent : Nat := 40759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40755 .summary, .result 34429 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 40755 .summary)
      LeftBound40754.bound (LeftBound40754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70895⟩⟩) (rawTerms := some (Proof.Events159.exact40755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34429 .summary)
      LeftBound34428.bound (LeftBound34428.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39537⟩⟩) (rawTerms := some (Proof.Events134.exact34429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40754.bound, LeftBound34428.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40754.bound, LeftBound34428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound40754.actual selector witness, LeftBound34428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40759

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
