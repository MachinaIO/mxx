import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1267
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1268
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1269
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1270
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1271
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1272
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1273
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1274
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1275
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1283

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound192446
def owner : Owner := ⟨.program ⟨257⟩, ⟨56023⟩⟩
def transferEvent : Nat := 192446
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192444 .coefficient, .predecessor 1 192445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192444 .coefficient)
      LeftBound192441.bound (LeftBound192441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192445 .coefficient)
      LeftBound191324.bound (LeftBound191324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events747.exact191331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192441.bound, LeftBound191324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192441.bound, LeftBound191324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192441.actual selector witness, LeftBound191324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192446

namespace LeftBound192447
def owner : Owner := ⟨.program ⟨257⟩, ⟨56023⟩⟩
def transferEvent : Nat := 192447
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192443 .summary, .result 191331 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192443 .summary)
      LeftBound192442.bound (LeftBound192442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53043⟩⟩) (rawTerms := some (Proof.Events751.exact192443RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 191331 .summary)
      LeftBound191326.bound (LeftBound191326.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56022⟩⟩) (rawTerms := some (Proof.Events747.exact191331RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound191326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192442.bound, LeftBound191326.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192442.bound, LeftBound191326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192442.actual selector witness, LeftBound191326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192447

namespace LeftBound192451
def owner : Owner := ⟨.program ⟨257⟩, ⟨59003⟩⟩
def transferEvent : Nat := 192451
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192449 .coefficient, .predecessor 1 192450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192449 .coefficient)
      LeftBound192446.bound (LeftBound192446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192450 .coefficient)
      LeftBound191112.bound (LeftBound191112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events746.exact191119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192446.bound, LeftBound191112.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192446.bound, LeftBound191112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192446.actual selector witness, LeftBound191112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192451

namespace LeftBound192452
def owner : Owner := ⟨.program ⟨257⟩, ⟨59003⟩⟩
def transferEvent : Nat := 192452
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192448 .summary, .result 191119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192448 .summary)
      LeftBound192447.bound (LeftBound192447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56023⟩⟩) (rawTerms := some (Proof.Events751.exact192448RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 191119 .summary)
      LeftBound191114.bound (LeftBound191114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59002⟩⟩) (rawTerms := some (Proof.Events746.exact191119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound191114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192447.bound, LeftBound191114.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192447.bound, LeftBound191114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192447.actual selector witness, LeftBound191114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192452

namespace LeftBound192456
def owner : Owner := ⟨.program ⟨257⟩, ⟨61983⟩⟩
def transferEvent : Nat := 192456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192454 .coefficient, .predecessor 1 192455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192454 .coefficient)
      LeftBound192451.bound (LeftBound192451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192455 .coefficient)
      LeftBound190900.bound (LeftBound190900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events745.exact190907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound190900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound190900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192451.bound, LeftBound190900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192451.bound, LeftBound190900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192451.actual selector witness, LeftBound190900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192456

namespace LeftBound192457
def owner : Owner := ⟨.program ⟨257⟩, ⟨61983⟩⟩
def transferEvent : Nat := 192457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192453 .summary, .result 190907 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192453 .summary)
      LeftBound192452.bound (LeftBound192452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59003⟩⟩) (rawTerms := some (Proof.Events751.exact192453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 190907 .summary)
      LeftBound190902.bound (LeftBound190902.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61982⟩⟩) (rawTerms := some (Proof.Events745.exact190907RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound190902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192452.bound, LeftBound190902.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192452.bound, LeftBound190902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192452.actual selector witness, LeftBound190902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192457

namespace LeftBound192461
def owner : Owner := ⟨.program ⟨257⟩, ⟨64963⟩⟩
def transferEvent : Nat := 192461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192459 .coefficient, .predecessor 1 192460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192459 .coefficient)
      LeftBound192456.bound (LeftBound192456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192460 .coefficient)
      LeftBound190688.bound (LeftBound190688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events744.exact190695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound190688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound190688.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192456.bound, LeftBound190688.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192456.bound, LeftBound190688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192456.actual selector witness, LeftBound190688.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192461

namespace LeftBound192462
def owner : Owner := ⟨.program ⟨257⟩, ⟨64963⟩⟩
def transferEvent : Nat := 192462
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192458 .summary, .result 190695 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192458 .summary)
      LeftBound192457.bound (LeftBound192457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61983⟩⟩) (rawTerms := some (Proof.Events751.exact192458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 190695 .summary)
      LeftBound190690.bound (LeftBound190690.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64962⟩⟩) (rawTerms := some (Proof.Events744.exact190695RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound190690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192457.bound, LeftBound190690.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192457.bound, LeftBound190690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192457.actual selector witness, LeftBound190690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192462

namespace LeftBound192466
def owner : Owner := ⟨.program ⟨257⟩, ⟨70404⟩⟩
def transferEvent : Nat := 192466
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192464 .coefficient, .predecessor 1 192465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192464 .coefficient)
      LeftBound192461.bound (LeftBound192461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192465 .coefficient)
      LeftBound190476.bound (LeftBound190476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events744.exact190483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound190476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound190476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192461.bound, LeftBound190476.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192461.bound, LeftBound190476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192461.actual selector witness, LeftBound190476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192466

namespace LeftBound192467
def owner : Owner := ⟨.program ⟨257⟩, ⟨70404⟩⟩
def transferEvent : Nat := 192467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192463 .summary, .result 190483 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192463 .summary)
      LeftBound192462.bound (LeftBound192462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64963⟩⟩) (rawTerms := some (Proof.Events751.exact192463RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 190483 .summary)
      LeftBound190478.bound (LeftBound190478.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70403⟩⟩) (rawTerms := some (Proof.Events744.exact190483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound190478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192462.bound, LeftBound190478.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192462.bound, LeftBound190478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192462.actual selector witness, LeftBound190478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192467

namespace LeftBound192471
def owner : Owner := ⟨.program ⟨257⟩, ⟨70405⟩⟩
def transferEvent : Nat := 192471
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192469 .coefficient, .predecessor 1 192470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192469 .coefficient)
      LeftBound192466.bound (LeftBound192466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192470 .coefficient)
      LeftBound190264.bound (LeftBound190264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events743.exact190271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound190264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound190264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192466.bound, LeftBound190264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192466.bound, LeftBound190264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192466.actual selector witness, LeftBound190264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192471

namespace LeftBound192472
def owner : Owner := ⟨.program ⟨257⟩, ⟨70405⟩⟩
def transferEvent : Nat := 192472
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192468 .summary, .result 190271 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192468 .summary)
      LeftBound192467.bound (LeftBound192467.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70404⟩⟩) (rawTerms := some (Proof.Events751.exact192468RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 190271 .summary)
      LeftBound190266.bound (LeftBound190266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28362⟩⟩) (rawTerms := some (Proof.Events743.exact190271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound190266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192467.bound, LeftBound190266.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192467.bound, LeftBound190266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192467.actual selector witness, LeftBound190266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192472

namespace LeftBound192476
def owner : Owner := ⟨.program ⟨257⟩, ⟨70406⟩⟩
def transferEvent : Nat := 192476
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192474 .coefficient, .predecessor 1 192475 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192474 .coefficient)
      LeftBound192471.bound (LeftBound192471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192475 .coefficient)
      LeftBound190052.bound (LeftBound190052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events742.exact190059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound190052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound190052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192471.bound, LeftBound190052.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192471.bound, LeftBound190052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192471.actual selector witness, LeftBound190052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192476

namespace LeftBound192477
def owner : Owner := ⟨.program ⟨257⟩, ⟨70406⟩⟩
def transferEvent : Nat := 192477
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192473 .summary, .result 190059 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192473 .summary)
      LeftBound192472.bound (LeftBound192472.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70405⟩⟩) (rawTerms := some (Proof.Events751.exact192473RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 190059 .summary)
      LeftBound190054.bound (LeftBound190054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31042⟩⟩) (rawTerms := some (Proof.Events742.exact190059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound190054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192472.bound, LeftBound190054.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192472.bound, LeftBound190054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192472.actual selector witness, LeftBound190054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192477

namespace LeftBound192481
def owner : Owner := ⟨.program ⟨257⟩, ⟨70407⟩⟩
def transferEvent : Nat := 192481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 192479 .coefficient, .predecessor 1 192480 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 192479 .coefficient)
      LeftBound192476.bound (LeftBound192476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events751.exact192478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 192480 .coefficient)
      LeftBound189840.bound (LeftBound189840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events741.exact189847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound189840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound189840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192476.bound, LeftBound189840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192476.bound, LeftBound189840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192476.actual selector witness, LeftBound189840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192481

namespace LeftBound192482
def owner : Owner := ⟨.program ⟨257⟩, ⟨70407⟩⟩
def transferEvent : Nat := 192482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 192478 .summary, .result 189847 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192478 .summary)
      LeftBound192477.bound (LeftBound192477.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70406⟩⟩) (rawTerms := some (Proof.Events751.exact192478RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 189847 .summary)
      LeftBound189842.bound (LeftBound189842.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36702⟩⟩) (rawTerms := some (Proof.Events741.exact189847RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound189842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound192477.bound, LeftBound189842.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192477.bound, LeftBound189842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound192477.actual selector witness, LeftBound189842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound192482

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
