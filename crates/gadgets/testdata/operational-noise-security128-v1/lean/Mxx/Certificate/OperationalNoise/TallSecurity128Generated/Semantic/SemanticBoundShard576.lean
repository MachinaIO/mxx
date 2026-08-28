import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard575

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90159
def owner : Owner := ⟨.program ⟨257⟩, ⟨10371⟩⟩
def transferEvent : Nat := 90159
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
end LeftBound90159

namespace LeftBound90160
def owner : Owner := ⟨.program ⟨257⟩, ⟨10371⟩⟩
def transferEvent : Nat := 90160
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90155 .summary) (.transfer 90159) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90155 .summary)
      LeftBound90153.bound (LeftBound90153.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10370⟩⟩) (rawTerms := some (Proof.Events352.exact90155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90159)
      LeftBound90159.bound (LeftBound90159.actual selector witness) := by
  exact .transfer (LeftBound90159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90153.bound LeftBound90159.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90153.bound, LeftBound90159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90153.actual selector witness) * (LeftBound90159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90160

namespace LeftBound90186
def owner : Owner := ⟨.program ⟨257⟩, ⟨71444⟩⟩
def transferEvent : Nat := 90186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90184 .coefficient, .predecessor 1 90185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90184 .coefficient)
      LeftBound90158.bound (LeftBound90158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90185 .coefficient)
      LeftBound90136.bound (LeftBound90136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90158.bound, LeftBound90136.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90158.bound, LeftBound90136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90158.actual selector witness, LeftBound90136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90186

namespace LeftBound90206
def owner : Owner := ⟨.program ⟨257⟩, ⟨71444⟩⟩
def transferEvent : Nat := 90206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90183 .summary, .result 90138 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90183 .summary)
      LeftBound90160.bound (LeftBound90160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10371⟩⟩) (rawTerms := some (Proof.Events352.exact90183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90138 .summary)
      LeftBound90137.bound (LeftBound90137.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71443⟩⟩) (rawTerms := some (Proof.Events352.exact90138RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90137.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90160.bound, LeftBound90137.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90160.bound, LeftBound90137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90160.actual selector witness, LeftBound90137.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90206

namespace LeftBound90210
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def transferEvent : Nat := 90210
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90208 .coefficient) (.predecessor 1 90209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90208 .coefficient)
      LeftBound90186.bound (LeftBound90186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90209 .coefficient)
      LeftBound16143.bound (LeftBound16143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90186.bound LeftBound16143.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90186.bound, LeftBound16143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90186.actual selector witness) * (LeftBound16143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90210

namespace LeftBound90211
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def transferEvent : Nat := 90211
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩ [⟨.result 16140 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16140 .coefficient)
      LeftAuthority16139.bound (LeftAuthority16139.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9499⟩⟩) (rawTerms := some (Proof.Events063.exact16140RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16139.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16139.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16139.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90211

namespace LeftBound90212
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def transferEvent : Nat := 90212
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90207 .summary) (.transfer 90211) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90207 .summary)
      LeftBound90206.bound (LeftBound90206.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71444⟩⟩) (rawTerms := some (Proof.Events352.exact90207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90211)
      LeftBound90211.bound (LeftBound90211.actual selector witness) := by
  exact .transfer (LeftBound90211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90206.bound LeftBound90211.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167293655866246524336865280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90206.bound, LeftBound90211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90206.actual selector witness) * (LeftBound90211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90212

namespace LeftBound90274
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def transferEvent : Nat := 90274
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90272 .coefficient, .predecessor 1 90273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90272 .coefficient)
      LeftBound90210.bound (LeftBound90210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90273 .coefficient)
      LeftBound75791.bound (LeftBound75791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90210.bound, LeftBound75791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90210.bound, LeftBound75791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90210.actual selector witness, LeftBound75791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90274

namespace LeftBound90294
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def transferEvent : Nat := 90294
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90271 .summary, .result 75868 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90271 .summary)
      LeftBound90212.bound (LeftBound90212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71445⟩⟩) (rawTerms := some (Proof.Events352.exact90271RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75868 .summary)
      LeftBound75829.bound (LeftBound75829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67592⟩⟩) (rawTerms := some (Proof.Events296.exact75868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75829.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90212.bound, LeftBound75829.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90212.bound, LeftBound75829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90212.actual selector witness, LeftBound75829.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90294

namespace LeftBound90298
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def transferEvent : Nat := 90298
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90296 .coefficient) (.predecessor 1 90297 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90296 .coefficient)
      LeftBound90274.bound (LeftBound90274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90297 .coefficient)
      LeftBound16133.bound (LeftBound16133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16133.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90274.bound LeftBound16133.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90274.bound, LeftBound16133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90274.actual selector witness) * (LeftBound16133.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90298

namespace LeftBound90299
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def transferEvent : Nat := 90299
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7101⟩⟩]⟩ [⟨.result 16130 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16130 .coefficient)
      LeftAuthority16129.bound (LeftAuthority16129.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7101⟩⟩) (rawTerms := some (Proof.Events063.exact16130RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16129.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16129.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16129.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90299

namespace LeftBound90300
def owner : Owner := ⟨.program ⟨257⟩, ⟨71447⟩⟩
def transferEvent : Nat := 90300
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90295 .summary) (.transfer 90299) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90295 .summary)
      LeftBound90294.bound (LeftBound90294.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71446⟩⟩) (rawTerms := some (Proof.Events352.exact90295RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90299)
      LeftBound90299.bound (LeftBound90299.actual selector witness) := by
  exact .transfer (LeftBound90299.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90294.bound LeftBound90299.bound
def bound : CoeffClass := .finite ⟨7702113697398803698856913678033037845150209519672183236728648848208035840, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90294.bound, LeftBound90299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90294.actual selector witness) * (LeftBound90299.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90300

namespace LeftBound90381
def owner : Owner := ⟨.program ⟨257⟩, ⟨10013⟩⟩
def transferEvent : Nat := 90381
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 90376 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90376 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90381

namespace LeftBound90385
def owner : Owner := ⟨.program ⟨257⟩, ⟨10014⟩⟩
def transferEvent : Nat := 90385
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90383 .coefficient) (.predecessor 1 90384 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90383 .coefficient)
      LeftBound90381.bound (LeftBound90381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90384 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90381.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90381.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90381.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90385

namespace LeftBound90397
def owner : Owner := ⟨.program ⟨257⟩, ⟨9903⟩⟩
def transferEvent : Nat := 90397
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 90392 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90392 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90397

namespace LeftBound90401
def owner : Owner := ⟨.program ⟨257⟩, ⟨9905⟩⟩
def transferEvent : Nat := 90401
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90399 .coefficient) (.predecessor 1 90400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90399 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90400 .coefficient)
      LeftAuthority16176.bound (LeftAuthority16176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftAuthority16176.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftAuthority16176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftAuthority16176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90401

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
