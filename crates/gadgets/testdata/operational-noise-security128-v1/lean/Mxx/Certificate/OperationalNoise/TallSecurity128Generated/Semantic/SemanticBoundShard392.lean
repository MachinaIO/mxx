import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard391

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63248
def owner : Owner := ⟨.program ⟨257⟩, ⟨13987⟩⟩
def transferEvent : Nat := 63248
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 63246 .coefficient) (.predecessor 1 63247 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63246 .coefficient)
      LeftAuthority2432.bound (LeftAuthority2432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63247 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2432.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2432.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2432.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63248

namespace LeftBound63253
def owner : Owner := ⟨.program ⟨257⟩, ⟨10780⟩⟩
def transferEvent : Nat := 63253
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63251 .coefficient) (.predecessor 1 63252 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63251 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63252 .coefficient)
      LeftBound19124.bound (LeftBound19124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound19124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound19124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound19124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63253

namespace LeftBound63258
def owner : Owner := ⟨.program ⟨257⟩, ⟨13988⟩⟩
def transferEvent : Nat := 63258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63256 .coefficient, .predecessor 1 63257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63256 .coefficient)
      LeftBound63253.bound (LeftBound63253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63257 .coefficient)
      LeftBound63248.bound (LeftBound63248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63253.bound, LeftBound63248.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63253.bound, LeftBound63248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63253.actual selector witness, LeftBound63248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63258

namespace LeftBound63262
def owner : Owner := ⟨.program ⟨257⟩, ⟨13989⟩⟩
def transferEvent : Nat := 63262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63260 .coefficient, .predecessor 1 63261 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63260 .coefficient)
      LeftBound63258.bound (LeftBound63258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63261 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63258.bound, LeftBound19116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63258.bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63258.actual selector witness, LeftBound19116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63262

namespace LeftBound63263
def owner : Owner := ⟨.program ⟨257⟩, ⟨13989⟩⟩
def transferEvent : Nat := 63263
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨124⟩⟩]⟩ [⟨.result 19117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19117 .coefficient)
      LeftBound19116.bound (LeftBound19116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨124⟩⟩) (rawTerms := some (Proof.Events074.exact19117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19116.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63263

namespace LeftBound63268
def owner : Owner := ⟨.program ⟨257⟩, ⟨13990⟩⟩
def transferEvent : Nat := 63268
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63266 .coefficient) (.predecessor 1 63267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63266 .coefficient)
      LeftBound63262.bound (LeftBound63262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63267 .coefficient)
      LeftBound19113.bound (LeftBound19113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63262.bound LeftBound19113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63262.bound, LeftBound19113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63262.actual selector witness) * (LeftBound19113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63268

namespace LeftBound63269
def owner : Owner := ⟨.program ⟨257⟩, ⟨13990⟩⟩
def transferEvent : Nat := 63269
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩ [⟨.result 19110 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19110 .coefficient)
      LeftAuthority19109.bound (LeftAuthority19109.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9553⟩⟩) (rawTerms := some (Proof.Events074.exact19110RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19109.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19109.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority19109.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63269

namespace LeftBound63270
def owner : Owner := ⟨.program ⟨257⟩, ⟨13990⟩⟩
def transferEvent : Nat := 63270
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63265 .summary) (.transfer 63269) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63265 .summary)
      LeftBound63263.bound (LeftBound63263.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13989⟩⟩) (rawTerms := some (Proof.Events247.exact63265RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 63269)
      LeftBound63269.bound (LeftBound63269.actual selector witness) := by
  exact .transfer (LeftBound63269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63263.bound LeftBound63269.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63263.bound, LeftBound63269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63263.actual selector witness) * (LeftBound63269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63270

namespace LeftBound63278
def owner : Owner := ⟨.program ⟨257⟩, ⟨37289⟩⟩
def transferEvent : Nat := 63278
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63276 .coefficient, .predecessor 1 63277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63276 .coefficient)
      LeftBound63268.bound (LeftBound63268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63277 .coefficient)
      LeftBound63240.bound (LeftBound63240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63240.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63268.bound, LeftBound63240.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63268.bound, LeftBound63240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63268.actual selector witness, LeftBound63240.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63278

namespace LeftBound63280
def owner : Owner := ⟨.program ⟨257⟩, ⟨37289⟩⟩
def transferEvent : Nat := 63280
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63275 .summary, .result 63245 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63275 .summary)
      LeftBound63270.bound (LeftBound63270.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13990⟩⟩) (rawTerms := some (Proof.Events247.exact63275RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63245 .summary)
      LeftBound63242.bound (LeftBound63242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37288⟩⟩) (rawTerms := some (Proof.Events247.exact63245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63242.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63270.bound, LeftBound63242.bound]
def bound : CoeffClass := .finite ⟨279208656896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63270.bound, LeftBound63242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63270.actual selector witness, LeftBound63242.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63280

namespace LeftBound63284
def owner : Owner := ⟨.program ⟨257⟩, ⟨39017⟩⟩
def transferEvent : Nat := 63284
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63282 .coefficient) (.predecessor 1 63283 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63282 .coefficient)
      LeftBound63278.bound (LeftBound63278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63283 .coefficient)
      LeftAuthority63216.bound (LeftAuthority63216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63216.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63278.bound LeftAuthority63216.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63278.bound, LeftAuthority63216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63278.actual selector witness) * (LeftAuthority63216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63284

namespace LeftBound63285
def owner : Owner := ⟨.program ⟨257⟩, ⟨39017⟩⟩
def transferEvent : Nat := 63285
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39016⟩⟩]⟩ [⟨.result 63217 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63217 .coefficient)
      LeftAuthority63216.bound (LeftAuthority63216.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39016⟩⟩) (rawTerms := some (Proof.Events246.exact63217RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63216.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63216.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority63216.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63285

namespace LeftBound63286
def owner : Owner := ⟨.program ⟨257⟩, ⟨39017⟩⟩
def transferEvent : Nat := 63286
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63281 .summary) (.transfer 63285) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63281 .summary)
      LeftBound63280.bound (LeftBound63280.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37289⟩⟩) (rawTerms := some (Proof.Events247.exact63281RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 63285)
      LeftBound63285.bound (LeftBound63285.actual selector witness) := by
  exact .transfer (LeftBound63285.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63280.bound LeftBound63285.bound
def bound : CoeffClass := .finite ⟨2997980125321012183040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63280.bound, LeftBound63285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63280.actual selector witness) * (LeftBound63285.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63286

namespace LeftBound63297
def owner : Owner := ⟨.program ⟨257⟩, ⟨37941⟩⟩
def transferEvent : Nat := 63297
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63295 .coefficient) (.value (.predecessor 1 63296 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63295 .coefficient)
      LeftAuthority63293.bound (LeftAuthority63293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63296 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63293.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63293.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority63293.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63297

namespace LeftBound63301
def owner : Owner := ⟨.program ⟨257⟩, ⟨37942⟩⟩
def transferEvent : Nat := 63301
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63299 .coefficient) (.predecessor 1 63300 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63299 .coefficient)
      LeftBound61367.bound (LeftBound61367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63300 .coefficient)
      LeftBound63297.bound (LeftBound63297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63297.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61367.bound LeftBound63297.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61367.bound, LeftBound63297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61367.actual selector witness) * (LeftBound63297.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63301

namespace LeftBound63302
def owner : Owner := ⟨.program ⟨257⟩, ⟨37942⟩⟩
def transferEvent : Nat := 63302
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨37939⟩⟩]⟩ [⟨.result 63294 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63294 .coefficient)
      LeftAuthority63293.bound (LeftAuthority63293.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨37939⟩⟩) (rawTerms := some (Proof.Events247.exact63294RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63293.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63293.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority63293.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63302

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
