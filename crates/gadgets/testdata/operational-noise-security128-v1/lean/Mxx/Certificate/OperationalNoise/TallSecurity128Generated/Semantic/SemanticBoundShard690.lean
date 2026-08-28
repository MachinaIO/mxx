import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard689

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106213
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def transferEvent : Nat := 106213
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43239⟩⟩]⟩ [⟨.result 106205 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 106205 .coefficient)
      LeftAuthority106204.bound (LeftAuthority106204.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43239⟩⟩) (rawTerms := some (Proof.Events414.exact106205RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106204.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106204.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority106204.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106213

namespace LeftBound106214
def owner : Owner := ⟨.program ⟨257⟩, ⟨43242⟩⟩
def transferEvent : Nat := 106214
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 106213) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 106213)
      LeftBound106213.bound (LeftBound106213.actual selector witness) := by
  exact .transfer (LeftBound106213.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound106213.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound106213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound106213.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106214

namespace LeftBound106293
def owner : Owner := ⟨.program ⟨257⟩, ⟨42499⟩⟩
def transferEvent : Nat := 106293
def frameStart : Nat := 106264
def rule : BoundRule := .product (.predecessor 0 106291 .coefficient) (.predecessor 1 106292 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106291 .coefficient)
      LeftAuthority106289.bound (LeftAuthority106289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106292 .coefficient)
      LeftAuthority106286.bound (LeftAuthority106286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106289.bound LeftAuthority106286.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106289.bound, LeftAuthority106286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority106289.actual selector witness) * (LeftAuthority106286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106293

namespace LeftBound106297
def owner : Owner := ⟨.program ⟨257⟩, ⟨42500⟩⟩
def transferEvent : Nat := 106297
def frameStart : Nat := 106264
def rule : BoundRule := .identity (.predecessor 0 106296 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106296 .coefficient)
      LeftBound106293.bound (LeftBound106293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106293.derived selector witness)

def rawBound : CoeffClass := LeftBound106293.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound106293.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106297

namespace LeftBound106314
def owner : Owner := ⟨.program ⟨257⟩, ⟨44070⟩⟩
def transferEvent : Nat := 106314
def frameStart : Nat := 106264
def rule : BoundRule := .sum [.predecessor 0 106312 .coefficient, .predecessor 1 106313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106312 .coefficient)
      LeftBound106297.bound (LeftBound106297.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106313 .coefficient)
      LeftAuthority106310.bound (LeftAuthority106310.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106297.bound, LeftAuthority106310.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106297.bound, LeftAuthority106310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound106297.actual selector witness, LeftAuthority106310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106314

namespace LeftBound106317
def owner : Owner := ⟨.program ⟨257⟩, ⟨44071⟩⟩
def transferEvent : Nat := 106317
def frameStart : Nat := 106264
def rule : BoundRule := .identity (.predecessor 0 106316 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106316 .coefficient)
      LeftBound106314.bound (LeftBound106314.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106314.derived selector witness)

def rawBound : CoeffClass := LeftBound106314.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound106314.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106317

namespace LeftBound106323
def owner : Owner := ⟨.program ⟨257⟩, ⟨44072⟩⟩
def transferEvent : Nat := 106323
def frameStart : Nat := 106264
def rule : BoundRule := .product (.predecessor 0 106321 .coefficient) (.predecessor 1 106322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106321 .coefficient)
      LeftAuthority106319.bound (LeftAuthority106319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106322 .coefficient)
      LeftBound106317.bound (LeftBound106317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106317.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority106319.bound LeftBound106317.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106319.bound, LeftBound106317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority106319.actual selector witness) * (LeftBound106317.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106323

namespace LeftBound106339
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 106339
def frameStart : Nat := 106264
def rule : BoundRule := .scale (.predecessor 0 106337 .coefficient) (.value (.predecessor 1 106338 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106337 .coefficient)
      LeftAuthority106335.bound (LeftAuthority106335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106338 .coefficient)
      LeftAuthority106326.bound (LeftAuthority106326.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106326.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106335.bound LeftAuthority106326.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106335.bound, LeftAuthority106326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority106335.actual selector witness) * (LeftAuthority106326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106339

namespace LeftBound106342
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 106342
def frameStart : Nat := 106264
def rule : BoundRule := .identity (.predecessor 0 106341 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106341 .coefficient)
      LeftAuthority106329.bound (LeftAuthority106329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106329.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106329.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority106329.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106342

namespace LeftBound106346
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 106346
def frameStart : Nat := 106264
def rule : BoundRule := .product (.predecessor 0 106344 .coefficient) (.predecessor 1 106345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106344 .coefficient)
      LeftBound106342.bound (LeftBound106342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106345 .coefficient)
      LeftBound106339.bound (LeftBound106339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound106342.bound LeftBound106339.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106342.bound, LeftBound106339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound106342.actual selector witness) * (LeftBound106339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106346

namespace LeftBound106351
def owner : Owner := ⟨.program ⟨257⟩, ⟨44073⟩⟩
def transferEvent : Nat := 106351
def frameStart : Nat := 106264
def rule : BoundRule := .sum [.predecessor 0 106349 .coefficient, .predecessor 1 106350 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106349 .coefficient)
      LeftBound106346.bound (LeftBound106346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106350 .coefficient)
      LeftBound106323.bound (LeftBound106323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106323.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106346.bound, LeftBound106323.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106346.bound, LeftBound106323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound106346.actual selector witness, LeftBound106323.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106351

namespace LeftBound106355
def owner : Owner := ⟨.program ⟨257⟩, ⟨44313⟩⟩
def transferEvent : Nat := 106355
def frameStart : Nat := 106264
def rule : BoundRule := .product (.predecessor 0 106353 .coefficient) (.predecessor 1 106354 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106353 .coefficient)
      LeftBound106351.bound (LeftBound106351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106354 .coefficient)
      LeftAuthority106308.bound (LeftAuthority106308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106308.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound106351.bound LeftAuthority106308.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106351.bound, LeftAuthority106308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound106351.actual selector witness) * (LeftAuthority106308.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106355

namespace LeftBound106366
def owner : Owner := ⟨.program ⟨257⟩, ⟨42798⟩⟩
def transferEvent : Nat := 106366
def frameStart : Nat := 106264
def rule : BoundRule := .product (.predecessor 0 106364 .coefficient) (.predecessor 1 106365 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106364 .coefficient)
      LeftAuthority106319.bound (LeftAuthority106319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106365 .coefficient)
      LeftAuthority106362.bound (LeftAuthority106362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106362.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106319.bound LeftAuthority106362.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106319.bound, LeftAuthority106362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority106319.actual selector witness) * (LeftAuthority106362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106366

namespace LeftBound106374
def owner : Owner := ⟨.program ⟨257⟩, ⟨42799⟩⟩
def transferEvent : Nat := 106374
def frameStart : Nat := 106264
def rule : BoundRule := .sum [.predecessor 0 106372 .coefficient, .predecessor 1 106373 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106372 .coefficient)
      LeftAuthority106370.bound (LeftAuthority106370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106373 .coefficient)
      LeftBound106366.bound (LeftBound106366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106366.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106370.bound, LeftBound106366.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106370.bound, LeftBound106366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority106370.actual selector witness, LeftBound106366.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106374

namespace LeftBound106378
def owner : Owner := ⟨.program ⟨257⟩, ⟨44314⟩⟩
def transferEvent : Nat := 106378
def frameStart : Nat := 106264
def rule : BoundRule := .sum [.predecessor 0 106376 .coefficient, .predecessor 1 106377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106376 .coefficient)
      LeftBound106374.bound (LeftBound106374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106377 .coefficient)
      LeftBound106355.bound (LeftBound106355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106374.bound, LeftBound106355.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106374.bound, LeftBound106355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound106374.actual selector witness, LeftBound106355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106378

namespace LeftBound106391
def owner : Owner := ⟨.program ⟨257⟩, ⟨44312⟩⟩
def transferEvent : Nat := 106391
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106389 .coefficient, .predecessor 1 106390 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106389 .coefficient)
      LeftBound106212.bound (LeftBound106212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106390 .coefficient)
      LeftBound106195.bound (LeftBound106195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106212.bound, LeftBound106195.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106212.bound, LeftBound106195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound106212.actual selector witness, LeftBound106195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106391

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
