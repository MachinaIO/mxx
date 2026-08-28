import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1346
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1379

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound206349
def owner : Owner := ⟨.program ⟨257⟩, ⟨32143⟩⟩
def transferEvent : Nat := 206349
def frameStart : Nat := 206253
def rule : BoundRule := .sum [.predecessor 0 206347 .coefficient, .predecessor 1 206348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206347 .coefficient)
      LeftAuthority206345.bound (LeftAuthority206345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206348 .coefficient)
      LeftBound206341.bound (LeftBound206341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority206345.bound, LeftBound206341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority206345.bound, LeftBound206341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority206345.actual selector witness, LeftBound206341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound206349

namespace LeftBound206353
def owner : Owner := ⟨.program ⟨257⟩, ⟨33953⟩⟩
def transferEvent : Nat := 206353
def frameStart : Nat := 206253
def rule : BoundRule := .sum [.predecessor 0 206351 .coefficient, .predecessor 1 206352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206351 .coefficient)
      LeftBound206349.bound (LeftBound206349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206352 .coefficient)
      LeftBound206330.bound (LeftBound206330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events805.exact206335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound206349.bound, LeftBound206330.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206349.bound, LeftBound206330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound206349.actual selector witness, LeftBound206330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound206353

namespace LeftBound206366
def owner : Owner := ⟨.program ⟨257⟩, ⟨33950⟩⟩
def transferEvent : Nat := 206366
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 206364 .coefficient, .predecessor 1 206365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206364 .coefficient)
      LeftBound206195.bound (LeftBound206195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206365 .coefficient)
      LeftBound206178.bound (LeftBound206178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events805.exact206185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound206195.bound, LeftBound206178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206195.bound, LeftBound206178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound206195.actual selector witness, LeftBound206178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound206366

namespace LeftBound206369
def owner : Owner := ⟨.program ⟨257⟩, ⟨33950⟩⟩
def transferEvent : Nat := 206369
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 206363 .summary, .result 206185 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206363 .summary)
      LeftBound206197.bound (LeftBound206197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32735⟩⟩) (rawTerms := some (Proof.Events806.exact206363RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206185 .summary)
      LeftBound206180.bound (LeftBound206180.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33949⟩⟩) (rawTerms := some (Proof.Events805.exact206185RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound206197.bound, LeftBound206180.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206197.bound, LeftBound206180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound206197.actual selector witness, LeftBound206180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound206369

namespace LeftBound206373
def owner : Owner := ⟨.program ⟨257⟩, ⟨33951⟩⟩
def transferEvent : Nat := 206373
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 206371 .coefficient) (.predecessor 1 206372 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206371 .coefficient)
      LeftBound206366.bound (LeftBound206366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206372 .coefficient)
      LeftBound15821.bound (LeftBound15821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound206366.bound LeftBound15821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206366.bound, LeftBound15821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound206366.actual selector witness) * (LeftBound15821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206373

namespace LeftBound206374
def owner : Owner := ⟨.program ⟨257⟩, ⟨33951⟩⟩
def transferEvent : Nat := 206374
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩ [⟨.result 15818 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15818 .coefficient)
      LeftAuthority15817.bound (LeftAuthority15817.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7145⟩⟩) (rawTerms := some (Proof.Events061.exact15818RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15817.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15817.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15817.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound206374

namespace LeftBound206375
def owner : Owner := ⟨.program ⟨257⟩, ⟨33951⟩⟩
def transferEvent : Nat := 206375
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 206370 .summary) (.transfer 206374) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206370 .summary)
      LeftBound206369.bound (LeftBound206369.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33950⟩⟩) (rawTerms := some (Proof.Events806.exact206370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound206369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 206374)
      LeftBound206374.bound (LeftBound206374.actual selector witness) := by
  exact .transfer (LeftBound206374.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound206369.bound LeftBound206374.bound
def bound : CoeffClass := .finite ⟨345628904428363669605693235694606923857920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206369.bound, LeftBound206374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound206369.actual selector witness) * (LeftBound206374.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206375

namespace LeftBound206390
def owner : Owner := ⟨.program ⟨257⟩, ⟨23929⟩⟩
def transferEvent : Nat := 206390
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 206388 .coefficient) (.predecessor 1 206389 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206388 .coefficient)
      LeftBound200407.bound (LeftBound200407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206389 .coefficient)
      LeftAuthority206386.bound (LeftAuthority206386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206386.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200407.bound LeftAuthority206386.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200407.bound, LeftAuthority206386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200407.actual selector witness) * (LeftAuthority206386.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206390

namespace LeftBound206391
def owner : Owner := ⟨.program ⟨257⟩, ⟨23929⟩⟩
def transferEvent : Nat := 206391
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23927⟩⟩]⟩ [⟨.result 206387 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206387 .coefficient)
      LeftAuthority206386.bound (LeftAuthority206386.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23927⟩⟩) (rawTerms := some (Proof.Events806.exact206387RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206386.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority206386.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority206386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority206386.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound206391

namespace LeftBound206392
def owner : Owner := ⟨.program ⟨257⟩, ⟨23929⟩⟩
def transferEvent : Nat := 206392
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 200411 .summary) (.transfer 206391) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200411 .summary)
      LeftBound200410.bound (LeftBound200410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23463⟩⟩) (rawTerms := some (Proof.Events782.exact200411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 206391)
      LeftBound206391.bound (LeftBound206391.actual selector witness) := by
  exact .transfer (LeftBound206391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200410.bound LeftBound206391.bound
def bound : CoeffClass := .finite ⟨32189003662929192193909661368320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200410.bound, LeftBound206391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200410.actual selector witness) * (LeftBound206391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206392

namespace LeftBound206403
def owner : Owner := ⟨.program ⟨257⟩, ⟨22714⟩⟩
def transferEvent : Nat := 206403
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 206401 .coefficient) (.value (.predecessor 1 206402 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206401 .coefficient)
      LeftAuthority206399.bound (LeftAuthority206399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206402 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority206399.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority206399.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority206399.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound206403

namespace LeftBound206407
def owner : Owner := ⟨.program ⟨257⟩, ⟨22715⟩⟩
def transferEvent : Nat := 206407
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 206405 .coefficient) (.predecessor 1 206406 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206405 .coefficient)
      LeftBound192992.bound (LeftBound192992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206406 .coefficient)
      LeftBound206403.bound (LeftBound206403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound206403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound206403.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192992.bound LeftBound206403.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192992.bound, LeftBound206403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192992.actual selector witness) * (LeftBound206403.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206407

namespace LeftBound206408
def owner : Owner := ⟨.program ⟨257⟩, ⟨22715⟩⟩
def transferEvent : Nat := 206408
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22712⟩⟩]⟩ [⟨.result 206400 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 206400 .coefficient)
      LeftAuthority206399.bound (LeftAuthority206399.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22712⟩⟩) (rawTerms := some (Proof.Events806.exact206400RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206399.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority206399.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority206399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority206399.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound206408

namespace LeftBound206409
def owner : Owner := ⟨.program ⟨257⟩, ⟨22715⟩⟩
def transferEvent : Nat := 206409
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 206408) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 206408)
      LeftBound206408.bound (LeftBound206408.actual selector witness) := by
  exact .transfer (LeftBound206408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound206408.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound206408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound206408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound206409

namespace LeftBound206504
def owner : Owner := ⟨.program ⟨257⟩, ⟨21825⟩⟩
def transferEvent : Nat := 206504
def frameStart : Nat := 206465
def rule : BoundRule := .identity (.predecessor 0 206503 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206503 .coefficient)
      LeftAuthority206501.bound (LeftAuthority206501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events806.exact206502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority206501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority206501.derived selector witness)

def rawBound : CoeffClass := LeftAuthority206501.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority206501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority206501.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound206504

namespace LeftBound206521
def owner : Owner := ⟨.program ⟨257⟩, ⟨23294⟩⟩
def transferEvent : Nat := 206521
def frameStart : Nat := 206465
def rule : BoundRule := .sum [.predecessor 0 206519 .coefficient, .predecessor 1 206520 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 206519 .coefficient)
      LeftBound206504.bound (LeftBound206504.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound206504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 206520 .coefficient)
      LeftAuthority206517.bound (LeftAuthority206517.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority206517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound206504.bound, LeftAuthority206517.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound206504.bound, LeftAuthority206517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound206504.actual selector witness, LeftAuthority206517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound206521

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
