import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1821

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound269341
def owner : Owner := ⟨.program ⟨257⟩, ⟨30416⟩⟩
def transferEvent : Nat := 269341
def frameStart : Nat := 269276
def rule : BoundRule := .product (.predecessor 0 269339 .coefficient) (.predecessor 1 269340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269339 .coefficient)
      LeftAuthority269337.bound (LeftAuthority269337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269340 .coefficient)
      LeftBound269335.bound (LeftBound269335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269335.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority269337.bound LeftBound269335.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269337.bound, LeftBound269335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority269337.actual selector witness) * (LeftBound269335.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269341

namespace LeftBound269349
def owner : Owner := ⟨.program ⟨257⟩, ⟨30417⟩⟩
def transferEvent : Nat := 269349
def frameStart : Nat := 269276
def rule : BoundRule := .sum [.predecessor 0 269347 .coefficient, .predecessor 1 269348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269347 .coefficient)
      LeftAuthority269345.bound (LeftAuthority269345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269348 .coefficient)
      LeftBound269341.bound (LeftBound269341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority269345.bound, LeftBound269341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269345.bound, LeftBound269341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority269345.actual selector witness, LeftBound269341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269349

namespace LeftBound269353
def owner : Owner := ⟨.program ⟨257⟩, ⟨30763⟩⟩
def transferEvent : Nat := 269353
def frameStart : Nat := 269276
def rule : BoundRule := .product (.predecessor 0 269351 .coefficient) (.predecessor 1 269352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269351 .coefficient)
      LeftBound269349.bound (LeftBound269349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269352 .coefficient)
      LeftAuthority269326.bound (LeftAuthority269326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound269349.bound LeftAuthority269326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269349.bound, LeftAuthority269326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound269349.actual selector witness) * (LeftAuthority269326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269353

namespace LeftBound269364
def owner : Owner := ⟨.program ⟨257⟩, ⟨29193⟩⟩
def transferEvent : Nat := 269364
def frameStart : Nat := 269276
def rule : BoundRule := .product (.predecessor 0 269362 .coefficient) (.predecessor 1 269363 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269362 .coefficient)
      LeftAuthority269337.bound (LeftAuthority269337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269363 .coefficient)
      LeftAuthority269360.bound (LeftAuthority269360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority269337.bound LeftAuthority269360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269337.bound, LeftAuthority269360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority269337.actual selector witness) * (LeftAuthority269360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269364

namespace LeftBound269372
def owner : Owner := ⟨.program ⟨257⟩, ⟨29194⟩⟩
def transferEvent : Nat := 269372
def frameStart : Nat := 269276
def rule : BoundRule := .sum [.predecessor 0 269370 .coefficient, .predecessor 1 269371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269370 .coefficient)
      LeftAuthority269368.bound (LeftAuthority269368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority269368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority269368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269371 .coefficient)
      LeftBound269364.bound (LeftBound269364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority269368.bound, LeftBound269364.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority269368.bound, LeftBound269364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority269368.actual selector witness, LeftBound269364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269372

namespace LeftBound269376
def owner : Owner := ⟨.program ⟨257⟩, ⟨30766⟩⟩
def transferEvent : Nat := 269376
def frameStart : Nat := 269276
def rule : BoundRule := .sum [.predecessor 0 269374 .coefficient, .predecessor 1 269375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269374 .coefficient)
      LeftBound269372.bound (LeftBound269372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269375 .coefficient)
      LeftBound269353.bound (LeftBound269353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269372.bound, LeftBound269353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269372.bound, LeftBound269353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269372.actual selector witness, LeftBound269353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269376

namespace LeftBound269389
def owner : Owner := ⟨.program ⟨257⟩, ⟨30765⟩⟩
def transferEvent : Nat := 269389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 269387 .coefficient, .predecessor 1 269388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269387 .coefficient)
      LeftBound269218.bound (LeftBound269218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269388 .coefficient)
      LeftBound269201.bound (LeftBound269201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1051.exact269208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269218.bound, LeftBound269201.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269218.bound, LeftBound269201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269218.actual selector witness, LeftBound269201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269389

namespace LeftBound269392
def owner : Owner := ⟨.program ⟨257⟩, ⟨30765⟩⟩
def transferEvent : Nat := 269392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 269386 .summary, .result 269208 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269386 .summary)
      LeftBound269220.bound (LeftBound269220.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29673⟩⟩) (rawTerms := some (Proof.Events1052.exact269386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound269220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269208 .summary)
      LeftBound269203.bound (LeftBound269203.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30764⟩⟩) (rawTerms := some (Proof.Events1051.exact269208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound269203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269220.bound, LeftBound269203.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269220.bound, LeftBound269203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269220.actual selector witness, LeftBound269203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269392

namespace LeftBound269416
def owner : Owner := ⟨.program ⟨257⟩, ⟨25897⟩⟩
def transferEvent : Nat := 269416
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 269414 .coefficient) (.predecessor 1 269415 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269414 .coefficient)
      LeftAuthority12970.bound (LeftAuthority12970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269415 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority12970.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12970.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority12970.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound269416

namespace LeftBound269421
def owner : Owner := ⟨.program ⟨257⟩, ⟨7634⟩⟩
def transferEvent : Nat := 269421
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 269419 .coefficient) (.predecessor 1 269420 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269419 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269420 .coefficient)
      LeftBound20586.bound (LeftBound20586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound20586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound20586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound20586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269421

namespace LeftBound269426
def owner : Owner := ⟨.program ⟨257⟩, ⟨25898⟩⟩
def transferEvent : Nat := 269426
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 269424 .coefficient, .predecessor 1 269425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269424 .coefficient)
      LeftBound269421.bound (LeftBound269421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269425 .coefficient)
      LeftBound269416.bound (LeftBound269416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269421.bound, LeftBound269416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269421.bound, LeftBound269416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269421.actual selector witness, LeftBound269416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269426

namespace LeftBound269430
def owner : Owner := ⟨.program ⟨257⟩, ⟨25899⟩⟩
def transferEvent : Nat := 269430
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 269428 .coefficient, .predecessor 1 269429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269428 .coefficient)
      LeftBound269426.bound (LeftBound269426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269429 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound269426.bound, LeftBound20578.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269426.bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound269426.actual selector witness, LeftBound20578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound269430

namespace LeftBound269431
def owner : Owner := ⟨.program ⟨257⟩, ⟨25899⟩⟩
def transferEvent : Nat := 269431
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩ [⟨.result 20579 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20579 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨104⟩⟩) (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20578.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20578.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound269431

namespace LeftBound269436
def owner : Owner := ⟨.program ⟨257⟩, ⟨25900⟩⟩
def transferEvent : Nat := 269436
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 269434 .coefficient) (.predecessor 1 269435 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 269434 .coefficient)
      LeftBound269430.bound (LeftBound269430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1052.exact269433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound269430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound269430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 269435 .coefficient)
      LeftAuthority12973.bound (LeftAuthority12973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound269430.bound LeftAuthority12973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269430.bound, LeftAuthority12973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound269430.actual selector witness) * (LeftAuthority12973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269436

namespace LeftBound269437
def owner : Owner := ⟨.program ⟨257⟩, ⟨25900⟩⟩
def transferEvent : Nat := 269437
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12856⟩⟩], []⟩ [⟨.result 12974 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12974 .coefficient)
      LeftAuthority12973.bound (LeftAuthority12973.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12856⟩⟩) (rawTerms := some (Proof.Events050.exact12974RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12973.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12973.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority12973.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound269437

namespace LeftBound269438
def owner : Owner := ⟨.program ⟨257⟩, ⟨25900⟩⟩
def transferEvent : Nat := 269438
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 269433 .summary) (.transfer 269437) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 269433 .summary)
      LeftBound269431.bound (LeftBound269431.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25899⟩⟩) (rawTerms := some (Proof.Events1052.exact269433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound269431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 269437)
      LeftBound269437.bound (LeftBound269437.actual selector witness) := by
  exact .transfer (LeftBound269437.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound269431.bound LeftBound269437.bound
def bound : CoeffClass := .finite ⟨25559040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound269431.bound, LeftBound269437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound269431.actual selector witness) * (LeftBound269437.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound269438

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
