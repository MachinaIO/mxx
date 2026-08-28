import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard334

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54461
def owner : Owner := ⟨.program ⟨257⟩, ⟨19232⟩⟩
def transferEvent : Nat := 54461
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨19229⟩⟩]⟩ [⟨.result 54453 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54453 .coefficient)
      LeftAuthority54452.bound (LeftAuthority54452.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨19229⟩⟩) (rawTerms := some (Proof.Events212.exact54453RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54452.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54452.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority54452.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54461

namespace LeftBound54462
def owner : Owner := ⟨.program ⟨257⟩, ⟨19232⟩⟩
def transferEvent : Nat := 54462
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 54461) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 54461)
      LeftBound54461.bound (LeftBound54461.actual selector witness) := by
  exact .transfer (LeftBound54461.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound54461.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound54461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound54461.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54462

namespace LeftBound54541
def owner : Owner := ⟨.program ⟨257⟩, ⟨18467⟩⟩
def transferEvent : Nat := 54541
def frameStart : Nat := 54512
def rule : BoundRule := .product (.predecessor 0 54539 .coefficient) (.predecessor 1 54540 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54539 .coefficient)
      LeftAuthority54537.bound (LeftAuthority54537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54540 .coefficient)
      LeftAuthority54534.bound (LeftAuthority54534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54537.bound LeftAuthority54534.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54537.bound, LeftAuthority54534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority54537.actual selector witness) * (LeftAuthority54534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54541

namespace LeftBound54545
def owner : Owner := ⟨.program ⟨257⟩, ⟨18468⟩⟩
def transferEvent : Nat := 54545
def frameStart : Nat := 54512
def rule : BoundRule := .identity (.predecessor 0 54544 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54544 .coefficient)
      LeftBound54541.bound (LeftBound54541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54541.derived selector witness)

def rawBound : CoeffClass := LeftBound54541.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound54541.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54545

namespace LeftBound54562
def owner : Owner := ⟨.program ⟨257⟩, ⟨20018⟩⟩
def transferEvent : Nat := 54562
def frameStart : Nat := 54512
def rule : BoundRule := .sum [.predecessor 0 54560 .coefficient, .predecessor 1 54561 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54560 .coefficient)
      LeftBound54545.bound (LeftBound54545.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54561 .coefficient)
      LeftAuthority54558.bound (LeftAuthority54558.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54545.bound, LeftAuthority54558.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54545.bound, LeftAuthority54558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54545.actual selector witness, LeftAuthority54558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54562

namespace LeftBound54565
def owner : Owner := ⟨.program ⟨257⟩, ⟨20019⟩⟩
def transferEvent : Nat := 54565
def frameStart : Nat := 54512
def rule : BoundRule := .identity (.predecessor 0 54564 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54564 .coefficient)
      LeftBound54562.bound (LeftBound54562.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54562.derived selector witness)

def rawBound : CoeffClass := LeftBound54562.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound54562.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54565

namespace LeftBound54571
def owner : Owner := ⟨.program ⟨257⟩, ⟨20020⟩⟩
def transferEvent : Nat := 54571
def frameStart : Nat := 54512
def rule : BoundRule := .product (.predecessor 0 54569 .coefficient) (.predecessor 1 54570 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54569 .coefficient)
      LeftAuthority54567.bound (LeftAuthority54567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54570 .coefficient)
      LeftBound54565.bound (LeftBound54565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54565.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority54567.bound LeftBound54565.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54567.bound, LeftBound54565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority54567.actual selector witness) * (LeftBound54565.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54571

namespace LeftBound54587
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 54587
def frameStart : Nat := 54512
def rule : BoundRule := .scale (.predecessor 0 54585 .coefficient) (.value (.predecessor 1 54586 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54585 .coefficient)
      LeftAuthority54583.bound (LeftAuthority54583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54583.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54586 .coefficient)
      LeftAuthority54574.bound (LeftAuthority54574.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54574.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54583.bound LeftAuthority54574.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54583.bound, LeftAuthority54574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority54583.actual selector witness) * (LeftAuthority54574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54587

namespace LeftBound54590
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 54590
def frameStart : Nat := 54512
def rule : BoundRule := .identity (.predecessor 0 54589 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54589 .coefficient)
      LeftAuthority54577.bound (LeftAuthority54577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54577.derived selector witness)

def rawBound : CoeffClass := LeftAuthority54577.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority54577.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54590

namespace LeftBound54594
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 54594
def frameStart : Nat := 54512
def rule : BoundRule := .product (.predecessor 0 54592 .coefficient) (.predecessor 1 54593 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54592 .coefficient)
      LeftBound54590.bound (LeftBound54590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54593 .coefficient)
      LeftBound54587.bound (LeftBound54587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54587.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound54590.bound LeftBound54587.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54590.bound, LeftBound54587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound54590.actual selector witness) * (LeftBound54587.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54594

namespace LeftBound54599
def owner : Owner := ⟨.program ⟨257⟩, ⟨20021⟩⟩
def transferEvent : Nat := 54599
def frameStart : Nat := 54512
def rule : BoundRule := .sum [.predecessor 0 54597 .coefficient, .predecessor 1 54598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54597 .coefficient)
      LeftBound54594.bound (LeftBound54594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54598 .coefficient)
      LeftBound54571.bound (LeftBound54571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54594.bound, LeftBound54571.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54594.bound, LeftBound54571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54594.actual selector witness, LeftBound54571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54599

namespace LeftBound54603
def owner : Owner := ⟨.program ⟨257⟩, ⟨20310⟩⟩
def transferEvent : Nat := 54603
def frameStart : Nat := 54512
def rule : BoundRule := .product (.predecessor 0 54601 .coefficient) (.predecessor 1 54602 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54601 .coefficient)
      LeftBound54599.bound (LeftBound54599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54602 .coefficient)
      LeftAuthority54556.bound (LeftAuthority54556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound54599.bound LeftAuthority54556.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54599.bound, LeftAuthority54556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound54599.actual selector witness) * (LeftAuthority54556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54603

namespace LeftBound54614
def owner : Owner := ⟨.program ⟨257⟩, ⟨18654⟩⟩
def transferEvent : Nat := 54614
def frameStart : Nat := 54512
def rule : BoundRule := .product (.predecessor 0 54612 .coefficient) (.predecessor 1 54613 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54612 .coefficient)
      LeftAuthority54567.bound (LeftAuthority54567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54613 .coefficient)
      LeftAuthority54610.bound (LeftAuthority54610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54567.bound LeftAuthority54610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54567.bound, LeftAuthority54610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority54567.actual selector witness) * (LeftAuthority54610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54614

namespace LeftBound54622
def owner : Owner := ⟨.program ⟨257⟩, ⟨18655⟩⟩
def transferEvent : Nat := 54622
def frameStart : Nat := 54512
def rule : BoundRule := .sum [.predecessor 0 54620 .coefficient, .predecessor 1 54621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54620 .coefficient)
      LeftAuthority54618.bound (LeftAuthority54618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54621 .coefficient)
      LeftBound54614.bound (LeftBound54614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54618.bound, LeftBound54614.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54618.bound, LeftBound54614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority54618.actual selector witness, LeftBound54614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54622

namespace LeftBound54626
def owner : Owner := ⟨.program ⟨257⟩, ⟨20311⟩⟩
def transferEvent : Nat := 54626
def frameStart : Nat := 54512
def rule : BoundRule := .sum [.predecessor 0 54624 .coefficient, .predecessor 1 54625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54624 .coefficient)
      LeftBound54622.bound (LeftBound54622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54625 .coefficient)
      LeftBound54603.bound (LeftBound54603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54622.bound, LeftBound54603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54622.bound, LeftBound54603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54622.actual selector witness, LeftBound54603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54626

namespace LeftBound54639
def owner : Owner := ⟨.program ⟨257⟩, ⟨20309⟩⟩
def transferEvent : Nat := 54639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54637 .coefficient, .predecessor 1 54638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54637 .coefficient)
      LeftBound54460.bound (LeftBound54460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54638 .coefficient)
      LeftBound54443.bound (LeftBound54443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54460.bound, LeftBound54443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54460.bound, LeftBound54443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54460.actual selector witness, LeftBound54443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54639

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
