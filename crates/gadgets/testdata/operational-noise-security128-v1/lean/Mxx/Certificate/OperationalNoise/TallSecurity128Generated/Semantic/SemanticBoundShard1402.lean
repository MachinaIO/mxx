import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard079
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1401

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound208944
def owner : Owner := ⟨.program ⟨257⟩, ⟨43001⟩⟩
def transferEvent : Nat := 208944
def frameStart : Nat := 208848
def rule : BoundRule := .sum [.predecessor 0 208942 .coefficient, .predecessor 1 208943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208942 .coefficient)
      LeftAuthority208940.bound (LeftAuthority208940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority208940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority208940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208943 .coefficient)
      LeftBound208936.bound (LeftBound208936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority208940.bound, LeftBound208936.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority208940.bound, LeftBound208936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority208940.actual selector witness, LeftBound208936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208944

namespace LeftBound208948
def owner : Owner := ⟨.program ⟨257⟩, ⟨44673⟩⟩
def transferEvent : Nat := 208948
def frameStart : Nat := 208848
def rule : BoundRule := .sum [.predecessor 0 208946 .coefficient, .predecessor 1 208947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208946 .coefficient)
      LeftBound208944.bound (LeftBound208944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208947 .coefficient)
      LeftBound208925.bound (LeftBound208925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208944.bound, LeftBound208925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208944.bound, LeftBound208925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208944.actual selector witness, LeftBound208925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208948

namespace LeftBound208961
def owner : Owner := ⟨.program ⟨257⟩, ⟨44672⟩⟩
def transferEvent : Nat := 208961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 208959 .coefficient, .predecessor 1 208960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208959 .coefficient)
      LeftBound208790.bound (LeftBound208790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208960 .coefficient)
      LeftBound208773.bound (LeftBound208773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events815.exact208780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208773.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208790.bound, LeftBound208773.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208790.bound, LeftBound208773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208790.actual selector witness, LeftBound208773.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208961

namespace LeftBound208964
def owner : Owner := ⟨.program ⟨257⟩, ⟨44672⟩⟩
def transferEvent : Nat := 208964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 208958 .summary, .result 208780 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208958 .summary)
      LeftBound208792.bound (LeftBound208792.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43539⟩⟩) (rawTerms := some (Proof.Events816.exact208958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208780 .summary)
      LeftBound208775.bound (LeftBound208775.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44671⟩⟩) (rawTerms := some (Proof.Events815.exact208780RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208792.bound, LeftBound208775.bound]
def bound : CoeffClass := .finite ⟨32193718473625891320532869316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208792.bound, LeftBound208775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208792.actual selector witness, LeftBound208775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208964

namespace LeftBound208988
def owner : Owner := ⟨.program ⟨257⟩, ⟨39797⟩⟩
def transferEvent : Nat := 208988
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 208986 .coefficient) (.predecessor 1 208987 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208986 .coefficient)
      LeftAuthority9886.bound (LeftAuthority9886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208987 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9886.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9886.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9886.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound208988

namespace LeftBound208993
def owner : Owner := ⟨.program ⟨257⟩, ⟨8588⟩⟩
def transferEvent : Nat := 208993
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 208991 .coefficient) (.predecessor 1 208992 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208991 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208992 .coefficient)
      LeftBound18582.bound (LeftBound18582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound18582.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound18582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound18582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound208993

namespace LeftBound208998
def owner : Owner := ⟨.program ⟨257⟩, ⟨39798⟩⟩
def transferEvent : Nat := 208998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 208996 .coefficient, .predecessor 1 208997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 208996 .coefficient)
      LeftBound208993.bound (LeftBound208993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 208997 .coefficient)
      LeftBound208988.bound (LeftBound208988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208988.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208993.bound, LeftBound208988.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208993.bound, LeftBound208988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208993.actual selector witness, LeftBound208988.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208998

namespace LeftBound209002
def owner : Owner := ⟨.program ⟨257⟩, ⟨39799⟩⟩
def transferEvent : Nat := 209002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209000 .coefficient, .predecessor 1 209001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209000 .coefficient)
      LeftBound208998.bound (LeftBound208998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209001 .coefficient)
      LeftBound18574.bound (LeftBound18574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound208998.bound, LeftBound18574.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound208998.bound, LeftBound18574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound208998.actual selector witness, LeftBound18574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209002

namespace LeftBound209003
def owner : Owner := ⟨.program ⟨257⟩, ⟨39799⟩⟩
def transferEvent : Nat := 209003
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨108⟩⟩]⟩ [⟨.result 18575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18575 .coefficient)
      LeftBound18574.bound (LeftBound18574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨108⟩⟩) (rawTerms := some (Proof.Events072.exact18575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound18574.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound18574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound209003

namespace LeftBound209008
def owner : Owner := ⟨.program ⟨257⟩, ⟨39800⟩⟩
def transferEvent : Nat := 209008
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 209006 .coefficient) (.predecessor 1 209007 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209006 .coefficient)
      LeftBound209002.bound (LeftBound209002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact209005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209007 .coefficient)
      LeftAuthority9889.bound (LeftAuthority9889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9889.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound209002.bound LeftAuthority9889.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209002.bound, LeftAuthority9889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound209002.actual selector witness) * (LeftAuthority9889.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209008

namespace LeftBound209009
def owner : Owner := ⟨.program ⟨257⟩, ⟨39800⟩⟩
def transferEvent : Nat := 209009
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14181⟩⟩], []⟩ [⟨.result 9890 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 9890 .coefficient)
      LeftAuthority9889.bound (LeftAuthority9889.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14181⟩⟩) (rawTerms := some (Proof.Events038.exact9890RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9889.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9889.bound []
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority9889.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound209009

namespace LeftBound209010
def owner : Owner := ⟨.program ⟨257⟩, ⟨39800⟩⟩
def transferEvent : Nat := 209010
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 209005 .summary) (.transfer 209009) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209005 .summary)
      LeftBound209003.bound (LeftBound209003.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39799⟩⟩) (rawTerms := some (Proof.Events816.exact209005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 209009)
      LeftBound209009.bound (LeftBound209009.actual selector witness) := by
  exact .transfer (LeftBound209009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound209003.bound LeftBound209009.bound
def bound : CoeffClass := .finite ⟨39190528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209003.bound, LeftBound209009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound209003.actual selector witness) * (LeftBound209009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209010

namespace LeftBound209016
def owner : Owner := ⟨.program ⟨257⟩, ⟨14182⟩⟩
def transferEvent : Nat := 209016
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 209014 .coefficient) (.predecessor 1 209015 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209014 .coefficient)
      LeftAuthority9889.bound (LeftAuthority9889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209015 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9889.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9889.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9889.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound209016

namespace LeftBound209021
def owner : Owner := ⟨.program ⟨257⟩, ⟨8605⟩⟩
def transferEvent : Nat := 209021
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 209019 .coefficient) (.predecessor 1 209020 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209019 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209020 .coefficient)
      LeftBound18623.bound (LeftBound18623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18623.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound18623.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound18623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound18623.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound209021

namespace LeftBound209026
def owner : Owner := ⟨.program ⟨257⟩, ⟨14183⟩⟩
def transferEvent : Nat := 209026
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209024 .coefficient, .predecessor 1 209025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209024 .coefficient)
      LeftBound209021.bound (LeftBound209021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact209023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209025 .coefficient)
      LeftBound209016.bound (LeftBound209016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact209018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209021.bound, LeftBound209016.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209021.bound, LeftBound209016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209021.actual selector witness, LeftBound209016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209026

namespace LeftBound209030
def owner : Owner := ⟨.program ⟨257⟩, ⟨14184⟩⟩
def transferEvent : Nat := 209030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 209028 .coefficient, .predecessor 1 209029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 209028 .coefficient)
      LeftBound209026.bound (LeftBound209026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact209027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 209029 .coefficient)
      LeftBound18615.bound (LeftBound18615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18615.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound209026.bound, LeftBound18615.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound209026.bound, LeftBound18615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound209026.actual selector witness, LeftBound18615.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound209030

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
