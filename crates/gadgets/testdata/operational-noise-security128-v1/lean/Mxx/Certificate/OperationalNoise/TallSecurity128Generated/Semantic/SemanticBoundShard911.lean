import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard910

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound137873
def owner : Owner := ⟨.program ⟨257⟩, ⟨26782⟩⟩
def transferEvent : Nat := 137873
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨26779⟩⟩]⟩ [⟨.result 137865 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 137865 .coefficient)
      LeftAuthority137864.bound (LeftAuthority137864.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26779⟩⟩) (rawTerms := some (Proof.Events538.exact137865RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137864.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority137864.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority137864.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound137873

namespace LeftBound137874
def owner : Owner := ⟨.program ⟨257⟩, ⟨26782⟩⟩
def transferEvent : Nat := 137874
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 134495 .summary) (.transfer 137873) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134495 .summary)
      LeftBound134493.bound (LeftBound134493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5473⟩⟩) (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 137873)
      LeftBound137873.bound (LeftBound137873.actual selector witness) := by
  exact .transfer (LeftBound137873.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134493.bound LeftBound137873.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134493.bound, LeftBound137873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134493.actual selector witness) * (LeftBound137873.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137874

namespace LeftBound137953
def owner : Owner := ⟨.program ⟨257⟩, ⟨25927⟩⟩
def transferEvent : Nat := 137953
def frameStart : Nat := 137924
def rule : BoundRule := .product (.predecessor 0 137951 .coefficient) (.predecessor 1 137952 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137951 .coefficient)
      LeftAuthority137949.bound (LeftAuthority137949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137949.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137952 .coefficient)
      LeftAuthority137946.bound (LeftAuthority137946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137946.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority137949.bound LeftAuthority137946.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137949.bound, LeftAuthority137946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority137949.actual selector witness) * (LeftAuthority137946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137953

namespace LeftBound137957
def owner : Owner := ⟨.program ⟨257⟩, ⟨25928⟩⟩
def transferEvent : Nat := 137957
def frameStart : Nat := 137924
def rule : BoundRule := .identity (.predecessor 0 137956 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137956 .coefficient)
      LeftBound137953.bound (LeftBound137953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137953.derived selector witness)

def rawBound : CoeffClass := LeftBound137953.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound137953.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound137957

namespace LeftBound137974
def owner : Owner := ⟨.program ⟨257⟩, ⟨27658⟩⟩
def transferEvent : Nat := 137974
def frameStart : Nat := 137924
def rule : BoundRule := .sum [.predecessor 0 137972 .coefficient, .predecessor 1 137973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137972 .coefficient)
      LeftBound137957.bound (LeftBound137957.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound137957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137973 .coefficient)
      LeftAuthority137970.bound (LeftAuthority137970.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority137970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound137957.bound, LeftAuthority137970.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137957.bound, LeftAuthority137970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound137957.actual selector witness, LeftAuthority137970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound137974

namespace LeftBound137977
def owner : Owner := ⟨.program ⟨257⟩, ⟨27659⟩⟩
def transferEvent : Nat := 137977
def frameStart : Nat := 137924
def rule : BoundRule := .identity (.predecessor 0 137976 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137976 .coefficient)
      LeftBound137974.bound (LeftBound137974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound137974.derived selector witness)

def rawBound : CoeffClass := LeftBound137974.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound137974.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound137977

namespace LeftBound137983
def owner : Owner := ⟨.program ⟨257⟩, ⟨27660⟩⟩
def transferEvent : Nat := 137983
def frameStart : Nat := 137924
def rule : BoundRule := .product (.predecessor 0 137981 .coefficient) (.predecessor 1 137982 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137981 .coefficient)
      LeftAuthority137979.bound (LeftAuthority137979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137982 .coefficient)
      LeftBound137977.bound (LeftBound137977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137977.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority137979.bound LeftBound137977.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137979.bound, LeftBound137977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority137979.actual selector witness) * (LeftBound137977.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound137983

namespace LeftBound137999
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 137999
def frameStart : Nat := 137924
def rule : BoundRule := .scale (.predecessor 0 137997 .coefficient) (.value (.predecessor 1 137998 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 137997 .coefficient)
      LeftAuthority137995.bound (LeftAuthority137995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact137996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 137998 .coefficient)
      LeftAuthority137986.bound (LeftAuthority137986.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority137986.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority137995.bound LeftAuthority137986.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137995.bound, LeftAuthority137986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority137995.actual selector witness) * (LeftAuthority137986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound137999

namespace LeftBound138002
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 138002
def frameStart : Nat := 137924
def rule : BoundRule := .identity (.predecessor 0 138001 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138001 .coefficient)
      LeftAuthority137989.bound (LeftAuthority137989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact137990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137989.derived selector witness)

def rawBound : CoeffClass := LeftAuthority137989.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority137989.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound138002

namespace LeftBound138006
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 138006
def frameStart : Nat := 137924
def rule : BoundRule := .product (.predecessor 0 138004 .coefficient) (.predecessor 1 138005 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138004 .coefficient)
      LeftBound138002.bound (LeftBound138002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138005 .coefficient)
      LeftBound137999.bound (LeftBound137999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138002.bound LeftBound137999.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138002.bound, LeftBound137999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138002.actual selector witness) * (LeftBound137999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138006

namespace LeftBound138011
def owner : Owner := ⟨.program ⟨257⟩, ⟨27661⟩⟩
def transferEvent : Nat := 138011
def frameStart : Nat := 137924
def rule : BoundRule := .sum [.predecessor 0 138009 .coefficient, .predecessor 1 138010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138009 .coefficient)
      LeftBound138006.bound (LeftBound138006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138010 .coefficient)
      LeftBound137983.bound (LeftBound137983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact137985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137983.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138006.bound, LeftBound137983.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138006.bound, LeftBound137983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138006.actual selector witness, LeftBound137983.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138011

namespace LeftBound138015
def owner : Owner := ⟨.program ⟨257⟩, ⟨27845⟩⟩
def transferEvent : Nat := 138015
def frameStart : Nat := 137924
def rule : BoundRule := .product (.predecessor 0 138013 .coefficient) (.predecessor 1 138014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138013 .coefficient)
      LeftBound138011.bound (LeftBound138011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138014 .coefficient)
      LeftAuthority137968.bound (LeftAuthority137968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound138011.bound LeftAuthority137968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138011.bound, LeftAuthority137968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound138011.actual selector witness) * (LeftAuthority137968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138015

namespace LeftBound138026
def owner : Owner := ⟨.program ⟨257⟩, ⟨26354⟩⟩
def transferEvent : Nat := 138026
def frameStart : Nat := 137924
def rule : BoundRule := .product (.predecessor 0 138024 .coefficient) (.predecessor 1 138025 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138024 .coefficient)
      LeftAuthority137979.bound (LeftAuthority137979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority137979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority137979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138025 .coefficient)
      LeftAuthority138022.bound (LeftAuthority138022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority137979.bound LeftAuthority138022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority137979.bound, LeftAuthority138022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority137979.actual selector witness) * (LeftAuthority138022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound138026

namespace LeftBound138034
def owner : Owner := ⟨.program ⟨257⟩, ⟨26355⟩⟩
def transferEvent : Nat := 138034
def frameStart : Nat := 137924
def rule : BoundRule := .sum [.predecessor 0 138032 .coefficient, .predecessor 1 138033 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138032 .coefficient)
      LeftAuthority138030.bound (LeftAuthority138030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority138030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority138030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138033 .coefficient)
      LeftBound138026.bound (LeftBound138026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority138030.bound, LeftBound138026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority138030.bound, LeftBound138026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority138030.actual selector witness, LeftBound138026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138034

namespace LeftBound138038
def owner : Owner := ⟨.program ⟨257⟩, ⟨27846⟩⟩
def transferEvent : Nat := 138038
def frameStart : Nat := 137924
def rule : BoundRule := .sum [.predecessor 0 138036 .coefficient, .predecessor 1 138037 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138036 .coefficient)
      LeftBound138034.bound (LeftBound138034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138037 .coefficient)
      LeftBound138015.bound (LeftBound138015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound138034.bound, LeftBound138015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound138034.bound, LeftBound138015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound138034.actual selector witness, LeftBound138015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138038

namespace LeftBound138051
def owner : Owner := ⟨.program ⟨257⟩, ⟨27844⟩⟩
def transferEvent : Nat := 138051
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 138049 .coefficient, .predecessor 1 138050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 138049 .coefficient)
      LeftBound137872.bound (LeftBound137872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events539.exact138048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 138050 .coefficient)
      LeftBound137855.bound (LeftBound137855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound137872.bound, LeftBound137855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound137872.bound, LeftBound137855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound137872.actual selector witness, LeftBound137855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound138051

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
