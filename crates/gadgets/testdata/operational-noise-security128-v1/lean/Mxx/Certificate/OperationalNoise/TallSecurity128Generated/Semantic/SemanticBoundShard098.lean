import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard097

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21005
def owner : Owner := ⟨.program ⟨257⟩, ⟨27732⟩⟩
def transferEvent : Nat := 21005
def frameStart : Nat := 20940
def rule : BoundRule := .product (.predecessor 0 21003 .coefficient) (.predecessor 1 21004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21003 .coefficient)
      LeftAuthority21001.bound (LeftAuthority21001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21001.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21004 .coefficient)
      LeftBound20999.bound (LeftBound20999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority21001.bound LeftBound20999.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21001.bound, LeftBound20999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority21001.actual selector witness) * (LeftBound20999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21005

namespace LeftBound21013
def owner : Owner := ⟨.program ⟨257⟩, ⟨27733⟩⟩
def transferEvent : Nat := 21013
def frameStart : Nat := 20940
def rule : BoundRule := .sum [.predecessor 0 21011 .coefficient, .predecessor 1 21012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21011 .coefficient)
      LeftAuthority21009.bound (LeftAuthority21009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21012 .coefficient)
      LeftBound21005.bound (LeftBound21005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21009.bound, LeftBound21005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21009.bound, LeftBound21005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority21009.actual selector witness, LeftBound21005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21013

namespace LeftBound21017
def owner : Owner := ⟨.program ⟨257⟩, ⟨28072⟩⟩
def transferEvent : Nat := 21017
def frameStart : Nat := 20940
def rule : BoundRule := .product (.predecessor 0 21015 .coefficient) (.predecessor 1 21016 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21015 .coefficient)
      LeftBound21013.bound (LeftBound21013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21013.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21016 .coefficient)
      LeftAuthority20990.bound (LeftAuthority20990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20990.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21013.bound LeftAuthority20990.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21013.bound, LeftAuthority20990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21013.actual selector witness) * (LeftAuthority20990.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21017

namespace LeftBound21028
def owner : Owner := ⟨.program ⟨257⟩, ⟨26506⟩⟩
def transferEvent : Nat := 21028
def frameStart : Nat := 20940
def rule : BoundRule := .product (.predecessor 0 21026 .coefficient) (.predecessor 1 21027 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21026 .coefficient)
      LeftAuthority21001.bound (LeftAuthority21001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21001.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21027 .coefficient)
      LeftAuthority21024.bound (LeftAuthority21024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21024.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21024.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority21001.bound LeftAuthority21024.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21001.bound, LeftAuthority21024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority21001.actual selector witness) * (LeftAuthority21024.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21028

namespace LeftBound21036
def owner : Owner := ⟨.program ⟨257⟩, ⟨26507⟩⟩
def transferEvent : Nat := 21036
def frameStart : Nat := 20940
def rule : BoundRule := .sum [.predecessor 0 21034 .coefficient, .predecessor 1 21035 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21034 .coefficient)
      LeftAuthority21032.bound (LeftAuthority21032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21035 .coefficient)
      LeftBound21028.bound (LeftBound21028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21028.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21032.bound, LeftBound21028.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21032.bound, LeftBound21028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority21032.actual selector witness, LeftBound21028.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21036

namespace LeftBound21040
def owner : Owner := ⟨.program ⟨257⟩, ⟨28075⟩⟩
def transferEvent : Nat := 21040
def frameStart : Nat := 20940
def rule : BoundRule := .sum [.predecessor 0 21038 .coefficient, .predecessor 1 21039 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21038 .coefficient)
      LeftBound21036.bound (LeftBound21036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21039 .coefficient)
      LeftBound21017.bound (LeftBound21017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21036.bound, LeftBound21017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21036.bound, LeftBound21017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21036.actual selector witness, LeftBound21017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21040

namespace LeftBound21053
def owner : Owner := ⟨.program ⟨257⟩, ⟨28074⟩⟩
def transferEvent : Nat := 21053
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21051 .coefficient, .predecessor 1 21052 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21051 .coefficient)
      LeftBound20882.bound (LeftBound20882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21052 .coefficient)
      LeftBound20865.bound (LeftBound20865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20865.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20865.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20882.bound, LeftBound20865.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20882.bound, LeftBound20865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound20882.actual selector witness, LeftBound20865.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21053

namespace LeftBound21056
def owner : Owner := ⟨.program ⟨257⟩, ⟨28074⟩⟩
def transferEvent : Nat := 21056
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21050 .summary, .result 20872 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21050 .summary)
      LeftBound20884.bound (LeftBound20884.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26985⟩⟩) (rawTerms := some (Proof.Events082.exact21050RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20872 .summary)
      LeftBound20867.bound (LeftBound20867.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28073⟩⟩) (rawTerms := some (Proof.Events081.exact20872RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20867.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20884.bound, LeftBound20867.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20884.bound, LeftBound20867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound20884.actual selector witness, LeftBound20867.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21056

namespace LeftBound21079
def owner : Owner := ⟨.program ⟨257⟩, ⟨102⟩⟩
def transferEvent : Nat := 21079
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 21078 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21078 .coefficient)
      LeftAuthority17048.bound (LeftAuthority17048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17048.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17048.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority17048.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21079

namespace LeftBound21083
def owner : Owner := ⟨.program ⟨257⟩, ⟨25627⟩⟩
def transferEvent : Nat := 21083
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 21081 .coefficient) (.predecessor 1 21082 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21081 .coefficient)
      LeftAuthority234.bound (LeftAuthority234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21082 .coefficient)
      LeftBound17055.bound (LeftBound17055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17055.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority234.bound LeftBound17055.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority234.bound, LeftBound17055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority234.actual selector witness) * (LeftBound17055.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21083

namespace LeftBound21087
def owner : Owner := ⟨.program ⟨257⟩, ⟨7276⟩⟩
def transferEvent : Nat := 21087
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 21086 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21086 .coefficient)
      LeftAuthority15892.bound (LeftAuthority15892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15892.derived selector witness)

def rawBound : CoeffClass := LeftAuthority15892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority15892.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21087

namespace LeftBound21091
def owner : Owner := ⟨.program ⟨257⟩, ⟨7594⟩⟩
def transferEvent : Nat := 21091
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21089 .coefficient) (.predecessor 1 21090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21089 .coefficient)
      LeftBound16921.bound (LeftBound16921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact16922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21090 .coefficient)
      LeftBound21087.bound (LeftBound21087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound16921.bound LeftBound21087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16921.bound, LeftBound21087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound16921.actual selector witness) * (LeftBound21087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21091

namespace LeftBound21096
def owner : Owner := ⟨.program ⟨257⟩, ⟨25628⟩⟩
def transferEvent : Nat := 21096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21094 .coefficient, .predecessor 1 21095 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21094 .coefficient)
      LeftBound21091.bound (LeftBound21091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21095 .coefficient)
      LeftBound21083.bound (LeftBound21083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21091.bound, LeftBound21083.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21091.bound, LeftBound21083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21091.actual selector witness, LeftBound21083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21096

namespace LeftBound21100
def owner : Owner := ⟨.program ⟨257⟩, ⟨25629⟩⟩
def transferEvent : Nat := 21100
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21098 .coefficient, .predecessor 1 21099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21098 .coefficient)
      LeftBound21096.bound (LeftBound21096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21099 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21096.bound, LeftBound21079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21096.bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21096.actual selector witness, LeftBound21079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21100

namespace LeftBound21101
def owner : Owner := ⟨.program ⟨257⟩, ⟨25629⟩⟩
def transferEvent : Nat := 21101
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩ [⟨.result 21080 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21080 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21079.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21079.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21101

namespace LeftBound21106
def owner : Owner := ⟨.program ⟨257⟩, ⟨65214⟩⟩
def transferEvent : Nat := 21106
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21104 .coefficient) (.predecessor 1 21105 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21104 .coefficient)
      LeftBound21100.bound (LeftBound21100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21105 .coefficient)
      LeftAuthority237.bound (LeftAuthority237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound21100.bound LeftAuthority237.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21100.bound, LeftAuthority237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound21100.actual selector witness) * (LeftAuthority237.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21106

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
