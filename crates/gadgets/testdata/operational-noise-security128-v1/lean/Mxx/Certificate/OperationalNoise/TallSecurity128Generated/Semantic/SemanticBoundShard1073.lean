import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1035
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1072

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound162042
def owner : Owner := ⟨.program ⟨257⟩, ⟨54091⟩⟩
def transferEvent : Nat := 162042
def frameStart : Nat := 161954
def rule : BoundRule := .product (.predecessor 0 162040 .coefficient) (.predecessor 1 162041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162040 .coefficient)
      LeftAuthority162015.bound (LeftAuthority162015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact162016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162041 .coefficient)
      LeftAuthority162038.bound (LeftAuthority162038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact162039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority162015.bound LeftAuthority162038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162015.bound, LeftAuthority162038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority162015.actual selector witness) * (LeftAuthority162038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162042

namespace LeftBound162050
def owner : Owner := ⟨.program ⟨257⟩, ⟨54092⟩⟩
def transferEvent : Nat := 162050
def frameStart : Nat := 161954
def rule : BoundRule := .sum [.predecessor 0 162048 .coefficient, .predecessor 1 162049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162048 .coefficient)
      LeftAuthority162046.bound (LeftAuthority162046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact162047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162049 .coefficient)
      LeftBound162042.bound (LeftBound162042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact162044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority162046.bound, LeftBound162042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162046.bound, LeftBound162042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority162046.actual selector witness, LeftBound162042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound162050

namespace LeftBound162054
def owner : Owner := ⟨.program ⟨257⟩, ⟨55838⟩⟩
def transferEvent : Nat := 162054
def frameStart : Nat := 161954
def rule : BoundRule := .sum [.predecessor 0 162052 .coefficient, .predecessor 1 162053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162052 .coefficient)
      LeftBound162050.bound (LeftBound162050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162053 .coefficient)
      LeftBound162031.bound (LeftBound162031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact162036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound162050.bound, LeftBound162031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound162050.bound, LeftBound162031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound162050.actual selector witness, LeftBound162031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound162054

namespace LeftBound162067
def owner : Owner := ⟨.program ⟨257⟩, ⟨55835⟩⟩
def transferEvent : Nat := 162067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 162065 .coefficient, .predecessor 1 162066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162065 .coefficient)
      LeftBound161896.bound (LeftBound161896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162066 .coefficient)
      LeftBound161879.bound (LeftBound161879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact161886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound161896.bound, LeftBound161879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound161896.bound, LeftBound161879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound161896.actual selector witness, LeftBound161879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound162067

namespace LeftBound162070
def owner : Owner := ⟨.program ⟨257⟩, ⟨55835⟩⟩
def transferEvent : Nat := 162070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 162064 .summary, .result 161886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162064 .summary)
      LeftBound161898.bound (LeftBound161898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54675⟩⟩) (rawTerms := some (Proof.Events633.exact162064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161886 .summary)
      LeftBound161881.bound (LeftBound161881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55834⟩⟩) (rawTerms := some (Proof.Events632.exact161886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound161898.bound, LeftBound161881.bound]
def bound : CoeffClass := .finite ⟨32189789464712143775715074244608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound161898.bound, LeftBound161881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound161898.actual selector witness, LeftBound161881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound162070

namespace LeftBound162074
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def transferEvent : Nat := 162074
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 162072 .coefficient) (.predecessor 1 162073 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162072 .coefficient)
      LeftBound162067.bound (LeftBound162067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162073 .coefficient)
      LeftBound15781.bound (LeftBound15781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15781.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound162067.bound LeftBound15781.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound162067.bound, LeftBound15781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound162067.actual selector witness) * (LeftBound15781.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162074

namespace LeftBound162075
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def transferEvent : Nat := 162075
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩ [⟨.result 15778 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15778 .coefficient)
      LeftAuthority15777.bound (LeftAuthority15777.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7125⟩⟩) (rawTerms := some (Proof.Events061.exact15778RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15777.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15777.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15777.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound162075

namespace LeftBound162076
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def transferEvent : Nat := 162076
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 162071 .summary) (.transfer 162075) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162071 .summary)
      LeftBound162070.bound (LeftBound162070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55835⟩⟩) (rawTerms := some (Proof.Events633.exact162071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 162075)
      LeftBound162075.bound (LeftBound162075.actual selector witness) := by
  exact .transfer (LeftBound162075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound162070.bound LeftBound162075.bound
def bound : CoeffClass := .finite ⟨345635232540160008926865507237008160849920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound162070.bound, LeftBound162075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound162070.actual selector witness) * (LeftBound162075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162076

namespace LeftBound162091
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def transferEvent : Nat := 162091
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 162089 .coefficient) (.predecessor 1 162090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162089 .coefficient)
      LeftBound155568.bound (LeftBound155568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162090 .coefficient)
      LeftAuthority162087.bound (LeftAuthority162087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162087.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound155568.bound LeftAuthority162087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155568.bound, LeftAuthority162087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound155568.actual selector witness) * (LeftAuthority162087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162091

namespace LeftBound162092
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def transferEvent : Nat := 162092
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52852⟩⟩]⟩ [⟨.result 162088 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162088 .coefficient)
      LeftAuthority162087.bound (LeftAuthority162087.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52852⟩⟩) (rawTerms := some (Proof.Events633.exact162088RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162087.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162087.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority162087.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority162087.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound162092

namespace LeftBound162093
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def transferEvent : Nat := 162093
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 155572 .summary) (.transfer 162092) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155572 .summary)
      LeftBound155571.bound (LeftBound155571.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52488⟩⟩) (rawTerms := some (Proof.Events607.exact155572RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 162092)
      LeftBound162092.bound (LeftBound162092.actual selector witness) := by
  exact .transfer (LeftBound162092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound155571.bound LeftBound162092.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155571.bound, LeftBound162092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound155571.actual selector witness) * (LeftBound162092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162093

namespace LeftBound162104
def owner : Owner := ⟨.program ⟨257⟩, ⟨51694⟩⟩
def transferEvent : Nat := 162104
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 162102 .coefficient) (.value (.predecessor 1 162103 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162102 .coefficient)
      LeftAuthority162100.bound (LeftAuthority162100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162100.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162103 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority162100.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162100.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority162100.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound162104

namespace LeftBound162108
def owner : Owner := ⟨.program ⟨257⟩, ⟨51695⟩⟩
def transferEvent : Nat := 162108
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 162106 .coefficient) (.predecessor 1 162107 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162106 .coefficient)
      LeftBound149117.bound (LeftBound149117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 162107 .coefficient)
      LeftBound162104.bound (LeftBound162104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162104.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149117.bound LeftBound162104.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149117.bound, LeftBound162104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149117.actual selector witness) * (LeftBound162104.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162108

namespace LeftBound162109
def owner : Owner := ⟨.program ⟨257⟩, ⟨51695⟩⟩
def transferEvent : Nat := 162109
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51692⟩⟩]⟩ [⟨.result 162101 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162101 .coefficient)
      LeftAuthority162100.bound (LeftAuthority162100.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51692⟩⟩) (rawTerms := some (Proof.Events633.exact162101RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162100.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162100.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority162100.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority162100.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound162109

namespace LeftBound162110
def owner : Owner := ⟨.program ⟨257⟩, ⟨51695⟩⟩
def transferEvent : Nat := 162110
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 162109) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 162109)
      LeftBound162109.bound (LeftBound162109.actual selector witness) := by
  exact .transfer (LeftBound162109.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound162109.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound162109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound162109.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound162110

namespace LeftBound162205
def owner : Owner := ⟨.program ⟨257⟩, ⟨50865⟩⟩
def transferEvent : Nat := 162205
def frameStart : Nat := 162166
def rule : BoundRule := .identity (.predecessor 0 162204 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 162204 .coefficient)
      LeftAuthority162202.bound (LeftAuthority162202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority162202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority162202.derived selector witness)

def rawBound : CoeffClass := LeftAuthority162202.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority162202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority162202.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound162205

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
