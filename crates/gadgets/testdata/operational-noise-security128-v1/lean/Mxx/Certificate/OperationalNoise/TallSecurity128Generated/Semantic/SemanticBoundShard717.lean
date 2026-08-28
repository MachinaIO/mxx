import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard106
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard678
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard681
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard716

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound109906
def owner : Owner := ⟨.program ⟨257⟩, ⟨64291⟩⟩
def transferEvent : Nat := 109906
def frameStart : Nat := 109847
def rule : BoundRule := .identity (.predecessor 0 109905 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109905 .coefficient)
      LeftBound109903.bound (LeftBound109903.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound109903.derived selector witness)

def rawBound : CoeffClass := LeftBound109903.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound109903.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound109906

namespace LeftBound109912
def owner : Owner := ⟨.program ⟨257⟩, ⟨64292⟩⟩
def transferEvent : Nat := 109912
def frameStart : Nat := 109847
def rule : BoundRule := .product (.predecessor 0 109910 .coefficient) (.predecessor 1 109911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109910 .coefficient)
      LeftAuthority109908.bound (LeftAuthority109908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109911 .coefficient)
      LeftBound109906.bound (LeftBound109906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109906.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority109908.bound LeftBound109906.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109908.bound, LeftBound109906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority109908.actual selector witness) * (LeftBound109906.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109912

namespace LeftBound109920
def owner : Owner := ⟨.program ⟨257⟩, ⟨64293⟩⟩
def transferEvent : Nat := 109920
def frameStart : Nat := 109847
def rule : BoundRule := .sum [.predecessor 0 109918 .coefficient, .predecessor 1 109919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109918 .coefficient)
      LeftAuthority109916.bound (LeftAuthority109916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109919 .coefficient)
      LeftBound109912.bound (LeftBound109912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109912.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority109916.bound, LeftBound109912.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109916.bound, LeftBound109912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority109916.actual selector witness, LeftBound109912.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109920

namespace LeftBound109924
def owner : Owner := ⟨.program ⟨257⟩, ⟨64904⟩⟩
def transferEvent : Nat := 109924
def frameStart : Nat := 109847
def rule : BoundRule := .product (.predecessor 0 109922 .coefficient) (.predecessor 1 109923 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109922 .coefficient)
      LeftBound109920.bound (LeftBound109920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109923 .coefficient)
      LeftAuthority109897.bound (LeftAuthority109897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound109920.bound LeftAuthority109897.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109920.bound, LeftAuthority109897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound109920.actual selector witness) * (LeftAuthority109897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109924

namespace LeftBound109935
def owner : Owner := ⟨.program ⟨257⟩, ⟨63102⟩⟩
def transferEvent : Nat := 109935
def frameStart : Nat := 109847
def rule : BoundRule := .product (.predecessor 0 109933 .coefficient) (.predecessor 1 109934 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109933 .coefficient)
      LeftAuthority109908.bound (LeftAuthority109908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109934 .coefficient)
      LeftAuthority109931.bound (LeftAuthority109931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority109908.bound LeftAuthority109931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109908.bound, LeftAuthority109931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority109908.actual selector witness) * (LeftAuthority109931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109935

namespace LeftBound109943
def owner : Owner := ⟨.program ⟨257⟩, ⟨63103⟩⟩
def transferEvent : Nat := 109943
def frameStart : Nat := 109847
def rule : BoundRule := .sum [.predecessor 0 109941 .coefficient, .predecessor 1 109942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109941 .coefficient)
      LeftAuthority109939.bound (LeftAuthority109939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority109939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority109939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109942 .coefficient)
      LeftBound109935.bound (LeftBound109935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority109939.bound, LeftBound109935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority109939.bound, LeftBound109935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority109939.actual selector witness, LeftBound109935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109943

namespace LeftBound109947
def owner : Owner := ⟨.program ⟨257⟩, ⟨64908⟩⟩
def transferEvent : Nat := 109947
def frameStart : Nat := 109847
def rule : BoundRule := .sum [.predecessor 0 109945 .coefficient, .predecessor 1 109946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109945 .coefficient)
      LeftBound109943.bound (LeftBound109943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109946 .coefficient)
      LeftBound109924.bound (LeftBound109924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109943.bound, LeftBound109924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109943.bound, LeftBound109924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109943.actual selector witness, LeftBound109924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109947

namespace LeftBound109960
def owner : Owner := ⟨.program ⟨257⟩, ⟨64906⟩⟩
def transferEvent : Nat := 109960
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 109958 .coefficient, .predecessor 1 109959 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109958 .coefficient)
      LeftBound109789.bound (LeftBound109789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109959 .coefficient)
      LeftBound109772.bound (LeftBound109772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events428.exact109779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109789.bound, LeftBound109772.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109789.bound, LeftBound109772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109789.actual selector witness, LeftBound109772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109960

namespace LeftBound109963
def owner : Owner := ⟨.program ⟨257⟩, ⟨64906⟩⟩
def transferEvent : Nat := 109963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 109957 .summary, .result 109779 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109957 .summary)
      LeftBound109791.bound (LeftBound109791.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63699⟩⟩) (rawTerms := some (Proof.Events429.exact109957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109779 .summary)
      LeftBound109774.bound (LeftBound109774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64905⟩⟩) (rawTerms := some (Proof.Events428.exact109779RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109791.bound, LeftBound109774.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109791.bound, LeftBound109774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109791.actual selector witness, LeftBound109774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109963

namespace LeftBound109987
def owner : Owner := ⟨.program ⟨257⟩, ⟨25263⟩⟩
def transferEvent : Nat := 109987
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 109985 .coefficient) (.predecessor 1 109986 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109985 .coefficient)
      LeftAuthority4811.bound (LeftAuthority4811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109986 .coefficient)
      LeftBound105151.bound (LeftBound105151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority4811.bound LeftBound105151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4811.bound, LeftBound105151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority4811.actual selector witness) * (LeftBound105151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound109987

namespace LeftBound109992
def owner : Owner := ⟨.program ⟨257⟩, ⟨8694⟩⟩
def transferEvent : Nat := 109992
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 109990 .coefficient) (.predecessor 1 109991 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109990 .coefficient)
      LeftBound105022.bound (LeftBound105022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109991 .coefficient)
      LeftBound22089.bound (LeftBound22089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound105022.bound LeftBound22089.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105022.bound, LeftBound22089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound105022.actual selector witness) * (LeftBound22089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound109992

namespace LeftBound109997
def owner : Owner := ⟨.program ⟨257⟩, ⟨25264⟩⟩
def transferEvent : Nat := 109997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 109995 .coefficient, .predecessor 1 109996 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109995 .coefficient)
      LeftBound109992.bound (LeftBound109992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 109996 .coefficient)
      LeftBound109987.bound (LeftBound109987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109992.bound, LeftBound109987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109992.bound, LeftBound109987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109992.actual selector witness, LeftBound109987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound109997

namespace LeftBound110001
def owner : Owner := ⟨.program ⟨257⟩, ⟨25265⟩⟩
def transferEvent : Nat := 110001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 109999 .coefficient, .predecessor 1 110000 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 109999 .coefficient)
      LeftBound109997.bound (LeftBound109997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110000 .coefficient)
      LeftBound22081.bound (LeftBound22081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound109997.bound, LeftBound22081.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound109997.bound, LeftBound22081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound109997.actual selector witness, LeftBound22081.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound110001

namespace LeftBound110002
def owner : Owner := ⟨.program ⟨257⟩, ⟨25265⟩⟩
def transferEvent : Nat := 110002
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨100⟩⟩]⟩ [⟨.result 22082 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22082 .coefficient)
      LeftBound22081.bound (LeftBound22081.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events086.exact22082RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22081.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22081.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22081.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound110002

namespace LeftBound110007
def owner : Owner := ⟨.program ⟨257⟩, ⟨59515⟩⟩
def transferEvent : Nat := 110007
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 110005 .coefficient) (.predecessor 1 110006 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 110005 .coefficient)
      LeftBound110001.bound (LeftBound110001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact110004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 110006 .coefficient)
      LeftAuthority4814.bound (LeftAuthority4814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4814.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound110001.bound LeftAuthority4814.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound110001.bound, LeftAuthority4814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound110001.actual selector witness) * (LeftAuthority4814.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound110007

namespace LeftBound110008
def owner : Owner := ⟨.program ⟨257⟩, ⟨59515⟩⟩
def transferEvent : Nat := 110008
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨59512⟩⟩], []⟩ [⟨.result 4815 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 4815 .coefficient)
      LeftAuthority4814.bound (LeftAuthority4814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨59512⟩⟩) (rawTerms := some (Proof.Events018.exact4815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4814.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority4814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound110008

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
