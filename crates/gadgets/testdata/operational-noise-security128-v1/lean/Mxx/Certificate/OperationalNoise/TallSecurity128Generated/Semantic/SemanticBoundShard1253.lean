import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1246
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1250
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1252

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound186772
def owner : Owner := ⟨.program ⟨257⟩, ⟨16659⟩⟩
def transferEvent : Nat := 186772
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 186771) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 186771)
      LeftBound186771.bound (LeftBound186771.actual selector witness) := by
  exact .transfer (LeftBound186771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound186771.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound186771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound186771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186772

namespace LeftBound186867
def owner : Owner := ⟨.program ⟨257⟩, ⟨15813⟩⟩
def transferEvent : Nat := 186867
def frameStart : Nat := 186828
def rule : BoundRule := .identity (.predecessor 0 186866 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186866 .coefficient)
      LeftAuthority186864.bound (LeftAuthority186864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186864.derived selector witness)

def rawBound : CoeffClass := LeftAuthority186864.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority186864.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186867

namespace LeftBound186884
def owner : Owner := ⟨.program ⟨257⟩, ⟨17218⟩⟩
def transferEvent : Nat := 186884
def frameStart : Nat := 186828
def rule : BoundRule := .sum [.predecessor 0 186882 .coefficient, .predecessor 1 186883 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186882 .coefficient)
      LeftBound186867.bound (LeftBound186867.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound186867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186883 .coefficient)
      LeftAuthority186880.bound (LeftAuthority186880.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority186880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186867.bound, LeftAuthority186880.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186867.bound, LeftAuthority186880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186867.actual selector witness, LeftAuthority186880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186884

namespace LeftBound186887
def owner : Owner := ⟨.program ⟨257⟩, ⟨17219⟩⟩
def transferEvent : Nat := 186887
def frameStart : Nat := 186828
def rule : BoundRule := .identity (.predecessor 0 186886 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186886 .coefficient)
      LeftBound186884.bound (LeftBound186884.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound186884.derived selector witness)

def rawBound : CoeffClass := LeftBound186884.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound186884.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186887

namespace LeftBound186893
def owner : Owner := ⟨.program ⟨257⟩, ⟨17220⟩⟩
def transferEvent : Nat := 186893
def frameStart : Nat := 186828
def rule : BoundRule := .product (.predecessor 0 186891 .coefficient) (.predecessor 1 186892 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186891 .coefficient)
      LeftAuthority186889.bound (LeftAuthority186889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186892 .coefficient)
      LeftBound186887.bound (LeftBound186887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186887.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority186889.bound LeftBound186887.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186889.bound, LeftBound186887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority186889.actual selector witness) * (LeftBound186887.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186893

namespace LeftBound186901
def owner : Owner := ⟨.program ⟨257⟩, ⟨17221⟩⟩
def transferEvent : Nat := 186901
def frameStart : Nat := 186828
def rule : BoundRule := .sum [.predecessor 0 186899 .coefficient, .predecessor 1 186900 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186899 .coefficient)
      LeftAuthority186897.bound (LeftAuthority186897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186900 .coefficient)
      LeftBound186893.bound (LeftBound186893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority186897.bound, LeftBound186893.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186897.bound, LeftBound186893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority186897.actual selector witness, LeftBound186893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186901

namespace LeftBound186905
def owner : Owner := ⟨.program ⟨257⟩, ⟨17846⟩⟩
def transferEvent : Nat := 186905
def frameStart : Nat := 186828
def rule : BoundRule := .product (.predecessor 0 186903 .coefficient) (.predecessor 1 186904 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186903 .coefficient)
      LeftBound186901.bound (LeftBound186901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186904 .coefficient)
      LeftAuthority186878.bound (LeftAuthority186878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186901.bound LeftAuthority186878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186901.bound, LeftAuthority186878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186901.actual selector witness) * (LeftAuthority186878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186905

namespace LeftBound186916
def owner : Owner := ⟨.program ⟨257⟩, ⟨16084⟩⟩
def transferEvent : Nat := 186916
def frameStart : Nat := 186828
def rule : BoundRule := .product (.predecessor 0 186914 .coefficient) (.predecessor 1 186915 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186914 .coefficient)
      LeftAuthority186889.bound (LeftAuthority186889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186915 .coefficient)
      LeftAuthority186912.bound (LeftAuthority186912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186912.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority186889.bound LeftAuthority186912.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186889.bound, LeftAuthority186912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority186889.actual selector witness) * (LeftAuthority186912.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186916

namespace LeftBound186924
def owner : Owner := ⟨.program ⟨257⟩, ⟨16085⟩⟩
def transferEvent : Nat := 186924
def frameStart : Nat := 186828
def rule : BoundRule := .sum [.predecessor 0 186922 .coefficient, .predecessor 1 186923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186922 .coefficient)
      LeftAuthority186920.bound (LeftAuthority186920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186923 .coefficient)
      LeftBound186916.bound (LeftBound186916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority186920.bound, LeftBound186916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186920.bound, LeftBound186916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority186920.actual selector witness, LeftBound186916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186924

namespace LeftBound186928
def owner : Owner := ⟨.program ⟨257⟩, ⟨17849⟩⟩
def transferEvent : Nat := 186928
def frameStart : Nat := 186828
def rule : BoundRule := .sum [.predecessor 0 186926 .coefficient, .predecessor 1 186927 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186926 .coefficient)
      LeftBound186924.bound (LeftBound186924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186927 .coefficient)
      LeftBound186905.bound (LeftBound186905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186924.bound, LeftBound186905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186924.bound, LeftBound186905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186924.actual selector witness, LeftBound186905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186928

namespace LeftBound186941
def owner : Owner := ⟨.program ⟨257⟩, ⟨17848⟩⟩
def transferEvent : Nat := 186941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186939 .coefficient, .predecessor 1 186940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186939 .coefficient)
      LeftBound186770.bound (LeftBound186770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186940 .coefficient)
      LeftBound186753.bound (LeftBound186753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186770.bound, LeftBound186753.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186770.bound, LeftBound186753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186770.actual selector witness, LeftBound186753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186941

namespace LeftBound186944
def owner : Owner := ⟨.program ⟨257⟩, ⟨17848⟩⟩
def transferEvent : Nat := 186944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186938 .summary, .result 186760 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186938 .summary)
      LeftBound186772.bound (LeftBound186772.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16659⟩⟩) (rawTerms := some (Proof.Events730.exact186938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186760 .summary)
      LeftBound186755.bound (LeftBound186755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17847⟩⟩) (rawTerms := some (Proof.Events729.exact186760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186772.bound, LeftBound186755.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186772.bound, LeftBound186755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186772.actual selector witness, LeftBound186755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186944

namespace LeftBound186948
def owner : Owner := ⟨.program ⟨257⟩, ⟨20749⟩⟩
def transferEvent : Nat := 186948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186946 .coefficient, .predecessor 1 186947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186946 .coefficient)
      LeftBound186941.bound (LeftBound186941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186947 .coefficient)
      LeftBound186459.bound (LeftBound186459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events728.exact186463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186459.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186941.bound, LeftBound186459.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186941.bound, LeftBound186459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186941.actual selector witness, LeftBound186459.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186948

namespace LeftBound186949
def owner : Owner := ⟨.program ⟨257⟩, ⟨20749⟩⟩
def transferEvent : Nat := 186949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186945 .summary, .result 186463 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186945 .summary)
      LeftBound186944.bound (LeftBound186944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17848⟩⟩) (rawTerms := some (Proof.Events730.exact186945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186463 .summary)
      LeftBound186462.bound (LeftBound186462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20748⟩⟩) (rawTerms := some (Proof.Events728.exact186463RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186462.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186944.bound, LeftBound186462.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186944.bound, LeftBound186462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186944.actual selector witness, LeftBound186462.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186949

namespace LeftBound186953
def owner : Owner := ⟨.program ⟨257⟩, ⟨23969⟩⟩
def transferEvent : Nat := 186953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186951 .coefficient, .predecessor 1 186952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186951 .coefficient)
      LeftBound186948.bound (LeftBound186948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186952 .coefficient)
      LeftBound185977.bound (LeftBound185977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186948.bound, LeftBound185977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186948.bound, LeftBound185977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186948.actual selector witness, LeftBound185977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186953

namespace LeftBound186954
def owner : Owner := ⟨.program ⟨257⟩, ⟨23969⟩⟩
def transferEvent : Nat := 186954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186950 .summary, .result 185981 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186950 .summary)
      LeftBound186949.bound (LeftBound186949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20749⟩⟩) (rawTerms := some (Proof.Events730.exact186950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185981 .summary)
      LeftBound185980.bound (LeftBound185980.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23968⟩⟩) (rawTerms := some (Proof.Events726.exact185981RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186949.bound, LeftBound185980.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186949.bound, LeftBound185980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186949.actual selector witness, LeftBound185980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186954

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
