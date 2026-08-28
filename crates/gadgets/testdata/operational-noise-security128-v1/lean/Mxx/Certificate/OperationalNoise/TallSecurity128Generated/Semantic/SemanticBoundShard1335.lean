import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1334

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound198863
def owner : Owner := ⟨.program ⟨257⟩, ⟨53580⟩⟩
def transferEvent : Nat := 198863
def frameStart : Nat := 198834
def rule : BoundRule := .product (.predecessor 0 198861 .coefficient) (.predecessor 1 198862 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198861 .coefficient)
      LeftAuthority198859.bound (LeftAuthority198859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198862 .coefficient)
      LeftAuthority198856.bound (LeftAuthority198856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority198859.bound LeftAuthority198856.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198859.bound, LeftAuthority198856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority198859.actual selector witness) * (LeftAuthority198856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198863

namespace LeftBound198867
def owner : Owner := ⟨.program ⟨257⟩, ⟨53581⟩⟩
def transferEvent : Nat := 198867
def frameStart : Nat := 198834
def rule : BoundRule := .identity (.predecessor 0 198866 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198866 .coefficient)
      LeftBound198863.bound (LeftBound198863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198863.derived selector witness)

def rawBound : CoeffClass := LeftBound198863.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound198863.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound198867

namespace LeftBound198884
def owner : Owner := ⟨.program ⟨257⟩, ⟨55274⟩⟩
def transferEvent : Nat := 198884
def frameStart : Nat := 198834
def rule : BoundRule := .sum [.predecessor 0 198882 .coefficient, .predecessor 1 198883 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198882 .coefficient)
      LeftBound198867.bound (LeftBound198867.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound198867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198883 .coefficient)
      LeftAuthority198880.bound (LeftAuthority198880.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority198880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound198867.bound, LeftAuthority198880.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198867.bound, LeftAuthority198880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound198867.actual selector witness, LeftAuthority198880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198884

namespace LeftBound198887
def owner : Owner := ⟨.program ⟨257⟩, ⟨55275⟩⟩
def transferEvent : Nat := 198887
def frameStart : Nat := 198834
def rule : BoundRule := .identity (.predecessor 0 198886 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198886 .coefficient)
      LeftBound198884.bound (LeftBound198884.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound198884.derived selector witness)

def rawBound : CoeffClass := LeftBound198884.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound198884.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound198887

namespace LeftBound198893
def owner : Owner := ⟨.program ⟨257⟩, ⟨55276⟩⟩
def transferEvent : Nat := 198893
def frameStart : Nat := 198834
def rule : BoundRule := .product (.predecessor 0 198891 .coefficient) (.predecessor 1 198892 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198891 .coefficient)
      LeftAuthority198889.bound (LeftAuthority198889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198892 .coefficient)
      LeftBound198887.bound (LeftBound198887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198887.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority198889.bound LeftBound198887.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198889.bound, LeftBound198887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority198889.actual selector witness) * (LeftBound198887.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198893

namespace LeftBound198909
def owner : Owner := ⟨.program ⟨257⟩, ⟨9530⟩⟩
def transferEvent : Nat := 198909
def frameStart : Nat := 198834
def rule : BoundRule := .scale (.predecessor 0 198907 .coefficient) (.value (.predecessor 1 198908 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198907 .coefficient)
      LeftAuthority198905.bound (LeftAuthority198905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198908 .coefficient)
      LeftAuthority198896.bound (LeftAuthority198896.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority198896.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority198905.bound LeftAuthority198896.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198905.bound, LeftAuthority198896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority198905.actual selector witness) * (LeftAuthority198896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound198909

namespace LeftBound198912
def owner : Owner := ⟨.program ⟨257⟩, ⟨7289⟩⟩
def transferEvent : Nat := 198912
def frameStart : Nat := 198834
def rule : BoundRule := .identity (.predecessor 0 198911 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198911 .coefficient)
      LeftAuthority198899.bound (LeftAuthority198899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198899.derived selector witness)

def rawBound : CoeffClass := LeftAuthority198899.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority198899.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound198912

namespace LeftBound198916
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def transferEvent : Nat := 198916
def frameStart : Nat := 198834
def rule : BoundRule := .product (.predecessor 0 198914 .coefficient) (.predecessor 1 198915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198914 .coefficient)
      LeftBound198912.bound (LeftBound198912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198915 .coefficient)
      LeftBound198909.bound (LeftBound198909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198909.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound198912.bound LeftBound198909.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198912.bound, LeftBound198909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound198912.actual selector witness) * (LeftBound198909.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198916

namespace LeftBound198921
def owner : Owner := ⟨.program ⟨257⟩, ⟨55277⟩⟩
def transferEvent : Nat := 198921
def frameStart : Nat := 198834
def rule : BoundRule := .sum [.predecessor 0 198919 .coefficient, .predecessor 1 198920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198919 .coefficient)
      LeftBound198916.bound (LeftBound198916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198920 .coefficient)
      LeftBound198893.bound (LeftBound198893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound198916.bound, LeftBound198893.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198916.bound, LeftBound198893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound198916.actual selector witness, LeftBound198893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198921

namespace LeftBound198925
def owner : Owner := ⟨.program ⟨257⟩, ⟨55524⟩⟩
def transferEvent : Nat := 198925
def frameStart : Nat := 198834
def rule : BoundRule := .product (.predecessor 0 198923 .coefficient) (.predecessor 1 198924 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198923 .coefficient)
      LeftBound198921.bound (LeftBound198921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198924 .coefficient)
      LeftAuthority198878.bound (LeftAuthority198878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound198921.bound LeftAuthority198878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198921.bound, LeftAuthority198878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound198921.actual selector witness) * (LeftAuthority198878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198925

namespace LeftBound198936
def owner : Owner := ⟨.program ⟨257⟩, ⟨53886⟩⟩
def transferEvent : Nat := 198936
def frameStart : Nat := 198834
def rule : BoundRule := .product (.predecessor 0 198934 .coefficient) (.predecessor 1 198935 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198934 .coefficient)
      LeftAuthority198889.bound (LeftAuthority198889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198935 .coefficient)
      LeftAuthority198932.bound (LeftAuthority198932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority198889.bound LeftAuthority198932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198889.bound, LeftAuthority198932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority198889.actual selector witness) * (LeftAuthority198932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198936

namespace LeftBound198944
def owner : Owner := ⟨.program ⟨257⟩, ⟨53887⟩⟩
def transferEvent : Nat := 198944
def frameStart : Nat := 198834
def rule : BoundRule := .sum [.predecessor 0 198942 .coefficient, .predecessor 1 198943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198942 .coefficient)
      LeftAuthority198940.bound (LeftAuthority198940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198943 .coefficient)
      LeftBound198936.bound (LeftBound198936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority198940.bound, LeftBound198936.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority198940.bound, LeftBound198936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority198940.actual selector witness, LeftBound198936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198944

namespace LeftBound198948
def owner : Owner := ⟨.program ⟨257⟩, ⟨55525⟩⟩
def transferEvent : Nat := 198948
def frameStart : Nat := 198834
def rule : BoundRule := .sum [.predecessor 0 198946 .coefficient, .predecessor 1 198947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198946 .coefficient)
      LeftBound198944.bound (LeftBound198944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198947 .coefficient)
      LeftBound198925.bound (LeftBound198925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound198944.bound, LeftBound198925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198944.bound, LeftBound198925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound198944.actual selector witness, LeftBound198925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198948

namespace LeftBound198961
def owner : Owner := ⟨.program ⟨257⟩, ⟨55523⟩⟩
def transferEvent : Nat := 198961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 198959 .coefficient, .predecessor 1 198960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198959 .coefficient)
      LeftBound198782.bound (LeftBound198782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198960 .coefficient)
      LeftBound198765.bound (LeftBound198765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound198782.bound, LeftBound198765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198782.bound, LeftBound198765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound198782.actual selector witness, LeftBound198765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198961

namespace LeftBound198964
def owner : Owner := ⟨.program ⟨257⟩, ⟨55523⟩⟩
def transferEvent : Nat := 198964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 198958 .summary, .result 198772 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 198958 .summary)
      LeftBound198784.bound (LeftBound198784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54452⟩⟩) (rawTerms := some (Proof.Events777.exact198958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound198784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 198772 .summary)
      LeftBound198767.bound (LeftBound198767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55522⟩⟩) (rawTerms := some (Proof.Events776.exact198772RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound198767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound198784.bound, LeftBound198767.bound]
def bound : CoeffClass := .finite ⟨2997907760060573155328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198784.bound, LeftBound198767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound198784.actual selector witness, LeftBound198767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound198964

namespace LeftBound198968
def owner : Owner := ⟨.program ⟨257⟩, ⟨55996⟩⟩
def transferEvent : Nat := 198968
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 198966 .coefficient) (.predecessor 1 198967 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 198966 .coefficient)
      LeftBound198961.bound (LeftBound198961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact198965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 198967 .coefficient)
      LeftAuthority198687.bound (LeftAuthority198687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority198687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority198687.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound198961.bound LeftAuthority198687.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound198961.bound, LeftAuthority198687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound198961.actual selector witness) * (LeftAuthority198687.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound198968

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
