import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1393

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound207811
def owner : Owner := ⟨.program ⟨257⟩, ⟨50031⟩⟩
def transferEvent : Nat := 207811
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207806 .summary) (.transfer 207810) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207806 .summary)
      LeftBound207805.bound (LeftBound207805.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49661⟩⟩) (rawTerms := some (Proof.Events811.exact207806RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 207810)
      LeftBound207810.bound (LeftBound207810.actual selector witness) := by
  exact .transfer (LeftBound207810.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound207805.bound LeftBound207810.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207805.bound, LeftBound207810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound207805.actual selector witness) * (LeftBound207810.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207811

namespace LeftBound207822
def owner : Owner := ⟨.program ⟨257⟩, ⟨48898⟩⟩
def transferEvent : Nat := 207822
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 207820 .coefficient) (.value (.predecessor 1 207821 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207820 .coefficient)
      LeftAuthority207818.bound (LeftAuthority207818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207818.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207818.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207821 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority207818.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207818.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority207818.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound207822

namespace LeftBound207826
def owner : Owner := ⟨.program ⟨257⟩, ⟨48899⟩⟩
def transferEvent : Nat := 207826
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 207824 .coefficient) (.predecessor 1 207825 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207824 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207825 .coefficient)
      LeftBound207822.bound (LeftBound207822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207822.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound207822.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound207822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound207822.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207826

namespace LeftBound207827
def owner : Owner := ⟨.program ⟨257⟩, ⟨48899⟩⟩
def transferEvent : Nat := 207827
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨48896⟩⟩]⟩ [⟨.result 207819 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207819 .coefficient)
      LeftAuthority207818.bound (LeftAuthority207818.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48896⟩⟩) (rawTerms := some (Proof.Events811.exact207819RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207818.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207818.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority207818.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority207818.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound207827

namespace LeftBound207828
def owner : Owner := ⟨.program ⟨257⟩, ⟨48899⟩⟩
def transferEvent : Nat := 207828
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 207827) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 207827)
      LeftBound207827.bound (LeftBound207827.actual selector witness) := by
  exact .transfer (LeftBound207827.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound207827.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound207827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound207827.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207828

namespace LeftBound207923
def owner : Owner := ⟨.program ⟨257⟩, ⟨48149⟩⟩
def transferEvent : Nat := 207923
def frameStart : Nat := 207884
def rule : BoundRule := .identity (.predecessor 0 207922 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207922 .coefficient)
      LeftAuthority207920.bound (LeftAuthority207920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207920.derived selector witness)

def rawBound : CoeffClass := LeftAuthority207920.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority207920.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound207923

namespace LeftBound207940
def owner : Owner := ⟨.program ⟨257⟩, ⟨49506⟩⟩
def transferEvent : Nat := 207940
def frameStart : Nat := 207884
def rule : BoundRule := .sum [.predecessor 0 207938 .coefficient, .predecessor 1 207939 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207938 .coefficient)
      LeftBound207923.bound (LeftBound207923.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound207923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207939 .coefficient)
      LeftAuthority207936.bound (LeftAuthority207936.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority207936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207923.bound, LeftAuthority207936.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207923.bound, LeftAuthority207936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207923.actual selector witness, LeftAuthority207936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207940

namespace LeftBound207943
def owner : Owner := ⟨.program ⟨257⟩, ⟨49507⟩⟩
def transferEvent : Nat := 207943
def frameStart : Nat := 207884
def rule : BoundRule := .identity (.predecessor 0 207942 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207942 .coefficient)
      LeftBound207940.bound (LeftBound207940.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound207940.derived selector witness)

def rawBound : CoeffClass := LeftBound207940.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound207940.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound207943

namespace LeftBound207949
def owner : Owner := ⟨.program ⟨257⟩, ⟨49508⟩⟩
def transferEvent : Nat := 207949
def frameStart : Nat := 207884
def rule : BoundRule := .product (.predecessor 0 207947 .coefficient) (.predecessor 1 207948 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207947 .coefficient)
      LeftAuthority207945.bound (LeftAuthority207945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207948 .coefficient)
      LeftBound207943.bound (LeftBound207943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority207945.bound LeftBound207943.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207945.bound, LeftBound207943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority207945.actual selector witness) * (LeftBound207943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207949

namespace LeftBound207957
def owner : Owner := ⟨.program ⟨257⟩, ⟨49509⟩⟩
def transferEvent : Nat := 207957
def frameStart : Nat := 207884
def rule : BoundRule := .sum [.predecessor 0 207955 .coefficient, .predecessor 1 207956 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207955 .coefficient)
      LeftAuthority207953.bound (LeftAuthority207953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207956 .coefficient)
      LeftBound207949.bound (LeftBound207949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority207953.bound, LeftBound207949.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207953.bound, LeftBound207949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority207953.actual selector witness, LeftBound207949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207957

namespace LeftBound207961
def owner : Owner := ⟨.program ⟨257⟩, ⟨50030⟩⟩
def transferEvent : Nat := 207961
def frameStart : Nat := 207884
def rule : BoundRule := .product (.predecessor 0 207959 .coefficient) (.predecessor 1 207960 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207959 .coefficient)
      LeftBound207957.bound (LeftBound207957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207960 .coefficient)
      LeftAuthority207934.bound (LeftAuthority207934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207934.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound207957.bound LeftAuthority207934.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207957.bound, LeftAuthority207934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound207957.actual selector witness) * (LeftAuthority207934.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207961

namespace LeftBound207972
def owner : Owner := ⟨.program ⟨257⟩, ⟨48364⟩⟩
def transferEvent : Nat := 207972
def frameStart : Nat := 207884
def rule : BoundRule := .product (.predecessor 0 207970 .coefficient) (.predecessor 1 207971 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207970 .coefficient)
      LeftAuthority207945.bound (LeftAuthority207945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207971 .coefficient)
      LeftAuthority207968.bound (LeftAuthority207968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority207945.bound LeftAuthority207968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207945.bound, LeftAuthority207968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority207945.actual selector witness) * (LeftAuthority207968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207972

namespace LeftBound207980
def owner : Owner := ⟨.program ⟨257⟩, ⟨48365⟩⟩
def transferEvent : Nat := 207980
def frameStart : Nat := 207884
def rule : BoundRule := .sum [.predecessor 0 207978 .coefficient, .predecessor 1 207979 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207978 .coefficient)
      LeftAuthority207976.bound (LeftAuthority207976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207979 .coefficient)
      LeftBound207972.bound (LeftBound207972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority207976.bound, LeftBound207972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207976.bound, LeftBound207972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority207976.actual selector witness, LeftBound207972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207980

namespace LeftBound207984
def owner : Owner := ⟨.program ⟨257⟩, ⟨50033⟩⟩
def transferEvent : Nat := 207984
def frameStart : Nat := 207884
def rule : BoundRule := .sum [.predecessor 0 207982 .coefficient, .predecessor 1 207983 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207982 .coefficient)
      LeftBound207980.bound (LeftBound207980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207983 .coefficient)
      LeftBound207961.bound (LeftBound207961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207980.bound, LeftBound207961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207980.bound, LeftBound207961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207980.actual selector witness, LeftBound207961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207984

namespace LeftBound207997
def owner : Owner := ⟨.program ⟨257⟩, ⟨50032⟩⟩
def transferEvent : Nat := 207997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207995 .coefficient, .predecessor 1 207996 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207995 .coefficient)
      LeftBound207826.bound (LeftBound207826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact207994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207996 .coefficient)
      LeftBound207809.bound (LeftBound207809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207809.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207826.bound, LeftBound207809.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207826.bound, LeftBound207809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207826.actual selector witness, LeftBound207809.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207997

namespace LeftBound208000
def owner : Owner := ⟨.program ⟨257⟩, ⟨50032⟩⟩
def transferEvent : Nat := 208000
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207994 .summary, .result 207816 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207994 .summary)
      LeftBound207828.bound (LeftBound207828.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48899⟩⟩) (rawTerms := some (Proof.Events812.exact207994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207816 .summary)
      LeftBound207811.bound (LeftBound207811.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50031⟩⟩) (rawTerms := some (Proof.Events811.exact207816RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207811.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207828.bound, LeftBound207811.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207828.bound, LeftBound207811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207828.actual selector witness, LeftBound207811.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound208000

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
