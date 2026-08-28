import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1890

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound280091
def owner : Owner := ⟨.program ⟨257⟩, ⟨17523⟩⟩
def transferEvent : Nat := 280091
def frameStart : Nat := 280014
def rule : BoundRule := .product (.predecessor 0 280089 .coefficient) (.predecessor 1 280090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280089 .coefficient)
      LeftBound280087.bound (LeftBound280087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280090 .coefficient)
      LeftAuthority280064.bound (LeftAuthority280064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280064.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280087.bound LeftAuthority280064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280087.bound, LeftAuthority280064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280087.actual selector witness) * (LeftAuthority280064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280091

namespace LeftBound280102
def owner : Owner := ⟨.program ⟨257⟩, ⟨15901⟩⟩
def transferEvent : Nat := 280102
def frameStart : Nat := 280014
def rule : BoundRule := .product (.predecessor 0 280100 .coefficient) (.predecessor 1 280101 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280100 .coefficient)
      LeftAuthority280075.bound (LeftAuthority280075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280101 .coefficient)
      LeftAuthority280098.bound (LeftAuthority280098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280098.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority280075.bound LeftAuthority280098.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority280075.bound, LeftAuthority280098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority280075.actual selector witness) * (LeftAuthority280098.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280102

namespace LeftBound280110
def owner : Owner := ⟨.program ⟨257⟩, ⟨15902⟩⟩
def transferEvent : Nat := 280110
def frameStart : Nat := 280014
def rule : BoundRule := .sum [.predecessor 0 280108 .coefficient, .predecessor 1 280109 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280108 .coefficient)
      LeftAuthority280106.bound (LeftAuthority280106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280109 .coefficient)
      LeftBound280102.bound (LeftBound280102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority280106.bound, LeftBound280102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority280106.bound, LeftBound280102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority280106.actual selector witness, LeftBound280102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280110

namespace LeftBound280114
def owner : Owner := ⟨.program ⟨257⟩, ⟨17528⟩⟩
def transferEvent : Nat := 280114
def frameStart : Nat := 280014
def rule : BoundRule := .sum [.predecessor 0 280112 .coefficient, .predecessor 1 280113 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280112 .coefficient)
      LeftBound280110.bound (LeftBound280110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280113 .coefficient)
      LeftBound280091.bound (LeftBound280091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280091.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280110.bound, LeftBound280091.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280110.bound, LeftBound280091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280110.actual selector witness, LeftBound280091.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280114

namespace LeftBound280127
def owner : Owner := ⟨.program ⟨257⟩, ⟨17525⟩⟩
def transferEvent : Nat := 280127
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280125 .coefficient, .predecessor 1 280126 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280125 .coefficient)
      LeftBound279956.bound (LeftBound279956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280126 .coefficient)
      LeftBound279939.bound (LeftBound279939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1093.exact279946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound279939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound279939.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound279956.bound, LeftBound279939.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound279956.bound, LeftBound279939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound279956.actual selector witness, LeftBound279939.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280127

namespace LeftBound280130
def owner : Owner := ⟨.program ⟨257⟩, ⟨17525⟩⟩
def transferEvent : Nat := 280130
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280124 .summary, .result 279946 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280124 .summary)
      LeftBound279958.bound (LeftBound279958.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16429⟩⟩) (rawTerms := some (Proof.Events1094.exact280124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 279946 .summary)
      LeftBound279941.bound (LeftBound279941.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17524⟩⟩) (rawTerms := some (Proof.Events1093.exact279946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound279941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound279958.bound, LeftBound279941.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound279958.bound, LeftBound279941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound279958.actual selector witness, LeftBound279941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280130

namespace LeftBound280134
def owner : Owner := ⟨.program ⟨257⟩, ⟨17526⟩⟩
def transferEvent : Nat := 280134
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 280132 .coefficient) (.predecessor 1 280133 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280132 .coefficient)
      LeftBound280127.bound (LeftBound280127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280133 .coefficient)
      LeftBound15881.bound (LeftBound15881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280127.bound LeftBound15881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280127.bound, LeftBound15881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280127.actual selector witness) * (LeftBound15881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280134

namespace LeftBound280135
def owner : Owner := ⟨.program ⟨257⟩, ⟨17526⟩⟩
def transferEvent : Nat := 280135
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩ [⟨.result 15878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15878 .coefficient)
      LeftAuthority15877.bound (LeftAuthority15877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7171⟩⟩) (rawTerms := some (Proof.Events062.exact15878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15877.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound280135

namespace LeftBound280136
def owner : Owner := ⟨.program ⟨257⟩, ⟨17526⟩⟩
def transferEvent : Nat := 280136
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280131 .summary) (.transfer 280135) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280131 .summary)
      LeftBound280130.bound (LeftBound280130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17525⟩⟩) (rawTerms := some (Proof.Events1094.exact280131RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 280135)
      LeftBound280135.bound (LeftBound280135.actual selector witness) := by
  exact .transfer (LeftBound280135.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280130.bound LeftBound280135.bound
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280130.bound, LeftBound280135.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280130.actual selector witness) * (LeftBound280135.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280136

namespace LeftBound280144
def owner : Owner := ⟨.program ⟨257⟩, ⟨7060⟩⟩
def transferEvent : Nat := 280144
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 280142 .coefficient) (.predecessor 1 280143 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280142 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280143 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound280144

namespace LeftBound280149
def owner : Owner := ⟨.program ⟨257⟩, ⟨7648⟩⟩
def transferEvent : Nat := 280149
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 280147 .coefficient) (.predecessor 1 280148 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280147 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280148 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound15895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound15895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280149

namespace LeftBound280154
def owner : Owner := ⟨.program ⟨257⟩, ⟨9289⟩⟩
def transferEvent : Nat := 280154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280152 .coefficient, .predecessor 1 280153 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280152 .coefficient)
      LeftBound280149.bound (LeftBound280149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280153 .coefficient)
      LeftBound280144.bound (LeftBound280144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280149.bound, LeftBound280144.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280149.bound, LeftBound280144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280149.actual selector witness, LeftBound280144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280154

namespace LeftBound280158
def owner : Owner := ⟨.program ⟨257⟩, ⟨9290⟩⟩
def transferEvent : Nat := 280158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280156 .coefficient, .predecessor 1 280157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280156 .coefficient)
      LeftBound280154.bound (LeftBound280154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280154.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280157 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280154.bound, LeftBound31515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280154.bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280154.actual selector witness, LeftBound31515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280158

namespace LeftBound280159
def owner : Owner := ⟨.program ⟨257⟩, ⟨9290⟩⟩
def transferEvent : Nat := 280159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩ [⟨.result 31516 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31516 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨118⟩⟩) (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound31515.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound31515.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound280159

namespace LeftBound280164
def owner : Owner := ⟨.program ⟨257⟩, ⟨9452⟩⟩
def transferEvent : Nat := 280164
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280162 .coefficient, .predecessor 1 280163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280162 .coefficient)
      LeftBound280158.bound (LeftBound280158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280163 .coefficient)
      LeftBound280158.bound (LeftBound280158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280158.bound, LeftBound280158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280158.bound, LeftBound280158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280158.actual selector witness, LeftBound280158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280164

namespace LeftBound280167
def owner : Owner := ⟨.program ⟨257⟩, ⟨9452⟩⟩
def transferEvent : Nat := 280167
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280161 .summary, .result 280161 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280161 .summary)
      LeftBound280159.bound (LeftBound280159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9290⟩⟩) (rawTerms := some (Proof.Events1094.exact280161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280161 .summary)
      LeftBound280159.bound (LeftBound280159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9290⟩⟩) (rawTerms := some (Proof.Events1094.exact280161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280159.bound, LeftBound280159.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280159.bound, LeftBound280159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280159.actual selector witness, LeftBound280159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280167

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
