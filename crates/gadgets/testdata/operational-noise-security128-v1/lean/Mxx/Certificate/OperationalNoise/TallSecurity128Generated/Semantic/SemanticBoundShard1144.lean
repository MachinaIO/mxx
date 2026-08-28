import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1088
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1143

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound171165
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def transferEvent : Nat := 171165
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩ [⟨.result 170884 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 170884 .coefficient)
      LeftAuthority170883.bound (LeftAuthority170883.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23996⟩⟩) (rawTerms := some (Proof.Events667.exact170884RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority170883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority170883.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority170883.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority170883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority170883.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound171165

namespace LeftBound171166
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def transferEvent : Nat := 171166
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 171161 .summary) (.transfer 171165) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171161 .summary)
      LeftBound171160.bound (LeftBound171160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23485⟩⟩) (rawTerms := some (Proof.Events668.exact171161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound171160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 171165)
      LeftBound171165.bound (LeftBound171165.actual selector witness) := by
  exact .transfer (LeftBound171165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound171160.bound LeftBound171165.bound
def bound : CoeffClass := .finite ⟨32189003662929192193909661368320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171160.bound, LeftBound171165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound171160.actual selector witness) * (LeftBound171165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171166

namespace LeftBound171177
def owner : Owner := ⟨.program ⟨257⟩, ⟨22758⟩⟩
def transferEvent : Nat := 171177
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 171175 .coefficient) (.value (.predecessor 1 171176 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171175 .coefficient)
      LeftAuthority171173.bound (LeftAuthority171173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171176 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority171173.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171173.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority171173.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound171177

namespace LeftBound171181
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def transferEvent : Nat := 171181
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 171179 .coefficient) (.predecessor 1 171180 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171179 .coefficient)
      LeftBound163742.bound (LeftBound163742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171180 .coefficient)
      LeftBound171177.bound (LeftBound171177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163742.bound LeftBound171177.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163742.bound, LeftBound171177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163742.actual selector witness) * (LeftBound171177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171181

namespace LeftBound171182
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def transferEvent : Nat := 171182
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩ [⟨.result 171174 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171174 .coefficient)
      LeftAuthority171173.bound (LeftAuthority171173.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22756⟩⟩) (rawTerms := some (Proof.Events668.exact171174RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171173.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority171173.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority171173.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound171182

namespace LeftBound171183
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def transferEvent : Nat := 171183
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163745 .summary) (.transfer 171182) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163745 .summary)
      LeftBound163743.bound (LeftBound163743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6466⟩⟩) (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 171182)
      LeftBound171182.bound (LeftBound171182.actual selector witness) := by
  exact .transfer (LeftBound171182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163743.bound LeftBound171182.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163743.bound, LeftBound171182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163743.actual selector witness) * (LeftBound171182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171183

namespace LeftBound171278
def owner : Owner := ⟨.program ⟨257⟩, ⟨21841⟩⟩
def transferEvent : Nat := 171278
def frameStart : Nat := 171239
def rule : BoundRule := .identity (.predecessor 0 171277 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171277 .coefficient)
      LeftAuthority171275.bound (LeftAuthority171275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171275.derived selector witness)

def rawBound : CoeffClass := LeftAuthority171275.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority171275.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound171278

namespace LeftBound171295
def owner : Owner := ⟨.program ⟨257⟩, ⟨23302⟩⟩
def transferEvent : Nat := 171295
def frameStart : Nat := 171239
def rule : BoundRule := .sum [.predecessor 0 171293 .coefficient, .predecessor 1 171294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171293 .coefficient)
      LeftBound171278.bound (LeftBound171278.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound171278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171294 .coefficient)
      LeftAuthority171291.bound (LeftAuthority171291.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority171291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171278.bound, LeftAuthority171291.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171278.bound, LeftAuthority171291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171278.actual selector witness, LeftAuthority171291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171295

namespace LeftBound171298
def owner : Owner := ⟨.program ⟨257⟩, ⟨23303⟩⟩
def transferEvent : Nat := 171298
def frameStart : Nat := 171239
def rule : BoundRule := .identity (.predecessor 0 171297 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171297 .coefficient)
      LeftBound171295.bound (LeftBound171295.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound171295.derived selector witness)

def rawBound : CoeffClass := LeftBound171295.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound171295.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound171298

namespace LeftBound171304
def owner : Owner := ⟨.program ⟨257⟩, ⟨23304⟩⟩
def transferEvent : Nat := 171304
def frameStart : Nat := 171239
def rule : BoundRule := .product (.predecessor 0 171302 .coefficient) (.predecessor 1 171303 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171302 .coefficient)
      LeftAuthority171300.bound (LeftAuthority171300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171303 .coefficient)
      LeftBound171298.bound (LeftBound171298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171298.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority171300.bound LeftBound171298.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171300.bound, LeftBound171298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority171300.actual selector witness) * (LeftBound171298.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171304

namespace LeftBound171312
def owner : Owner := ⟨.program ⟨257⟩, ⟨23305⟩⟩
def transferEvent : Nat := 171312
def frameStart : Nat := 171239
def rule : BoundRule := .sum [.predecessor 0 171310 .coefficient, .predecessor 1 171311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171310 .coefficient)
      LeftAuthority171308.bound (LeftAuthority171308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171311 .coefficient)
      LeftBound171304.bound (LeftBound171304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171304.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority171308.bound, LeftBound171304.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171308.bound, LeftBound171304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority171308.actual selector witness, LeftBound171304.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171312

namespace LeftBound171316
def owner : Owner := ⟨.program ⟨257⟩, ⟨23997⟩⟩
def transferEvent : Nat := 171316
def frameStart : Nat := 171239
def rule : BoundRule := .product (.predecessor 0 171314 .coefficient) (.predecessor 1 171315 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171314 .coefficient)
      LeftBound171312.bound (LeftBound171312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171315 .coefficient)
      LeftAuthority171289.bound (LeftAuthority171289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound171312.bound LeftAuthority171289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171312.bound, LeftAuthority171289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound171312.actual selector witness) * (LeftAuthority171289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171316

namespace LeftBound171327
def owner : Owner := ⟨.program ⟨257⟩, ⟨22164⟩⟩
def transferEvent : Nat := 171327
def frameStart : Nat := 171239
def rule : BoundRule := .product (.predecessor 0 171325 .coefficient) (.predecessor 1 171326 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171325 .coefficient)
      LeftAuthority171300.bound (LeftAuthority171300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171300.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171326 .coefficient)
      LeftAuthority171323.bound (LeftAuthority171323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171323.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171323.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority171300.bound LeftAuthority171323.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171300.bound, LeftAuthority171323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority171300.actual selector witness) * (LeftAuthority171323.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171327

namespace LeftBound171335
def owner : Owner := ⟨.program ⟨257⟩, ⟨22165⟩⟩
def transferEvent : Nat := 171335
def frameStart : Nat := 171239
def rule : BoundRule := .sum [.predecessor 0 171333 .coefficient, .predecessor 1 171334 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171333 .coefficient)
      LeftAuthority171331.bound (LeftAuthority171331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171331.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171334 .coefficient)
      LeftBound171327.bound (LeftBound171327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority171331.bound, LeftBound171327.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171331.bound, LeftBound171327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority171331.actual selector witness, LeftBound171327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171335

namespace LeftBound171339
def owner : Owner := ⟨.program ⟨257⟩, ⟨24001⟩⟩
def transferEvent : Nat := 171339
def frameStart : Nat := 171239
def rule : BoundRule := .sum [.predecessor 0 171337 .coefficient, .predecessor 1 171338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171337 .coefficient)
      LeftBound171335.bound (LeftBound171335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171338 .coefficient)
      LeftBound171316.bound (LeftBound171316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171335.bound, LeftBound171316.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171335.bound, LeftBound171316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171335.actual selector witness, LeftBound171316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171339

namespace LeftBound171352
def owner : Owner := ⟨.program ⟨257⟩, ⟨23999⟩⟩
def transferEvent : Nat := 171352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 171350 .coefficient, .predecessor 1 171351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171350 .coefficient)
      LeftBound171181.bound (LeftBound171181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events669.exact171349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171351 .coefficient)
      LeftBound171164.bound (LeftBound171164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171181.bound, LeftBound171164.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171181.bound, LeftBound171164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171181.actual selector witness, LeftBound171164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171352

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
