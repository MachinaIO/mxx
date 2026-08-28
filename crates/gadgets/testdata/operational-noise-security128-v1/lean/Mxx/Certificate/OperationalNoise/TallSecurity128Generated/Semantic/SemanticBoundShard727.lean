import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard725
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard726

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound111186
def owner : Owner := ⟨.program ⟨257⟩, ⟨53878⟩⟩
def transferEvent : Nat := 111186
def frameStart : Nat := 111084
def rule : BoundRule := .product (.predecessor 0 111184 .coefficient) (.predecessor 1 111185 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111184 .coefficient)
      LeftAuthority111139.bound (LeftAuthority111139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111185 .coefficient)
      LeftAuthority111182.bound (LeftAuthority111182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority111139.bound LeftAuthority111182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111139.bound, LeftAuthority111182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority111139.actual selector witness) * (LeftAuthority111182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111186

namespace LeftBound111194
def owner : Owner := ⟨.program ⟨257⟩, ⟨53879⟩⟩
def transferEvent : Nat := 111194
def frameStart : Nat := 111084
def rule : BoundRule := .sum [.predecessor 0 111192 .coefficient, .predecessor 1 111193 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111192 .coefficient)
      LeftAuthority111190.bound (LeftAuthority111190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111193 .coefficient)
      LeftBound111186.bound (LeftBound111186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority111190.bound, LeftBound111186.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111190.bound, LeftBound111186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority111190.actual selector witness, LeftBound111186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111194

namespace LeftBound111198
def owner : Owner := ⟨.program ⟨257⟩, ⟨55514⟩⟩
def transferEvent : Nat := 111198
def frameStart : Nat := 111084
def rule : BoundRule := .sum [.predecessor 0 111196 .coefficient, .predecessor 1 111197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111196 .coefficient)
      LeftBound111194.bound (LeftBound111194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111197 .coefficient)
      LeftBound111175.bound (LeftBound111175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111194.bound, LeftBound111175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111194.bound, LeftBound111175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111194.actual selector witness, LeftBound111175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111198

namespace LeftBound111211
def owner : Owner := ⟨.program ⟨257⟩, ⟨55512⟩⟩
def transferEvent : Nat := 111211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 111209 .coefficient, .predecessor 1 111210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111209 .coefficient)
      LeftBound111032.bound (LeftBound111032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111210 .coefficient)
      LeftBound111015.bound (LeftBound111015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events433.exact111022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111032.bound, LeftBound111015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111032.bound, LeftBound111015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111032.actual selector witness, LeftBound111015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111211

namespace LeftBound111214
def owner : Owner := ⟨.program ⟨257⟩, ⟨55512⟩⟩
def transferEvent : Nat := 111214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 111208 .summary, .result 111022 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111208 .summary)
      LeftBound111034.bound (LeftBound111034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54442⟩⟩) (rawTerms := some (Proof.Events434.exact111208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111022 .summary)
      LeftBound111017.bound (LeftBound111017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55511⟩⟩) (rawTerms := some (Proof.Events433.exact111022RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111034.bound, LeftBound111017.bound]
def bound : CoeffClass := .finite ⟨2997907760060573155328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111034.bound, LeftBound111017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111034.actual selector witness, LeftBound111017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111214

namespace LeftBound111218
def owner : Owner := ⟨.program ⟨257⟩, ⟨55965⟩⟩
def transferEvent : Nat := 111218
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 111216 .coefficient) (.predecessor 1 111217 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111216 .coefficient)
      LeftBound111211.bound (LeftBound111211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111217 .coefficient)
      LeftAuthority110937.bound (LeftAuthority110937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events433.exact110938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority110937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority110937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound111211.bound LeftAuthority110937.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111211.bound, LeftAuthority110937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound111211.actual selector witness) * (LeftAuthority110937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111218

namespace LeftBound111219
def owner : Owner := ⟨.program ⟨257⟩, ⟨55965⟩⟩
def transferEvent : Nat := 111219
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨55963⟩⟩]⟩ [⟨.result 110938 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110938 .coefficient)
      LeftAuthority110937.bound (LeftAuthority110937.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨55963⟩⟩) (rawTerms := some (Proof.Events433.exact110938RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority110937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority110937.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority110937.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority110937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority110937.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound111219

namespace LeftBound111220
def owner : Owner := ⟨.program ⟨257⟩, ⟨55965⟩⟩
def transferEvent : Nat := 111220
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 111215 .summary) (.transfer 111219) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111215 .summary)
      LeftBound111214.bound (LeftBound111214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55512⟩⟩) (rawTerms := some (Proof.Events434.exact111215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 111219)
      LeftBound111219.bound (LeftBound111219.actual selector witness) := by
  exact .transfer (LeftBound111219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound111214.bound LeftBound111219.bound
def bound : CoeffClass := .finite ⟨32189789464711941702873220382720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111214.bound, LeftBound111219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound111214.actual selector witness) * (LeftBound111219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111220

namespace LeftBound111231
def owner : Owner := ⟨.program ⟨257⟩, ⟨54758⟩⟩
def transferEvent : Nat := 111231
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 111229 .coefficient) (.value (.predecessor 1 111230 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111229 .coefficient)
      LeftAuthority111227.bound (LeftAuthority111227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111230 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority111227.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111227.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111227.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound111231

namespace LeftBound111235
def owner : Owner := ⟨.program ⟨257⟩, ⟨54759⟩⟩
def transferEvent : Nat := 111235
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 111233 .coefficient) (.predecessor 1 111234 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111233 .coefficient)
      LeftBound105242.bound (LeftBound105242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111234 .coefficient)
      LeftBound111231.bound (LeftBound111231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105242.bound LeftBound111231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105242.bound, LeftBound111231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105242.actual selector witness) * (LeftBound111231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111235

namespace LeftBound111236
def owner : Owner := ⟨.program ⟨257⟩, ⟨54759⟩⟩
def transferEvent : Nat := 111236
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨54756⟩⟩]⟩ [⟨.result 111228 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111228 .coefficient)
      LeftAuthority111227.bound (LeftAuthority111227.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54756⟩⟩) (rawTerms := some (Proof.Events434.exact111228RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111227.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority111227.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111227.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound111236

namespace LeftBound111237
def owner : Owner := ⟨.program ⟨257⟩, ⟨54759⟩⟩
def transferEvent : Nat := 111237
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 111236) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 111236)
      LeftBound111236.bound (LeftBound111236.actual selector witness) := by
  exact .transfer (LeftBound111236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound111236.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound111236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound111236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111237

namespace LeftBound111332
def owner : Owner := ⟨.program ⟨257⟩, ⟨53877⟩⟩
def transferEvent : Nat := 111332
def frameStart : Nat := 111293
def rule : BoundRule := .identity (.predecessor 0 111331 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111331 .coefficient)
      LeftAuthority111329.bound (LeftAuthority111329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111329.derived selector witness)

def rawBound : CoeffClass := LeftAuthority111329.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority111329.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound111332

namespace LeftBound111349
def owner : Owner := ⟨.program ⟨257⟩, ⟨55350⟩⟩
def transferEvent : Nat := 111349
def frameStart : Nat := 111293
def rule : BoundRule := .sum [.predecessor 0 111347 .coefficient, .predecessor 1 111348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111347 .coefficient)
      LeftBound111332.bound (LeftBound111332.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound111332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111348 .coefficient)
      LeftAuthority111345.bound (LeftAuthority111345.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority111345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111332.bound, LeftAuthority111345.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111332.bound, LeftAuthority111345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111332.actual selector witness, LeftAuthority111345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111349

namespace LeftBound111352
def owner : Owner := ⟨.program ⟨257⟩, ⟨55351⟩⟩
def transferEvent : Nat := 111352
def frameStart : Nat := 111293
def rule : BoundRule := .identity (.predecessor 0 111351 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111351 .coefficient)
      LeftBound111349.bound (LeftBound111349.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound111349.derived selector witness)

def rawBound : CoeffClass := LeftBound111349.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound111349.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound111352

namespace LeftBound111358
def owner : Owner := ⟨.program ⟨257⟩, ⟨55352⟩⟩
def transferEvent : Nat := 111358
def frameStart : Nat := 111293
def rule : BoundRule := .product (.predecessor 0 111356 .coefficient) (.predecessor 1 111357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111356 .coefficient)
      LeftAuthority111354.bound (LeftAuthority111354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111357 .coefficient)
      LeftBound111352.bound (LeftBound111352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events434.exact111353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111352.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority111354.bound LeftBound111352.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111354.bound, LeftBound111352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority111354.actual selector witness) * (LeftBound111352.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111358

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
