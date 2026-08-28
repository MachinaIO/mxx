import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard059
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1184

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound177962
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def transferEvent : Nat := 177962
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 177957 .summary) (.transfer 177961) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177957 .summary)
      LeftBound177956.bound (LeftBound177956.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71372⟩⟩) (rawTerms := some (Proof.Events695.exact177957RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 177961)
      LeftBound177961.bound (LeftBound177961.actual selector witness) := by
  exact .transfer (LeftBound177961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound177956.bound LeftBound177961.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167293655866246524336865280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177956.bound, LeftBound177961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound177956.actual selector witness) * (LeftBound177961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound177962

namespace LeftBound178024
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def transferEvent : Nat := 178024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 178022 .coefficient, .predecessor 1 178023 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178022 .coefficient)
      LeftBound177960.bound (LeftBound177960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178023 .coefficient)
      LeftBound163541.bound (LeftBound163541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177960.bound, LeftBound163541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177960.bound, LeftBound163541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177960.actual selector witness, LeftBound163541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound178024

namespace LeftBound178044
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def transferEvent : Nat := 178044
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 178021 .summary, .result 163618 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178021 .summary)
      LeftBound177962.bound (LeftBound177962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71373⟩⟩) (rawTerms := some (Proof.Events695.exact178021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163618 .summary)
      LeftBound163579.bound (LeftBound163579.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67548⟩⟩) (rawTerms := some (Proof.Events639.exact163618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177962.bound, LeftBound163579.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177962.bound, LeftBound163579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177962.actual selector witness, LeftBound163579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound178044

namespace LeftBound178048
def owner : Owner := ⟨.program ⟨257⟩, ⟨71375⟩⟩
def transferEvent : Nat := 178048
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 178046 .coefficient) (.predecessor 1 178047 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178046 .coefficient)
      LeftBound178024.bound (LeftBound178024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178047 .coefficient)
      LeftBound16373.bound (LeftBound16373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound178024.bound LeftBound16373.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178024.bound, LeftBound16373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound178024.actual selector witness) * (LeftBound16373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound178048

namespace LeftBound178049
def owner : Owner := ⟨.program ⟨257⟩, ⟨71375⟩⟩
def transferEvent : Nat := 178049
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7127⟩⟩]⟩ [⟨.result 16370 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16370 .coefficient)
      LeftAuthority16369.bound (LeftAuthority16369.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7127⟩⟩) (rawTerms := some (Proof.Events063.exact16370RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16369.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16369.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16369.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound178049

namespace LeftBound178050
def owner : Owner := ⟨.program ⟨257⟩, ⟨71375⟩⟩
def transferEvent : Nat := 178050
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178045 .summary) (.transfer 178049) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178045 .summary)
      LeftBound178044.bound (LeftBound178044.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71374⟩⟩) (rawTerms := some (Proof.Events695.exact178045RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 178049)
      LeftBound178049.bound (LeftBound178049.actual selector witness) := by
  exact .transfer (LeftBound178049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound178044.bound LeftBound178049.bound
def bound : CoeffClass := .finite ⟨7702113697398803698856913678033037845150209519672183236728648848208035840, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178044.bound, LeftBound178049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound178044.actual selector witness) * (LeftBound178049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound178050

namespace LeftBound178131
def owner : Owner := ⟨.program ⟨257⟩, ⟨6199⟩⟩
def transferEvent : Nat := 178131
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 178126 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178126 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound178131

namespace LeftBound178135
def owner : Owner := ⟨.program ⟨257⟩, ⟨7007⟩⟩
def transferEvent : Nat := 178135
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 178133 .coefficient) (.predecessor 1 178134 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178133 .coefficient)
      LeftBound178131.bound (LeftBound178131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178134 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178131.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178131.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178131.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound178135

namespace LeftBound178147
def owner : Owner := ⟨.program ⟨257⟩, ⟨6184⟩⟩
def transferEvent : Nat := 178147
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 178142 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178142 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound178147

namespace LeftBound178151
def owner : Owner := ⟨.program ⟨257⟩, ⟨8919⟩⟩
def transferEvent : Nat := 178151
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 178149 .coefficient) (.predecessor 1 178150 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178149 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178150 .coefficient)
      LeftAuthority16416.bound (LeftAuthority16416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16416.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftAuthority16416.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftAuthority16416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftAuthority16416.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound178151

namespace LeftBound178156
def owner : Owner := ⟨.program ⟨257⟩, ⟨9423⟩⟩
def transferEvent : Nat := 178156
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 178154 .coefficient, .predecessor 1 178155 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178154 .coefficient)
      LeftBound178151.bound (LeftBound178151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178155 .coefficient)
      LeftBound178135.bound (LeftBound178135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178135.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound178151.bound, LeftBound178135.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178151.bound, LeftBound178135.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound178151.actual selector witness, LeftBound178135.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound178156

namespace LeftBound178160
def owner : Owner := ⟨.program ⟨257⟩, ⟨9424⟩⟩
def transferEvent : Nat := 178160
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 178158 .coefficient, .predecessor 1 178159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178158 .coefficient)
      LeftBound178156.bound (LeftBound178156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178159 .coefficient)
      LeftAuthority178110.bound (LeftAuthority178110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority178110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority178110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound178156.bound, LeftAuthority178110.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178156.bound, LeftAuthority178110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound178156.actual selector witness, LeftAuthority178110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound178160

namespace LeftBound178161
def owner : Owner := ⟨.program ⟨257⟩, ⟨9424⟩⟩
def transferEvent : Nat := 178161
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9⟩⟩]⟩ [⟨.result 178111 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178111 .coefficient)
      LeftAuthority178110.bound (LeftAuthority178110.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9⟩⟩) (rawTerms := some (Proof.Events695.exact178111RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority178110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority178110.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority178110.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority178110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority178110.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound178161

namespace LeftBound178166
def owner : Owner := ⟨.program ⟨257⟩, ⟨67520⟩⟩
def transferEvent : Nat := 178166
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 178164 .coefficient) (.predecessor 1 178165 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 178164 .coefficient)
      LeftBound178160.bound (LeftBound178160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 178165 .coefficient)
      LeftBound9031.bound (LeftBound9031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9031.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178160.bound LeftBound9031.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178160.bound, LeftBound9031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178160.actual selector witness) * (LeftBound9031.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound178166

namespace LeftBound178167
def owner : Owner := ⟨.program ⟨257⟩, ⟨67520⟩⟩
def transferEvent : Nat := 178167
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67514⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 8807 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6774⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 8807 .coefficient)
      LeftAuthority8806.bound (LeftAuthority8806.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67514⟩⟩) (rawTerms := some (Proof.Events034.exact8807RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8806.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority8806.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority8806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority8806.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound178167

namespace LeftBound178168
def owner : Owner := ⟨.program ⟨257⟩, ⟨67520⟩⟩
def transferEvent : Nat := 178168
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48398⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 8815 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6800⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 8815 .coefficient)
      LeftAuthority8814.bound (LeftAuthority8814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48398⟩⟩) (rawTerms := some (Proof.Events034.exact8815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority8814.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority8814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority8814.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound178168

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
