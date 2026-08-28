import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1417
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1418

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound211159
def owner : Owner := ⟨.program ⟨257⟩, ⟨26411⟩⟩
def transferEvent : Nat := 211159
def frameStart : Nat := 211049
def rule : BoundRule := .sum [.predecessor 0 211157 .coefficient, .predecessor 1 211158 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211157 .coefficient)
      LeftAuthority211155.bound (LeftAuthority211155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211158 .coefficient)
      LeftBound211151.bound (LeftBound211151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority211155.bound, LeftBound211151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211155.bound, LeftBound211151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority211155.actual selector witness, LeftBound211151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211159

namespace LeftBound211163
def owner : Owner := ⟨.program ⟨257⟩, ⟨27923⟩⟩
def transferEvent : Nat := 211163
def frameStart : Nat := 211049
def rule : BoundRule := .sum [.predecessor 0 211161 .coefficient, .predecessor 1 211162 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211161 .coefficient)
      LeftBound211159.bound (LeftBound211159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211162 .coefficient)
      LeftBound211140.bound (LeftBound211140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound211159.bound, LeftBound211140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211159.bound, LeftBound211140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound211159.actual selector witness, LeftBound211140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211163

namespace LeftBound211176
def owner : Owner := ⟨.program ⟨257⟩, ⟨27921⟩⟩
def transferEvent : Nat := 211176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 211174 .coefficient, .predecessor 1 211175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211174 .coefficient)
      LeftBound210997.bound (LeftBound210997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound210997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound210997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211175 .coefficient)
      LeftBound210980.bound (LeftBound210980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact210987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound210980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound210980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound210997.bound, LeftBound210980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound210997.bound, LeftBound210980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound210997.actual selector witness, LeftBound210980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211176

namespace LeftBound211179
def owner : Owner := ⟨.program ⟨257⟩, ⟨27921⟩⟩
def transferEvent : Nat := 211179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 211173 .summary, .result 210987 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211173 .summary)
      LeftBound210999.bound (LeftBound210999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26852⟩⟩) (rawTerms := some (Proof.Events824.exact211173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound210999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 210987 .summary)
      LeftBound210982.bound (LeftBound210982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27920⟩⟩) (rawTerms := some (Proof.Events824.exact210987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound210982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound210999.bound, LeftBound210982.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound210999.bound, LeftBound210982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound210999.actual selector witness, LeftBound210982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211179

namespace LeftBound211183
def owner : Owner := ⟨.program ⟨257⟩, ⟨28291⟩⟩
def transferEvent : Nat := 211183
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 211181 .coefficient) (.predecessor 1 211182 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211181 .coefficient)
      LeftBound211176.bound (LeftBound211176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211182 .coefficient)
      LeftAuthority210902.bound (LeftAuthority210902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events823.exact210903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority210902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority210902.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound211176.bound LeftAuthority210902.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211176.bound, LeftAuthority210902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound211176.actual selector witness) * (LeftAuthority210902.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound211183

namespace LeftBound211184
def owner : Owner := ⟨.program ⟨257⟩, ⟨28291⟩⟩
def transferEvent : Nat := 211184
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28289⟩⟩]⟩ [⟨.result 210903 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 210903 .coefficient)
      LeftAuthority210902.bound (LeftAuthority210902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28289⟩⟩) (rawTerms := some (Proof.Events823.exact210903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority210902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority210902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority210902.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority210902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority210902.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound211184

namespace LeftBound211185
def owner : Owner := ⟨.program ⟨257⟩, ⟨28291⟩⟩
def transferEvent : Nat := 211185
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 211180 .summary) (.transfer 211184) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211180 .summary)
      LeftBound211179.bound (LeftBound211179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27921⟩⟩) (rawTerms := some (Proof.Events824.exact211180RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 211184)
      LeftBound211184.bound (LeftBound211184.actual selector witness) := by
  exact .transfer (LeftBound211184.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound211179.bound LeftBound211184.bound
def bound : CoeffClass := .finite ⟨32191557518723128098041228165120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211179.bound, LeftBound211184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound211179.actual selector witness) * (LeftBound211184.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound211185

namespace LeftBound211196
def owner : Owner := ⟨.program ⟨257⟩, ⟨27158⟩⟩
def transferEvent : Nat := 211196
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 211194 .coefficient) (.value (.predecessor 1 211195 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211194 .coefficient)
      LeftAuthority211192.bound (LeftAuthority211192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211195 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority211192.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211192.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority211192.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound211196

namespace LeftBound211200
def owner : Owner := ⟨.program ⟨257⟩, ⟨27159⟩⟩
def transferEvent : Nat := 211200
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 211198 .coefficient) (.predecessor 1 211199 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211198 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211199 .coefficient)
      LeftBound211196.bound (LeftBound211196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events824.exact211197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound211196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound211196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound211196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound211200

namespace LeftBound211201
def owner : Owner := ⟨.program ⟨257⟩, ⟨27159⟩⟩
def transferEvent : Nat := 211201
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27156⟩⟩]⟩ [⟨.result 211193 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211193 .coefficient)
      LeftAuthority211192.bound (LeftAuthority211192.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27156⟩⟩) (rawTerms := some (Proof.Events824.exact211193RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211192.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority211192.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority211192.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound211201

namespace LeftBound211202
def owner : Owner := ⟨.program ⟨257⟩, ⟨27159⟩⟩
def transferEvent : Nat := 211202
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 211201) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 211201)
      LeftBound211201.bound (LeftBound211201.actual selector witness) := by
  exact .transfer (LeftBound211201.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound211201.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound211201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound211201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound211202

namespace LeftBound211297
def owner : Owner := ⟨.program ⟨257⟩, ⟨26409⟩⟩
def transferEvent : Nat := 211297
def frameStart : Nat := 211258
def rule : BoundRule := .identity (.predecessor 0 211296 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211296 .coefficient)
      LeftAuthority211294.bound (LeftAuthority211294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211294.derived selector witness)

def rawBound : CoeffClass := LeftAuthority211294.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority211294.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound211297

namespace LeftBound211314
def owner : Owner := ⟨.program ⟨257⟩, ⟨27766⟩⟩
def transferEvent : Nat := 211314
def frameStart : Nat := 211258
def rule : BoundRule := .sum [.predecessor 0 211312 .coefficient, .predecessor 1 211313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211312 .coefficient)
      LeftBound211297.bound (LeftBound211297.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound211297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211313 .coefficient)
      LeftAuthority211310.bound (LeftAuthority211310.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority211310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound211297.bound, LeftAuthority211310.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211297.bound, LeftAuthority211310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound211297.actual selector witness, LeftAuthority211310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211314

namespace LeftBound211317
def owner : Owner := ⟨.program ⟨257⟩, ⟨27767⟩⟩
def transferEvent : Nat := 211317
def frameStart : Nat := 211258
def rule : BoundRule := .identity (.predecessor 0 211316 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211316 .coefficient)
      LeftBound211314.bound (LeftBound211314.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound211314.derived selector witness)

def rawBound : CoeffClass := LeftBound211314.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound211314.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound211317

namespace LeftBound211323
def owner : Owner := ⟨.program ⟨257⟩, ⟨27768⟩⟩
def transferEvent : Nat := 211323
def frameStart : Nat := 211258
def rule : BoundRule := .product (.predecessor 0 211321 .coefficient) (.predecessor 1 211322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211321 .coefficient)
      LeftAuthority211319.bound (LeftAuthority211319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211322 .coefficient)
      LeftBound211317.bound (LeftBound211317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211317.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority211319.bound LeftBound211317.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211319.bound, LeftBound211317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority211319.actual selector witness) * (LeftBound211317.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound211323

namespace LeftBound211331
def owner : Owner := ⟨.program ⟨257⟩, ⟨27769⟩⟩
def transferEvent : Nat := 211331
def frameStart : Nat := 211258
def rule : BoundRule := .sum [.predecessor 0 211329 .coefficient, .predecessor 1 211330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 211329 .coefficient)
      LeftAuthority211327.bound (LeftAuthority211327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 211330 .coefficient)
      LeftBound211323.bound (LeftBound211323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events825.exact211325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211323.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority211327.bound, LeftBound211323.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211327.bound, LeftBound211323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority211327.actual selector witness, LeftBound211323.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound211331

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
