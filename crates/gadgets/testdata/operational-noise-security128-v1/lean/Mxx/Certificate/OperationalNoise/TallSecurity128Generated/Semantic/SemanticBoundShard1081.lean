import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1065
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1066
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1069
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1073
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1080

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound163191
def owner : Owner := ⟨.program ⟨257⟩, ⟨52857⟩⟩
def transferEvent : Nat := 163191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163189 .coefficient, .predecessor 1 163190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163189 .coefficient)
      LeftBound163186.bound (LeftBound163186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163190 .coefficient)
      LeftBound162286.bound (LeftBound162286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163186.bound, LeftBound162286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163186.bound, LeftBound162286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163186.actual selector witness, LeftBound162286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163191

namespace LeftBound163192
def owner : Owner := ⟨.program ⟨257⟩, ⟨52857⟩⟩
def transferEvent : Nat := 163192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163188 .summary, .result 162293 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163188 .summary)
      LeftBound163187.bound (LeftBound163187.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33797⟩⟩) (rawTerms := some (Proof.Events637.exact163188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162293 .summary)
      LeftBound162288.bound (LeftBound162288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52856⟩⟩) (rawTerms := some (Proof.Events633.exact162293RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163187.bound, LeftBound162288.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163187.bound, LeftBound162288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163187.actual selector witness, LeftBound162288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163192

namespace LeftBound163196
def owner : Owner := ⟨.program ⟨257⟩, ⟨55837⟩⟩
def transferEvent : Nat := 163196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163194 .coefficient, .predecessor 1 163195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163194 .coefficient)
      LeftBound163191.bound (LeftBound163191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163195 .coefficient)
      LeftBound162074.bound (LeftBound162074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events633.exact162081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound162074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound162074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163191.bound, LeftBound162074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163191.bound, LeftBound162074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163191.actual selector witness, LeftBound162074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163196

namespace LeftBound163197
def owner : Owner := ⟨.program ⟨257⟩, ⟨55837⟩⟩
def transferEvent : Nat := 163197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163193 .summary, .result 162081 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163193 .summary)
      LeftBound163192.bound (LeftBound163192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52857⟩⟩) (rawTerms := some (Proof.Events637.exact163193RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 162081 .summary)
      LeftBound162076.bound (LeftBound162076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55836⟩⟩) (rawTerms := some (Proof.Events633.exact162081RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound162076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163192.bound, LeftBound162076.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163192.bound, LeftBound162076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163192.actual selector witness, LeftBound162076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163197

namespace LeftBound163201
def owner : Owner := ⟨.program ⟨257⟩, ⟨58817⟩⟩
def transferEvent : Nat := 163201
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163199 .coefficient, .predecessor 1 163200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163199 .coefficient)
      LeftBound163196.bound (LeftBound163196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163200 .coefficient)
      LeftBound161862.bound (LeftBound161862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events632.exact161869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163196.bound, LeftBound161862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163196.bound, LeftBound161862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163196.actual selector witness, LeftBound161862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163201

namespace LeftBound163202
def owner : Owner := ⟨.program ⟨257⟩, ⟨58817⟩⟩
def transferEvent : Nat := 163202
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163198 .summary, .result 161869 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163198 .summary)
      LeftBound163197.bound (LeftBound163197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55837⟩⟩) (rawTerms := some (Proof.Events637.exact163198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161869 .summary)
      LeftBound161864.bound (LeftBound161864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58816⟩⟩) (rawTerms := some (Proof.Events632.exact161869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163197.bound, LeftBound161864.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163197.bound, LeftBound161864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163197.actual selector witness, LeftBound161864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163202

namespace LeftBound163206
def owner : Owner := ⟨.program ⟨257⟩, ⟨61797⟩⟩
def transferEvent : Nat := 163206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163204 .coefficient, .predecessor 1 163205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163204 .coefficient)
      LeftBound163201.bound (LeftBound163201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163205 .coefficient)
      LeftBound161650.bound (LeftBound161650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events631.exact161657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163201.bound, LeftBound161650.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163201.bound, LeftBound161650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163201.actual selector witness, LeftBound161650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163206

namespace LeftBound163207
def owner : Owner := ⟨.program ⟨257⟩, ⟨61797⟩⟩
def transferEvent : Nat := 163207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163203 .summary, .result 161657 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163203 .summary)
      LeftBound163202.bound (LeftBound163202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58817⟩⟩) (rawTerms := some (Proof.Events637.exact163203RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161657 .summary)
      LeftBound161652.bound (LeftBound161652.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61796⟩⟩) (rawTerms := some (Proof.Events631.exact161657RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163202.bound, LeftBound161652.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163202.bound, LeftBound161652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163202.actual selector witness, LeftBound161652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163207

namespace LeftBound163211
def owner : Owner := ⟨.program ⟨257⟩, ⟨64777⟩⟩
def transferEvent : Nat := 163211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163209 .coefficient, .predecessor 1 163210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163209 .coefficient)
      LeftBound163206.bound (LeftBound163206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163210 .coefficient)
      LeftBound161438.bound (LeftBound161438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events630.exact161445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163206.bound, LeftBound161438.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163206.bound, LeftBound161438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163206.actual selector witness, LeftBound161438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163211

namespace LeftBound163212
def owner : Owner := ⟨.program ⟨257⟩, ⟨64777⟩⟩
def transferEvent : Nat := 163212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163208 .summary, .result 161445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163208 .summary)
      LeftBound163207.bound (LeftBound163207.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61797⟩⟩) (rawTerms := some (Proof.Events637.exact163208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161445 .summary)
      LeftBound161440.bound (LeftBound161440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64776⟩⟩) (rawTerms := some (Proof.Events630.exact161445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163207.bound, LeftBound161440.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163207.bound, LeftBound161440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163207.actual selector witness, LeftBound161440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163212

namespace LeftBound163216
def owner : Owner := ⟨.program ⟨257⟩, ⟨69930⟩⟩
def transferEvent : Nat := 163216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163214 .coefficient, .predecessor 1 163215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163214 .coefficient)
      LeftBound163211.bound (LeftBound163211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163215 .coefficient)
      LeftBound161226.bound (LeftBound161226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events629.exact161233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163211.bound, LeftBound161226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163211.bound, LeftBound161226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163211.actual selector witness, LeftBound161226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163216

namespace LeftBound163217
def owner : Owner := ⟨.program ⟨257⟩, ⟨69930⟩⟩
def transferEvent : Nat := 163217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163213 .summary, .result 161233 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163213 .summary)
      LeftBound163212.bound (LeftBound163212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64777⟩⟩) (rawTerms := some (Proof.Events637.exact163213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161233 .summary)
      LeftBound161228.bound (LeftBound161228.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69929⟩⟩) (rawTerms := some (Proof.Events629.exact161233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163212.bound, LeftBound161228.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163212.bound, LeftBound161228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163212.actual selector witness, LeftBound161228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163217

namespace LeftBound163221
def owner : Owner := ⟨.program ⟨257⟩, ⟨69931⟩⟩
def transferEvent : Nat := 163221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163219 .coefficient, .predecessor 1 163220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163219 .coefficient)
      LeftBound163216.bound (LeftBound163216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163220 .coefficient)
      LeftBound161014.bound (LeftBound161014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events628.exact161021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound161014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound161014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163216.bound, LeftBound161014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163216.bound, LeftBound161014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163216.actual selector witness, LeftBound161014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163221

namespace LeftBound163222
def owner : Owner := ⟨.program ⟨257⟩, ⟨69931⟩⟩
def transferEvent : Nat := 163222
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163218 .summary, .result 161021 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163218 .summary)
      LeftBound163217.bound (LeftBound163217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69930⟩⟩) (rawTerms := some (Proof.Events637.exact163218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 161021 .summary)
      LeftBound161016.bound (LeftBound161016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28212⟩⟩) (rawTerms := some (Proof.Events628.exact161021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound161016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163217.bound, LeftBound161016.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163217.bound, LeftBound161016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163217.actual selector witness, LeftBound161016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163222

namespace LeftBound163226
def owner : Owner := ⟨.program ⟨257⟩, ⟨69932⟩⟩
def transferEvent : Nat := 163226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 163224 .coefficient, .predecessor 1 163225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 163224 .coefficient)
      LeftBound163221.bound (LeftBound163221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events637.exact163223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 163225 .coefficient)
      LeftBound160802.bound (LeftBound160802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events628.exact160809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163221.bound, LeftBound160802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163221.bound, LeftBound160802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163221.actual selector witness, LeftBound160802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163226

namespace LeftBound163227
def owner : Owner := ⟨.program ⟨257⟩, ⟨69932⟩⟩
def transferEvent : Nat := 163227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 163223 .summary, .result 160809 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163223 .summary)
      LeftBound163222.bound (LeftBound163222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69931⟩⟩) (rawTerms := some (Proof.Events637.exact163223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160809 .summary)
      LeftBound160804.bound (LeftBound160804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30892⟩⟩) (rawTerms := some (Proof.Events628.exact160809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound160804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound163222.bound, LeftBound160804.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163222.bound, LeftBound160804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound163222.actual selector witness, LeftBound160804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound163227

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
