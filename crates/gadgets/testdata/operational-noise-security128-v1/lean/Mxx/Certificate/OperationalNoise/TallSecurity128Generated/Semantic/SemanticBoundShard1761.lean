import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1728
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1731
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1739
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1742
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1746
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1749
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1750
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1753
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1757
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1760

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound260074
def owner : Owner := ⟨.program ⟨257⟩, ⟨20501⟩⟩
def transferEvent : Nat := 260074
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260070 .summary, .result 259588 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260070 .summary)
      LeftBound260069.bound (LeftBound260069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17624⟩⟩) (rawTerms := some (Proof.Events1015.exact260070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259588 .summary)
      LeftBound259587.bound (LeftBound259587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20500⟩⟩) (rawTerms := some (Proof.Events1014.exact259588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260069.bound, LeftBound259587.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260069.bound, LeftBound259587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260069.actual selector witness, LeftBound259587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260074

namespace LeftBound260078
def owner : Owner := ⟨.program ⟨257⟩, ⟨23721⟩⟩
def transferEvent : Nat := 260078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260076 .coefficient, .predecessor 1 260077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260076 .coefficient)
      LeftBound260073.bound (LeftBound260073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact260075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260077 .coefficient)
      LeftBound259102.bound (LeftBound259102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260073.bound, LeftBound259102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260073.bound, LeftBound259102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260073.actual selector witness, LeftBound259102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260078

namespace LeftBound260079
def owner : Owner := ⟨.program ⟨257⟩, ⟨23721⟩⟩
def transferEvent : Nat := 260079
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260075 .summary, .result 259106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260075 .summary)
      LeftBound260074.bound (LeftBound260074.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20501⟩⟩) (rawTerms := some (Proof.Events1015.exact260075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259106 .summary)
      LeftBound259105.bound (LeftBound259105.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23720⟩⟩) (rawTerms := some (Proof.Events1012.exact259106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260074.bound, LeftBound259105.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260074.bound, LeftBound259105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260074.actual selector witness, LeftBound259105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260079

namespace LeftBound260083
def owner : Owner := ⟨.program ⟨257⟩, ⟨33741⟩⟩
def transferEvent : Nat := 260083
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260081 .coefficient, .predecessor 1 260082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260081 .coefficient)
      LeftBound260078.bound (LeftBound260078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact260080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260082 .coefficient)
      LeftBound258620.bound (LeftBound258620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260078.bound, LeftBound258620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260078.bound, LeftBound258620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260078.actual selector witness, LeftBound258620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260083

namespace LeftBound260084
def owner : Owner := ⟨.program ⟨257⟩, ⟨33741⟩⟩
def transferEvent : Nat := 260084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260080 .summary, .result 258624 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260080 .summary)
      LeftBound260079.bound (LeftBound260079.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23721⟩⟩) (rawTerms := some (Proof.Events1015.exact260080RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258624 .summary)
      LeftBound258623.bound (LeftBound258623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33740⟩⟩) (rawTerms := some (Proof.Events1010.exact258624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260079.bound, LeftBound258623.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260079.bound, LeftBound258623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260079.actual selector witness, LeftBound258623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260084

namespace LeftBound260088
def owner : Owner := ⟨.program ⟨257⟩, ⟨52801⟩⟩
def transferEvent : Nat := 260088
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260086 .coefficient, .predecessor 1 260087 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260086 .coefficient)
      LeftBound260083.bound (LeftBound260083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact260085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260087 .coefficient)
      LeftBound258138.bound (LeftBound258138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1008.exact258142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260083.bound, LeftBound258138.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260083.bound, LeftBound258138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260083.actual selector witness, LeftBound258138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260088

namespace LeftBound260089
def owner : Owner := ⟨.program ⟨257⟩, ⟨52801⟩⟩
def transferEvent : Nat := 260089
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260085 .summary, .result 258142 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260085 .summary)
      LeftBound260084.bound (LeftBound260084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33741⟩⟩) (rawTerms := some (Proof.Events1015.exact260085RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258142 .summary)
      LeftBound258141.bound (LeftBound258141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52800⟩⟩) (rawTerms := some (Proof.Events1008.exact258142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260084.bound, LeftBound258141.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260084.bound, LeftBound258141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260084.actual selector witness, LeftBound258141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260089

namespace LeftBound260093
def owner : Owner := ⟨.program ⟨257⟩, ⟨55781⟩⟩
def transferEvent : Nat := 260093
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260091 .coefficient, .predecessor 1 260092 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260091 .coefficient)
      LeftBound260088.bound (LeftBound260088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact260090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260092 .coefficient)
      LeftBound257656.bound (LeftBound257656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1006.exact257660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260088.bound, LeftBound257656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260088.bound, LeftBound257656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260088.actual selector witness, LeftBound257656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260093

namespace LeftBound260094
def owner : Owner := ⟨.program ⟨257⟩, ⟨55781⟩⟩
def transferEvent : Nat := 260094
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260090 .summary, .result 257660 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260090 .summary)
      LeftBound260089.bound (LeftBound260089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52801⟩⟩) (rawTerms := some (Proof.Events1015.exact260090RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257660 .summary)
      LeftBound257659.bound (LeftBound257659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55780⟩⟩) (rawTerms := some (Proof.Events1006.exact257660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound257659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260089.bound, LeftBound257659.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260089.bound, LeftBound257659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260089.actual selector witness, LeftBound257659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260094

namespace LeftBound260098
def owner : Owner := ⟨.program ⟨257⟩, ⟨58761⟩⟩
def transferEvent : Nat := 260098
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260096 .coefficient, .predecessor 1 260097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260096 .coefficient)
      LeftBound260093.bound (LeftBound260093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact260095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260097 .coefficient)
      LeftBound257174.bound (LeftBound257174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1004.exact257178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound257174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound257174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260093.bound, LeftBound257174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260093.bound, LeftBound257174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260093.actual selector witness, LeftBound257174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260098

namespace LeftBound260099
def owner : Owner := ⟨.program ⟨257⟩, ⟨58761⟩⟩
def transferEvent : Nat := 260099
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260095 .summary, .result 257178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260095 .summary)
      LeftBound260094.bound (LeftBound260094.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55781⟩⟩) (rawTerms := some (Proof.Events1015.exact260095RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 257178 .summary)
      LeftBound257177.bound (LeftBound257177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58760⟩⟩) (rawTerms := some (Proof.Events1004.exact257178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound257177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260094.bound, LeftBound257177.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260094.bound, LeftBound257177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260094.actual selector witness, LeftBound257177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260099

namespace LeftBound260103
def owner : Owner := ⟨.program ⟨257⟩, ⟨61741⟩⟩
def transferEvent : Nat := 260103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260101 .coefficient, .predecessor 1 260102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260101 .coefficient)
      LeftBound260098.bound (LeftBound260098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260098.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260102 .coefficient)
      LeftBound256692.bound (LeftBound256692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1002.exact256696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound256692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound256692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260098.bound, LeftBound256692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260098.bound, LeftBound256692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260098.actual selector witness, LeftBound256692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260103

namespace LeftBound260104
def owner : Owner := ⟨.program ⟨257⟩, ⟨61741⟩⟩
def transferEvent : Nat := 260104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260100 .summary, .result 256696 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260100 .summary)
      LeftBound260099.bound (LeftBound260099.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58761⟩⟩) (rawTerms := some (Proof.Events1016.exact260100RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 256696 .summary)
      LeftBound256695.bound (LeftBound256695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61740⟩⟩) (rawTerms := some (Proof.Events1002.exact256696RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound256695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260099.bound, LeftBound256695.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260099.bound, LeftBound256695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260099.actual selector witness, LeftBound256695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260104

namespace LeftBound260108
def owner : Owner := ⟨.program ⟨257⟩, ⟨64721⟩⟩
def transferEvent : Nat := 260108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260106 .coefficient, .predecessor 1 260107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260106 .coefficient)
      LeftBound260103.bound (LeftBound260103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260107 .coefficient)
      LeftBound256210.bound (LeftBound256210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1000.exact256214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound256210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound256210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260103.bound, LeftBound256210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260103.bound, LeftBound256210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260103.actual selector witness, LeftBound256210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260108

namespace LeftBound260109
def owner : Owner := ⟨.program ⟨257⟩, ⟨64721⟩⟩
def transferEvent : Nat := 260109
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260105 .summary, .result 256214 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260105 .summary)
      LeftBound260104.bound (LeftBound260104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61741⟩⟩) (rawTerms := some (Proof.Events1016.exact260105RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 256214 .summary)
      LeftBound256213.bound (LeftBound256213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64720⟩⟩) (rawTerms := some (Proof.Events1000.exact256214RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound256213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260104.bound, LeftBound256213.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260104.bound, LeftBound256213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260104.actual selector witness, LeftBound256213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260109

namespace LeftBound260113
def owner : Owner := ⟨.program ⟨257⟩, ⟨69786⟩⟩
def transferEvent : Nat := 260113
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260111 .coefficient, .predecessor 1 260112 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260111 .coefficient)
      LeftBound260108.bound (LeftBound260108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260112 .coefficient)
      LeftBound255728.bound (LeftBound255728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events998.exact255732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260108.bound, LeftBound255728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260108.bound, LeftBound255728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260108.actual selector witness, LeftBound255728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260113

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
