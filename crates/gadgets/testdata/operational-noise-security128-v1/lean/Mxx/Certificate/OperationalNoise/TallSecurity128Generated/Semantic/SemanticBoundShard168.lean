import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard152
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard153
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard155
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard156
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard157
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard159
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard160
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard161
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard163
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31562
def owner : Owner := ⟨.program ⟨257⟩, ⟨33620⟩⟩
def transferEvent : Nat := 31562
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31558 .summary, .result 30877 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31558 .summary)
      LeftBound31557.bound (LeftBound31557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23600⟩⟩) (rawTerms := some (Proof.Events123.exact31558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 30877 .summary)
      LeftBound30872.bound (LeftBound30872.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33619⟩⟩) (rawTerms := some (Proof.Events120.exact30877RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31557.bound, LeftBound30872.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31557.bound, LeftBound30872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31557.actual selector witness, LeftBound30872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31562

namespace LeftBound31566
def owner : Owner := ⟨.program ⟨257⟩, ⟨52680⟩⟩
def transferEvent : Nat := 31566
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31564 .coefficient, .predecessor 1 31565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31564 .coefficient)
      LeftBound31561.bound (LeftBound31561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31565 .coefficient)
      LeftBound30658.bound (LeftBound30658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events119.exact30665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31561.bound, LeftBound30658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31561.bound, LeftBound30658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31561.actual selector witness, LeftBound30658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31566

namespace LeftBound31567
def owner : Owner := ⟨.program ⟨257⟩, ⟨52680⟩⟩
def transferEvent : Nat := 31567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31563 .summary, .result 30665 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31563 .summary)
      LeftBound31562.bound (LeftBound31562.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33620⟩⟩) (rawTerms := some (Proof.Events123.exact31563RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 30665 .summary)
      LeftBound30660.bound (LeftBound30660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52679⟩⟩) (rawTerms := some (Proof.Events119.exact30665RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31562.bound, LeftBound30660.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31562.bound, LeftBound30660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31562.actual selector witness, LeftBound30660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31567

namespace LeftBound31571
def owner : Owner := ⟨.program ⟨257⟩, ⟨55660⟩⟩
def transferEvent : Nat := 31571
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31569 .coefficient, .predecessor 1 31570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31569 .coefficient)
      LeftBound31566.bound (LeftBound31566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31570 .coefficient)
      LeftBound30446.bound (LeftBound30446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events118.exact30453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31566.bound, LeftBound30446.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31566.bound, LeftBound30446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31566.actual selector witness, LeftBound30446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31571

namespace LeftBound31572
def owner : Owner := ⟨.program ⟨257⟩, ⟨55660⟩⟩
def transferEvent : Nat := 31572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31568 .summary, .result 30453 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31568 .summary)
      LeftBound31567.bound (LeftBound31567.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52680⟩⟩) (rawTerms := some (Proof.Events123.exact31568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 30453 .summary)
      LeftBound30448.bound (LeftBound30448.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55659⟩⟩) (rawTerms := some (Proof.Events118.exact30453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30448.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31567.bound, LeftBound30448.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31567.bound, LeftBound30448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31567.actual selector witness, LeftBound30448.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31572

namespace LeftBound31576
def owner : Owner := ⟨.program ⟨257⟩, ⟨58640⟩⟩
def transferEvent : Nat := 31576
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31574 .coefficient, .predecessor 1 31575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31574 .coefficient)
      LeftBound31571.bound (LeftBound31571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31575 .coefficient)
      LeftBound30234.bound (LeftBound30234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events118.exact30241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31571.bound, LeftBound30234.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31571.bound, LeftBound30234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31571.actual selector witness, LeftBound30234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31576

namespace LeftBound31577
def owner : Owner := ⟨.program ⟨257⟩, ⟨58640⟩⟩
def transferEvent : Nat := 31577
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31573 .summary, .result 30241 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31573 .summary)
      LeftBound31572.bound (LeftBound31572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55660⟩⟩) (rawTerms := some (Proof.Events123.exact31573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 30241 .summary)
      LeftBound30236.bound (LeftBound30236.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58639⟩⟩) (rawTerms := some (Proof.Events118.exact30241RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30236.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31572.bound, LeftBound30236.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31572.bound, LeftBound30236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31572.actual selector witness, LeftBound30236.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31577

namespace LeftBound31581
def owner : Owner := ⟨.program ⟨257⟩, ⟨61620⟩⟩
def transferEvent : Nat := 31581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31579 .coefficient, .predecessor 1 31580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31579 .coefficient)
      LeftBound31576.bound (LeftBound31576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31580 .coefficient)
      LeftBound30022.bound (LeftBound30022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31576.bound, LeftBound30022.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31576.bound, LeftBound30022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31576.actual selector witness, LeftBound30022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31581

namespace LeftBound31582
def owner : Owner := ⟨.program ⟨257⟩, ⟨61620⟩⟩
def transferEvent : Nat := 31582
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31578 .summary, .result 30029 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31578 .summary)
      LeftBound31577.bound (LeftBound31577.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58640⟩⟩) (rawTerms := some (Proof.Events123.exact31578RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 30029 .summary)
      LeftBound30024.bound (LeftBound30024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61619⟩⟩) (rawTerms := some (Proof.Events117.exact30029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31577.bound, LeftBound30024.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31577.bound, LeftBound30024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31577.actual selector witness, LeftBound30024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31582

namespace LeftBound31586
def owner : Owner := ⟨.program ⟨257⟩, ⟨64600⟩⟩
def transferEvent : Nat := 31586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31584 .coefficient, .predecessor 1 31585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31584 .coefficient)
      LeftBound31581.bound (LeftBound31581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31585 .coefficient)
      LeftBound29810.bound (LeftBound29810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31581.bound, LeftBound29810.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31581.bound, LeftBound29810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31581.actual selector witness, LeftBound29810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31586

namespace LeftBound31587
def owner : Owner := ⟨.program ⟨257⟩, ⟨64600⟩⟩
def transferEvent : Nat := 31587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31583 .summary, .result 29817 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31583 .summary)
      LeftBound31582.bound (LeftBound31582.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61620⟩⟩) (rawTerms := some (Proof.Events123.exact31583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 29817 .summary)
      LeftBound29812.bound (LeftBound29812.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64599⟩⟩) (rawTerms := some (Proof.Events116.exact29817RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31582.bound, LeftBound29812.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31582.bound, LeftBound29812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31582.actual selector witness, LeftBound29812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31587

namespace LeftBound31591
def owner : Owner := ⟨.program ⟨257⟩, ⟨69481⟩⟩
def transferEvent : Nat := 31591
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31589 .coefficient, .predecessor 1 31590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31589 .coefficient)
      LeftBound31586.bound (LeftBound31586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31590 .coefficient)
      LeftBound29598.bound (LeftBound29598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31586.bound, LeftBound29598.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31586.bound, LeftBound29598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31586.actual selector witness, LeftBound29598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31591

namespace LeftBound31592
def owner : Owner := ⟨.program ⟨257⟩, ⟨69481⟩⟩
def transferEvent : Nat := 31592
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31588 .summary, .result 29605 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31588 .summary)
      LeftBound31587.bound (LeftBound31587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64600⟩⟩) (rawTerms := some (Proof.Events123.exact31588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 29605 .summary)
      LeftBound29600.bound (LeftBound29600.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69480⟩⟩) (rawTerms := some (Proof.Events115.exact29605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29600.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31587.bound, LeftBound29600.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31587.bound, LeftBound29600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31587.actual selector witness, LeftBound29600.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31592

namespace LeftBound31596
def owner : Owner := ⟨.program ⟨257⟩, ⟨69482⟩⟩
def transferEvent : Nat := 31596
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31594 .coefficient, .predecessor 1 31595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31594 .coefficient)
      LeftBound31591.bound (LeftBound31591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31595 .coefficient)
      LeftBound29386.bound (LeftBound29386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31591.bound, LeftBound29386.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31591.bound, LeftBound29386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31591.actual selector witness, LeftBound29386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31596

namespace LeftBound31597
def owner : Owner := ⟨.program ⟨257⟩, ⟨69482⟩⟩
def transferEvent : Nat := 31597
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31593 .summary, .result 29393 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31593 .summary)
      LeftBound31592.bound (LeftBound31592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69481⟩⟩) (rawTerms := some (Proof.Events123.exact31593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 29393 .summary)
      LeftBound29388.bound (LeftBound29388.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28069⟩⟩) (rawTerms := some (Proof.Events114.exact29393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31592.bound, LeftBound29388.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31592.bound, LeftBound29388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31592.actual selector witness, LeftBound29388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31597

namespace LeftBound31601
def owner : Owner := ⟨.program ⟨257⟩, ⟨69483⟩⟩
def transferEvent : Nat := 31601
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31599 .coefficient, .predecessor 1 31600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31599 .coefficient)
      LeftBound31596.bound (LeftBound31596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31600 .coefficient)
      LeftBound29174.bound (LeftBound29174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31596.bound, LeftBound29174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31596.bound, LeftBound29174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31596.actual selector witness, LeftBound29174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31601

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
