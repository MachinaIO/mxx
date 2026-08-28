import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1699
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1702
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1706
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1710
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1713
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1717
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1720
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1721
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1724
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1728
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1761

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound260114
def owner : Owner := ⟨.program ⟨257⟩, ⟨69786⟩⟩
def transferEvent : Nat := 260114
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260110 .summary, .result 255732 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260110 .summary)
      LeftBound260109.bound (LeftBound260109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64721⟩⟩) (rawTerms := some (Proof.Events1016.exact260110RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 255732 .summary)
      LeftBound255731.bound (LeftBound255731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69785⟩⟩) (rawTerms := some (Proof.Events998.exact255732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound255731.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260109.bound, LeftBound255731.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260109.bound, LeftBound255731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260109.actual selector witness, LeftBound255731.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260114

namespace LeftBound260118
def owner : Owner := ⟨.program ⟨257⟩, ⟨69787⟩⟩
def transferEvent : Nat := 260118
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260116 .coefficient, .predecessor 1 260117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260116 .coefficient)
      LeftBound260113.bound (LeftBound260113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260117 .coefficient)
      LeftBound255246.bound (LeftBound255246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events997.exact255250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound255246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound255246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260113.bound, LeftBound255246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260113.bound, LeftBound255246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260113.actual selector witness, LeftBound255246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260118

namespace LeftBound260119
def owner : Owner := ⟨.program ⟨257⟩, ⟨69787⟩⟩
def transferEvent : Nat := 260119
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260115 .summary, .result 255250 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260115 .summary)
      LeftBound260114.bound (LeftBound260114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69786⟩⟩) (rawTerms := some (Proof.Events1016.exact260115RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 255250 .summary)
      LeftBound255249.bound (LeftBound255249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28167⟩⟩) (rawTerms := some (Proof.Events997.exact255250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound255249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260114.bound, LeftBound255249.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260114.bound, LeftBound255249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260114.actual selector witness, LeftBound255249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260119

namespace LeftBound260123
def owner : Owner := ⟨.program ⟨257⟩, ⟨69788⟩⟩
def transferEvent : Nat := 260123
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260121 .coefficient, .predecessor 1 260122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260121 .coefficient)
      LeftBound260118.bound (LeftBound260118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260122 .coefficient)
      LeftBound254764.bound (LeftBound254764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events995.exact254768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260118.bound, LeftBound254764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260118.bound, LeftBound254764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260118.actual selector witness, LeftBound254764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260123

namespace LeftBound260124
def owner : Owner := ⟨.program ⟨257⟩, ⟨69788⟩⟩
def transferEvent : Nat := 260124
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260120 .summary, .result 254768 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260120 .summary)
      LeftBound260119.bound (LeftBound260119.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69787⟩⟩) (rawTerms := some (Proof.Events1016.exact260120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254768 .summary)
      LeftBound254767.bound (LeftBound254767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30847⟩⟩) (rawTerms := some (Proof.Events995.exact254768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260119.bound, LeftBound254767.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260119.bound, LeftBound254767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260119.actual selector witness, LeftBound254767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260124

namespace LeftBound260128
def owner : Owner := ⟨.program ⟨257⟩, ⟨69789⟩⟩
def transferEvent : Nat := 260128
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260126 .coefficient, .predecessor 1 260127 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260126 .coefficient)
      LeftBound260123.bound (LeftBound260123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260127 .coefficient)
      LeftBound254282.bound (LeftBound254282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events993.exact254286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound254282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound254282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260123.bound, LeftBound254282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260123.bound, LeftBound254282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260123.actual selector witness, LeftBound254282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260128

namespace LeftBound260129
def owner : Owner := ⟨.program ⟨257⟩, ⟨69789⟩⟩
def transferEvent : Nat := 260129
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260125 .summary, .result 254286 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260125 .summary)
      LeftBound260124.bound (LeftBound260124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69788⟩⟩) (rawTerms := some (Proof.Events1016.exact260125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 254286 .summary)
      LeftBound254285.bound (LeftBound254285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36507⟩⟩) (rawTerms := some (Proof.Events993.exact254286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound254285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260124.bound, LeftBound254285.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260124.bound, LeftBound254285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260124.actual selector witness, LeftBound254285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260129

namespace LeftBound260133
def owner : Owner := ⟨.program ⟨257⟩, ⟨69790⟩⟩
def transferEvent : Nat := 260133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260131 .coefficient, .predecessor 1 260132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260131 .coefficient)
      LeftBound260128.bound (LeftBound260128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260132 .coefficient)
      LeftBound253800.bound (LeftBound253800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events991.exact253804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound253800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound253800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260128.bound, LeftBound253800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260128.bound, LeftBound253800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260128.actual selector witness, LeftBound253800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260133

namespace LeftBound260134
def owner : Owner := ⟨.program ⟨257⟩, ⟨69790⟩⟩
def transferEvent : Nat := 260134
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260130 .summary, .result 253804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260130 .summary)
      LeftBound260129.bound (LeftBound260129.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69789⟩⟩) (rawTerms := some (Proof.Events1016.exact260130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 253804 .summary)
      LeftBound253803.bound (LeftBound253803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39187⟩⟩) (rawTerms := some (Proof.Events991.exact253804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound253803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260129.bound, LeftBound253803.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260129.bound, LeftBound253803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260129.actual selector witness, LeftBound253803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260134

namespace LeftBound260138
def owner : Owner := ⟨.program ⟨257⟩, ⟨69791⟩⟩
def transferEvent : Nat := 260138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260136 .coefficient, .predecessor 1 260137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260136 .coefficient)
      LeftBound260133.bound (LeftBound260133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260137 .coefficient)
      LeftBound253318.bound (LeftBound253318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events989.exact253322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound253318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound253318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260133.bound, LeftBound253318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260133.bound, LeftBound253318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260133.actual selector witness, LeftBound253318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260138

namespace LeftBound260139
def owner : Owner := ⟨.program ⟨257⟩, ⟨69791⟩⟩
def transferEvent : Nat := 260139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260135 .summary, .result 253322 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260135 .summary)
      LeftBound260134.bound (LeftBound260134.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69790⟩⟩) (rawTerms := some (Proof.Events1016.exact260135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 253322 .summary)
      LeftBound253321.bound (LeftBound253321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41867⟩⟩) (rawTerms := some (Proof.Events989.exact253322RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound253321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260134.bound, LeftBound253321.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260134.bound, LeftBound253321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260134.actual selector witness, LeftBound253321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260139

namespace LeftBound260143
def owner : Owner := ⟨.program ⟨257⟩, ⟨69792⟩⟩
def transferEvent : Nat := 260143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260141 .coefficient, .predecessor 1 260142 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260141 .coefficient)
      LeftBound260138.bound (LeftBound260138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260142 .coefficient)
      LeftBound252836.bound (LeftBound252836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events987.exact252840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260138.bound, LeftBound252836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260138.bound, LeftBound252836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260138.actual selector witness, LeftBound252836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260143

namespace LeftBound260144
def owner : Owner := ⟨.program ⟨257⟩, ⟨69792⟩⟩
def transferEvent : Nat := 260144
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260140 .summary, .result 252840 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260140 .summary)
      LeftBound260139.bound (LeftBound260139.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69791⟩⟩) (rawTerms := some (Proof.Events1016.exact260140RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252840 .summary)
      LeftBound252839.bound (LeftBound252839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44547⟩⟩) (rawTerms := some (Proof.Events987.exact252840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260139.bound, LeftBound252839.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260139.bound, LeftBound252839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260139.actual selector witness, LeftBound252839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260144

namespace LeftBound260148
def owner : Owner := ⟨.program ⟨257⟩, ⟨69793⟩⟩
def transferEvent : Nat := 260148
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260146 .coefficient, .predecessor 1 260147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260146 .coefficient)
      LeftBound260143.bound (LeftBound260143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260147 .coefficient)
      LeftBound252354.bound (LeftBound252354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events985.exact252358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260143.bound, LeftBound252354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260143.bound, LeftBound252354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260143.actual selector witness, LeftBound252354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260148

namespace LeftBound260149
def owner : Owner := ⟨.program ⟨257⟩, ⟨69793⟩⟩
def transferEvent : Nat := 260149
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260145 .summary, .result 252358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260145 .summary)
      LeftBound260144.bound (LeftBound260144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69792⟩⟩) (rawTerms := some (Proof.Events1016.exact260145RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252358 .summary)
      LeftBound252357.bound (LeftBound252357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47227⟩⟩) (rawTerms := some (Proof.Events985.exact252358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260144.bound, LeftBound252357.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260144.bound, LeftBound252357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260144.actual selector witness, LeftBound252357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260149

namespace LeftBound260153
def owner : Owner := ⟨.program ⟨257⟩, ⟨69794⟩⟩
def transferEvent : Nat := 260153
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 260151 .coefficient, .predecessor 1 260152 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260151 .coefficient)
      LeftBound260148.bound (LeftBound260148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260152 .coefficient)
      LeftBound251872.bound (LeftBound251872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events983.exact251876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260148.bound, LeftBound251872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260148.bound, LeftBound251872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260148.actual selector witness, LeftBound251872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260153

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
