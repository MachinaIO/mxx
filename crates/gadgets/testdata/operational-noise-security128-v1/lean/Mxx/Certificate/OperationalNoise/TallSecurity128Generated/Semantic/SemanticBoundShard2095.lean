import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1083
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1590
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1692
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2094

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound308123
def owner : Owner := ⟨.program ⟨257⟩, ⟨71183⟩⟩
def transferEvent : Nat := 308123
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308121 .coefficient, .predecessor 1 308122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308121 .coefficient)
      LeftBound308118.bound (LeftBound308118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308122 .coefficient)
      LeftBound251173.bound (LeftBound251173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308118.bound, LeftBound251173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308118.bound, LeftBound251173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308118.actual selector witness, LeftBound251173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308123

namespace LeftBound308124
def owner : Owner := ⟨.program ⟨257⟩, ⟨71183⟩⟩
def transferEvent : Nat := 308124
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308120 .summary, .result 251234 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308120 .summary)
      LeftBound308119.bound (LeftBound308119.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71093⟩⟩) (rawTerms := some (Proof.Events1203.exact308120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251234 .summary)
      LeftBound251175.bound (LeftBound251175.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71182⟩⟩) (rawTerms := some (Proof.Events981.exact251234RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308119.bound, LeftBound251175.bound]
def bound : CoeffClass := .finite ⟨30808454790312530031291914359231165163455306056856023605184929939366871092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308119.bound, LeftBound251175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308119.actual selector witness, LeftBound251175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308124

namespace LeftBound308128
def owner : Owner := ⟨.program ⟨257⟩, ⟨71215⟩⟩
def transferEvent : Nat := 308128
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308126 .coefficient, .predecessor 1 308127 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308126 .coefficient)
      LeftBound308123.bound (LeftBound308123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308127 .coefficient)
      LeftBound236548.bound (LeftBound236548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308123.bound, LeftBound236548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308123.bound, LeftBound236548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308123.actual selector witness, LeftBound236548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308128

namespace LeftBound308129
def owner : Owner := ⟨.program ⟨257⟩, ⟨71215⟩⟩
def transferEvent : Nat := 308129
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308125 .summary, .result 236609 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308125 .summary)
      LeftBound308124.bound (LeftBound308124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71183⟩⟩) (rawTerms := some (Proof.Events1203.exact308125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236609 .summary)
      LeftBound236550.bound (LeftBound236550.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71214⟩⟩) (rawTerms := some (Proof.Events924.exact236609RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308124.bound, LeftBound236550.bound]
def bound : CoeffClass := .finite ⟨38510568487711333730148828037264203008605515576528206841913578787574906932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308124.bound, LeftBound236550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308124.actual selector witness, LeftBound236550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308129

namespace LeftBound308133
def owner : Owner := ⟨.program ⟨257⟩, ⟨71247⟩⟩
def transferEvent : Nat := 308133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308131 .coefficient, .predecessor 1 308132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308131 .coefficient)
      LeftBound308128.bound (LeftBound308128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308132 .coefficient)
      LeftBound221923.bound (LeftBound221923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact221984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound221923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308128.bound, LeftBound221923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308128.bound, LeftBound221923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308128.actual selector witness, LeftBound221923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308133

namespace LeftBound308134
def owner : Owner := ⟨.program ⟨257⟩, ⟨71247⟩⟩
def transferEvent : Nat := 308134
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308130 .summary, .result 221984 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308130 .summary)
      LeftBound308129.bound (LeftBound308129.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71215⟩⟩) (rawTerms := some (Proof.Events1203.exact308130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221984 .summary)
      LeftBound221925.bound (LeftBound221925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71246⟩⟩) (rawTerms := some (Proof.Events867.exact221984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound221925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308129.bound, LeftBound221925.bound]
def bound : CoeffClass := .finite ⟨46212682185110137429005741715297240853755725096200390078642227635782942772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308129.bound, LeftBound221925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308129.actual selector witness, LeftBound221925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308134

namespace LeftBound308138
def owner : Owner := ⟨.program ⟨257⟩, ⟨71308⟩⟩
def transferEvent : Nat := 308138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308136 .coefficient, .predecessor 1 308137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308136 .coefficient)
      LeftBound308133.bound (LeftBound308133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308137 .coefficient)
      LeftBound207298.bound (LeftBound207298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308133.bound, LeftBound207298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308133.bound, LeftBound207298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308133.actual selector witness, LeftBound207298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308138

namespace LeftBound308139
def owner : Owner := ⟨.program ⟨257⟩, ⟨71308⟩⟩
def transferEvent : Nat := 308139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308135 .summary, .result 207359 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308135 .summary)
      LeftBound308134.bound (LeftBound308134.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71247⟩⟩) (rawTerms := some (Proof.Events1203.exact308135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207359 .summary)
      LeftBound207300.bound (LeftBound207300.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71307⟩⟩) (rawTerms := some (Proof.Events809.exact207359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308134.bound, LeftBound207300.bound]
def bound : CoeffClass := .finite ⟨53914795882508941127862655393330278698905934615872573315370876483990978612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308134.bound, LeftBound207300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308134.actual selector witness, LeftBound207300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308139

namespace LeftBound308143
def owner : Owner := ⟨.program ⟨257⟩, ⟨71340⟩⟩
def transferEvent : Nat := 308143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308141 .coefficient, .predecessor 1 308142 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308141 .coefficient)
      LeftBound308138.bound (LeftBound308138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308142 .coefficient)
      LeftBound192673.bound (LeftBound192673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events752.exact192734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308138.bound, LeftBound192673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308138.bound, LeftBound192673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308138.actual selector witness, LeftBound192673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308143

namespace LeftBound308144
def owner : Owner := ⟨.program ⟨257⟩, ⟨71340⟩⟩
def transferEvent : Nat := 308144
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308140 .summary, .result 192734 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308140 .summary)
      LeftBound308139.bound (LeftBound308139.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71308⟩⟩) (rawTerms := some (Proof.Events1203.exact308140RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192734 .summary)
      LeftBound192675.bound (LeftBound192675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71339⟩⟩) (rawTerms := some (Proof.Events752.exact192734RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308139.bound, LeftBound192675.bound]
def bound : CoeffClass := .finite ⟨61616909579907744826719569071363316544056144135544756552099525332199014452, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308139.bound, LeftBound192675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308139.actual selector witness, LeftBound192675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308144

namespace LeftBound308148
def owner : Owner := ⟨.program ⟨257⟩, ⟨71376⟩⟩
def transferEvent : Nat := 308148
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308146 .coefficient, .predecessor 1 308147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308146 .coefficient)
      LeftBound308143.bound (LeftBound308143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308147 .coefficient)
      LeftBound178048.bound (LeftBound178048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308143.bound, LeftBound178048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308143.bound, LeftBound178048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308143.actual selector witness, LeftBound178048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308148

namespace LeftBound308149
def owner : Owner := ⟨.program ⟨257⟩, ⟨71376⟩⟩
def transferEvent : Nat := 308149
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308145 .summary, .result 178109 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308145 .summary)
      LeftBound308144.bound (LeftBound308144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71340⟩⟩) (rawTerms := some (Proof.Events1203.exact308145RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178109 .summary)
      LeftBound178050.bound (LeftBound178050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71375⟩⟩) (rawTerms := some (Proof.Events695.exact178109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308144.bound, LeftBound178050.bound]
def bound : CoeffClass := .finite ⟨69319023277306548525576482749396354389206353655216939788828174180407050292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308144.bound, LeftBound178050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308144.actual selector witness, LeftBound178050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308149

namespace LeftBound308153
def owner : Owner := ⟨.program ⟨257⟩, ⟨71377⟩⟩
def transferEvent : Nat := 308153
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308151 .coefficient, .predecessor 1 308152 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308151 .coefficient)
      LeftBound308148.bound (LeftBound308148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308152 .coefficient)
      LeftBound163423.bound (LeftBound163423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events638.exact163484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308148.bound, LeftBound163423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308148.bound, LeftBound163423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308148.actual selector witness, LeftBound163423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308153

namespace LeftBound308154
def owner : Owner := ⟨.program ⟨257⟩, ⟨71377⟩⟩
def transferEvent : Nat := 308154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308150 .summary, .result 163484 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308150 .summary)
      LeftBound308149.bound (LeftBound308149.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71376⟩⟩) (rawTerms := some (Proof.Events1203.exact308150RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163484 .summary)
      LeftBound163425.bound (LeftBound163425.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71152⟩⟩) (rawTerms := some (Proof.Events638.exact163484RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308149.bound, LeftBound163425.bound]
def bound : CoeffClass := .finite ⟨77021136974705352224433396427429392234356563174889123025556823028615086132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308149.bound, LeftBound163425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308149.actual selector witness, LeftBound163425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308154

namespace LeftBound308158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71378⟩⟩
def transferEvent : Nat := 308158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308156 .coefficient, .predecessor 1 308157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308156 .coefficient)
      LeftBound308153.bound (LeftBound308153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308157 .coefficient)
      LeftBound148798.bound (LeftBound148798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308153.bound, LeftBound148798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308153.bound, LeftBound148798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308153.actual selector witness, LeftBound148798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308158

namespace LeftBound308159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71378⟩⟩
def transferEvent : Nat := 308159
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308155 .summary, .result 148859 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308155 .summary)
      LeftBound308154.bound (LeftBound308154.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71377⟩⟩) (rawTerms := some (Proof.Events1203.exact308155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148859 .summary)
      LeftBound148800.bound (LeftBound148800.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71027⟩⟩) (rawTerms := some (Proof.Events581.exact148859RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308154.bound, LeftBound148800.bound]
def bound : CoeffClass := .finite ⟨84723250672104155923290310105462430079506772694561306262285471876823121972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308154.bound, LeftBound148800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308154.actual selector witness, LeftBound148800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308159

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
