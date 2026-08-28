import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1597
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1601
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1605
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1608
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1612
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1660

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound245508
def owner : Owner := ⟨.program ⟨257⟩, ⟨70027⟩⟩
def transferEvent : Nat := 245508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245506 .coefficient, .predecessor 1 245507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245506 .coefficient)
      LeftBound245503.bound (LeftBound245503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245507 .coefficient)
      LeftBound239175.bound (LeftBound239175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events934.exact239179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound239175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound239175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245503.bound, LeftBound239175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245503.bound, LeftBound239175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245503.actual selector witness, LeftBound239175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245508

namespace LeftBound245509
def owner : Owner := ⟨.program ⟨257⟩, ⟨70027⟩⟩
def transferEvent : Nat := 245509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245505 .summary, .result 239179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245505 .summary)
      LeftBound245504.bound (LeftBound245504.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70026⟩⟩) (rawTerms := some (Proof.Events959.exact245505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 239179 .summary)
      LeftBound239178.bound (LeftBound239178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39262⟩⟩) (rawTerms := some (Proof.Events934.exact239179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound239178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245504.bound, LeftBound239178.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245504.bound, LeftBound239178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245504.actual selector witness, LeftBound239178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245509

namespace LeftBound245513
def owner : Owner := ⟨.program ⟨257⟩, ⟨70028⟩⟩
def transferEvent : Nat := 245513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245511 .coefficient, .predecessor 1 245512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245511 .coefficient)
      LeftBound245508.bound (LeftBound245508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245512 .coefficient)
      LeftBound238693.bound (LeftBound238693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245508.bound, LeftBound238693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245508.bound, LeftBound238693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245508.actual selector witness, LeftBound238693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245513

namespace LeftBound245514
def owner : Owner := ⟨.program ⟨257⟩, ⟨70028⟩⟩
def transferEvent : Nat := 245514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245510 .summary, .result 238697 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245510 .summary)
      LeftBound245509.bound (LeftBound245509.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70027⟩⟩) (rawTerms := some (Proof.Events959.exact245510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238697 .summary)
      LeftBound238696.bound (LeftBound238696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41942⟩⟩) (rawTerms := some (Proof.Events932.exact238697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245509.bound, LeftBound238696.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245509.bound, LeftBound238696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245509.actual selector witness, LeftBound238696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245514

namespace LeftBound245518
def owner : Owner := ⟨.program ⟨257⟩, ⟨70029⟩⟩
def transferEvent : Nat := 245518
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245516 .coefficient, .predecessor 1 245517 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245516 .coefficient)
      LeftBound245513.bound (LeftBound245513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245517 .coefficient)
      LeftBound238211.bound (LeftBound238211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238211.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245513.bound, LeftBound238211.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245513.bound, LeftBound238211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245513.actual selector witness, LeftBound238211.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245518

namespace LeftBound245519
def owner : Owner := ⟨.program ⟨257⟩, ⟨70029⟩⟩
def transferEvent : Nat := 245519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245515 .summary, .result 238215 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245515 .summary)
      LeftBound245514.bound (LeftBound245514.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70028⟩⟩) (rawTerms := some (Proof.Events959.exact245515RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238215 .summary)
      LeftBound238214.bound (LeftBound238214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44622⟩⟩) (rawTerms := some (Proof.Events930.exact238215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245514.bound, LeftBound238214.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245514.bound, LeftBound238214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245514.actual selector witness, LeftBound238214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245519

namespace LeftBound245523
def owner : Owner := ⟨.program ⟨257⟩, ⟨70030⟩⟩
def transferEvent : Nat := 245523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245521 .coefficient, .predecessor 1 245522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245521 .coefficient)
      LeftBound245518.bound (LeftBound245518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245522 .coefficient)
      LeftBound237729.bound (LeftBound237729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245518.bound, LeftBound237729.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245518.bound, LeftBound237729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245518.actual selector witness, LeftBound237729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245523

namespace LeftBound245524
def owner : Owner := ⟨.program ⟨257⟩, ⟨70030⟩⟩
def transferEvent : Nat := 245524
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245520 .summary, .result 237733 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245520 .summary)
      LeftBound245519.bound (LeftBound245519.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70029⟩⟩) (rawTerms := some (Proof.Events959.exact245520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237733 .summary)
      LeftBound237732.bound (LeftBound237732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47302⟩⟩) (rawTerms := some (Proof.Events928.exact237733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245519.bound, LeftBound237732.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245519.bound, LeftBound237732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245519.actual selector witness, LeftBound237732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245524

namespace LeftBound245528
def owner : Owner := ⟨.program ⟨257⟩, ⟨70031⟩⟩
def transferEvent : Nat := 245528
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 245526 .coefficient, .predecessor 1 245527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245526 .coefficient)
      LeftBound245523.bound (LeftBound245523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245527 .coefficient)
      LeftBound237247.bound (LeftBound237247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events926.exact237251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245523.bound, LeftBound237247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245523.bound, LeftBound237247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245523.actual selector witness, LeftBound237247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245528

namespace LeftBound245529
def owner : Owner := ⟨.program ⟨257⟩, ⟨70031⟩⟩
def transferEvent : Nat := 245529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 245525 .summary, .result 237251 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245525 .summary)
      LeftBound245524.bound (LeftBound245524.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70030⟩⟩) (rawTerms := some (Proof.Events959.exact245525RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237251 .summary)
      LeftBound237250.bound (LeftBound237250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49982⟩⟩) (rawTerms := some (Proof.Events926.exact237251RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound245524.bound, LeftBound237250.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245524.bound, LeftBound237250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound245524.actual selector witness, LeftBound237250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound245529

namespace LeftBound245533
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def transferEvent : Nat := 245533
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 245531 .coefficient) (.predecessor 1 245532 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245531 .coefficient)
      LeftBound245528.bound (LeftBound245528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245532 .coefficient)
      LeftAuthority236752.bound (LeftAuthority236752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236752.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound245528.bound LeftAuthority236752.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245528.bound, LeftAuthority236752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound245528.actual selector witness) * (LeftAuthority236752.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245533

namespace LeftBound245534
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def transferEvent : Nat := 245534
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩ [⟨.result 236753 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236753 .coefficient)
      LeftAuthority236752.bound (LeftAuthority236752.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71172⟩⟩) (rawTerms := some (Proof.Events924.exact236753RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority236752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority236752.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority236752.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority236752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority236752.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound245534

namespace LeftBound245535
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def transferEvent : Nat := 245535
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 245530 .summary) (.transfer 245534) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245530 .summary)
      LeftBound245529.bound (LeftBound245529.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70031⟩⟩) (rawTerms := some (Proof.Events959.exact245530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound245529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 245534)
      LeftBound245534.bound (LeftBound245534.actual selector witness) := by
  exact .transfer (LeftBound245534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound245529.bound LeftBound245534.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound245529.bound, LeftBound245534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound245529.actual selector witness) * (LeftBound245534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245535

namespace LeftBound245614
def owner : Owner := ⟨.program ⟨257⟩, ⟨68352⟩⟩
def transferEvent : Nat := 245614
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 245612 .coefficient) (.value (.predecessor 1 245613 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245612 .coefficient)
      LeftAuthority245610.bound (LeftAuthority245610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245613 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority245610.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245610.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority245610.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound245614

namespace LeftBound245618
def owner : Owner := ⟨.program ⟨257⟩, ⟨68353⟩⟩
def transferEvent : Nat := 245618
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 245616 .coefficient) (.predecessor 1 245617 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 245616 .coefficient)
      LeftBound236867.bound (LeftBound236867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 245617 .coefficient)
      LeftBound245614.bound (LeftBound245614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events959.exact245615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound245614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound245614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236867.bound LeftBound245614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236867.bound, LeftBound245614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236867.actual selector witness) * (LeftBound245614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245618

namespace LeftBound245619
def owner : Owner := ⟨.program ⟨257⟩, ⟨68353⟩⟩
def transferEvent : Nat := 245619
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68350⟩⟩]⟩ [⟨.result 245611 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 245611 .coefficient)
      LeftAuthority245610.bound (LeftAuthority245610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68350⟩⟩) (rawTerms := some (Proof.Events959.exact245611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority245610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority245610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority245610.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority245610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority245610.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound245619

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
