import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard969
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard970
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard971
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard972
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard973
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard974
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard975
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard977
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard978

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound148542
def owner : Owner := ⟨.program ⟨257⟩, ⟨9456⟩⟩
def transferEvent : Nat := 148542
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148536 .summary, .result 148536 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148536 .summary)
      LeftBound148534.bound (LeftBound148534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9306⟩⟩) (rawTerms := some (Proof.Events580.exact148536RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148536 .summary)
      LeftBound148534.bound (LeftBound148534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9306⟩⟩) (rawTerms := some (Proof.Events580.exact148536RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148534.bound, LeftBound148534.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148534.bound, LeftBound148534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148534.actual selector witness, LeftBound148534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148542

namespace LeftBound148546
def owner : Owner := ⟨.program ⟨257⟩, ⟨17563⟩⟩
def transferEvent : Nat := 148546
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148544 .coefficient, .predecessor 1 148545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148544 .coefficient)
      LeftBound148539.bound (LeftBound148539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148545 .coefficient)
      LeftBound148509.bound (LeftBound148509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148509.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148539.bound, LeftBound148509.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148539.bound, LeftBound148509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148539.actual selector witness, LeftBound148509.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148546

namespace LeftBound148547
def owner : Owner := ⟨.program ⟨257⟩, ⟨17563⟩⟩
def transferEvent : Nat := 148547
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148543 .summary, .result 148516 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148543 .summary)
      LeftBound148542.bound (LeftBound148542.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9456⟩⟩) (rawTerms := some (Proof.Events580.exact148543RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148516 .summary)
      LeftBound148511.bound (LeftBound148511.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17562⟩⟩) (rawTerms := some (Proof.Events580.exact148516RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148542.bound, LeftBound148511.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148542.bound, LeftBound148511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148542.actual selector witness, LeftBound148511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148547

namespace LeftBound148551
def owner : Owner := ⟨.program ⟨257⟩, ⟨20433⟩⟩
def transferEvent : Nat := 148551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148549 .coefficient, .predecessor 1 148550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148549 .coefficient)
      LeftBound148546.bound (LeftBound148546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148550 .coefficient)
      LeftBound148297.bound (LeftBound148297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events579.exact148304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148546.bound, LeftBound148297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148546.bound, LeftBound148297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148546.actual selector witness, LeftBound148297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148551

namespace LeftBound148552
def owner : Owner := ⟨.program ⟨257⟩, ⟨20433⟩⟩
def transferEvent : Nat := 148552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148548 .summary, .result 148304 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148548 .summary)
      LeftBound148547.bound (LeftBound148547.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17563⟩⟩) (rawTerms := some (Proof.Events580.exact148548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148304 .summary)
      LeftBound148299.bound (LeftBound148299.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20432⟩⟩) (rawTerms := some (Proof.Events579.exact148304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148547.bound, LeftBound148299.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148547.bound, LeftBound148299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148547.actual selector witness, LeftBound148299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148552

namespace LeftBound148556
def owner : Owner := ⟨.program ⟨257⟩, ⟨23653⟩⟩
def transferEvent : Nat := 148556
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148554 .coefficient, .predecessor 1 148555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148554 .coefficient)
      LeftBound148551.bound (LeftBound148551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148555 .coefficient)
      LeftBound148085.bound (LeftBound148085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events578.exact148092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148551.bound, LeftBound148085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148551.bound, LeftBound148085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148551.actual selector witness, LeftBound148085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148556

namespace LeftBound148557
def owner : Owner := ⟨.program ⟨257⟩, ⟨23653⟩⟩
def transferEvent : Nat := 148557
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148553 .summary, .result 148092 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148553 .summary)
      LeftBound148552.bound (LeftBound148552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20433⟩⟩) (rawTerms := some (Proof.Events580.exact148553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148092 .summary)
      LeftBound148087.bound (LeftBound148087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23652⟩⟩) (rawTerms := some (Proof.Events578.exact148092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148552.bound, LeftBound148087.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148552.bound, LeftBound148087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148552.actual selector witness, LeftBound148087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148557

namespace LeftBound148561
def owner : Owner := ⟨.program ⟨257⟩, ⟨33673⟩⟩
def transferEvent : Nat := 148561
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148559 .coefficient, .predecessor 1 148560 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148559 .coefficient)
      LeftBound148556.bound (LeftBound148556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148560 .coefficient)
      LeftBound147873.bound (LeftBound147873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events577.exact147880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound147873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound147873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148556.bound, LeftBound147873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148556.bound, LeftBound147873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148556.actual selector witness, LeftBound147873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148561

namespace LeftBound148562
def owner : Owner := ⟨.program ⟨257⟩, ⟨33673⟩⟩
def transferEvent : Nat := 148562
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148558 .summary, .result 147880 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148558 .summary)
      LeftBound148557.bound (LeftBound148557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23653⟩⟩) (rawTerms := some (Proof.Events580.exact148558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 147880 .summary)
      LeftBound147875.bound (LeftBound147875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33672⟩⟩) (rawTerms := some (Proof.Events577.exact147880RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound147875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148557.bound, LeftBound147875.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148557.bound, LeftBound147875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148557.actual selector witness, LeftBound147875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148562

namespace LeftBound148566
def owner : Owner := ⟨.program ⟨257⟩, ⟨52733⟩⟩
def transferEvent : Nat := 148566
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148564 .coefficient, .predecessor 1 148565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148564 .coefficient)
      LeftBound148561.bound (LeftBound148561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148565 .coefficient)
      LeftBound147661.bound (LeftBound147661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events576.exact147668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound147661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound147661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148561.bound, LeftBound147661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148561.bound, LeftBound147661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148561.actual selector witness, LeftBound147661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148566

namespace LeftBound148567
def owner : Owner := ⟨.program ⟨257⟩, ⟨52733⟩⟩
def transferEvent : Nat := 148567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148563 .summary, .result 147668 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148563 .summary)
      LeftBound148562.bound (LeftBound148562.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33673⟩⟩) (rawTerms := some (Proof.Events580.exact148563RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 147668 .summary)
      LeftBound147663.bound (LeftBound147663.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52732⟩⟩) (rawTerms := some (Proof.Events576.exact147668RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound147663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148562.bound, LeftBound147663.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148562.bound, LeftBound147663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148562.actual selector witness, LeftBound147663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148567

namespace LeftBound148571
def owner : Owner := ⟨.program ⟨257⟩, ⟨55713⟩⟩
def transferEvent : Nat := 148571
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148569 .coefficient, .predecessor 1 148570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148569 .coefficient)
      LeftBound148566.bound (LeftBound148566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148570 .coefficient)
      LeftBound147449.bound (LeftBound147449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events576.exact147456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound147449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound147449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148566.bound, LeftBound147449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148566.bound, LeftBound147449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148566.actual selector witness, LeftBound147449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148571

namespace LeftBound148572
def owner : Owner := ⟨.program ⟨257⟩, ⟨55713⟩⟩
def transferEvent : Nat := 148572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148568 .summary, .result 147456 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148568 .summary)
      LeftBound148567.bound (LeftBound148567.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52733⟩⟩) (rawTerms := some (Proof.Events580.exact148568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 147456 .summary)
      LeftBound147451.bound (LeftBound147451.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55712⟩⟩) (rawTerms := some (Proof.Events576.exact147456RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound147451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148567.bound, LeftBound147451.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148567.bound, LeftBound147451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148567.actual selector witness, LeftBound147451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148572

namespace LeftBound148576
def owner : Owner := ⟨.program ⟨257⟩, ⟨58693⟩⟩
def transferEvent : Nat := 148576
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148574 .coefficient, .predecessor 1 148575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148574 .coefficient)
      LeftBound148571.bound (LeftBound148571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148575 .coefficient)
      LeftBound147237.bound (LeftBound147237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events575.exact147244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound147237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound147237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148571.bound, LeftBound147237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148571.bound, LeftBound147237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148571.actual selector witness, LeftBound147237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148576

namespace LeftBound148577
def owner : Owner := ⟨.program ⟨257⟩, ⟨58693⟩⟩
def transferEvent : Nat := 148577
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148573 .summary, .result 147244 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148573 .summary)
      LeftBound148572.bound (LeftBound148572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55713⟩⟩) (rawTerms := some (Proof.Events580.exact148573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 147244 .summary)
      LeftBound147239.bound (LeftBound147239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58692⟩⟩) (rawTerms := some (Proof.Events575.exact147244RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound147239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148572.bound, LeftBound147239.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148572.bound, LeftBound147239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148572.actual selector witness, LeftBound147239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148577

namespace LeftBound148581
def owner : Owner := ⟨.program ⟨257⟩, ⟨61673⟩⟩
def transferEvent : Nat := 148581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148579 .coefficient, .predecessor 1 148580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148579 .coefficient)
      LeftBound148576.bound (LeftBound148576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148580 .coefficient)
      LeftBound147025.bound (LeftBound147025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events574.exact147032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound147025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound147025.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148576.bound, LeftBound147025.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148576.bound, LeftBound147025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148576.actual selector witness, LeftBound147025.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148581

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
