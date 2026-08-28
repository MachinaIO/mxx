import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1329
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1333
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1336
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1337
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1340
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1344
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1347
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1351
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1354

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound201553
def owner : Owner := ⟨.program ⟨257⟩, ⟨17821⟩⟩
def transferEvent : Nat := 201553
def frameStart : Nat := 201453
def rule : BoundRule := .sum [.predecessor 0 201551 .coefficient, .predecessor 1 201552 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201551 .coefficient)
      LeftBound201549.bound (LeftBound201549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201552 .coefficient)
      LeftBound201530.bound (LeftBound201530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201549.bound, LeftBound201530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201549.bound, LeftBound201530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201549.actual selector witness, LeftBound201530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201553

namespace LeftBound201566
def owner : Owner := ⟨.program ⟨257⟩, ⟨17820⟩⟩
def transferEvent : Nat := 201566
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201564 .coefficient, .predecessor 1 201565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201564 .coefficient)
      LeftBound201395.bound (LeftBound201395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201565 .coefficient)
      LeftBound201378.bound (LeftBound201378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events786.exact201385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201395.bound, LeftBound201378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201395.bound, LeftBound201378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201395.actual selector witness, LeftBound201378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201566

namespace LeftBound201569
def owner : Owner := ⟨.program ⟨257⟩, ⟨17820⟩⟩
def transferEvent : Nat := 201569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201563 .summary, .result 201385 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201563 .summary)
      LeftBound201397.bound (LeftBound201397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16639⟩⟩) (rawTerms := some (Proof.Events787.exact201563RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201385 .summary)
      LeftBound201380.bound (LeftBound201380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17819⟩⟩) (rawTerms := some (Proof.Events786.exact201385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201397.bound, LeftBound201380.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201397.bound, LeftBound201380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201397.actual selector witness, LeftBound201380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201569

namespace LeftBound201573
def owner : Owner := ⟨.program ⟨257⟩, ⟨20718⟩⟩
def transferEvent : Nat := 201573
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201571 .coefficient, .predecessor 1 201572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201571 .coefficient)
      LeftBound201566.bound (LeftBound201566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201572 .coefficient)
      LeftBound201084.bound (LeftBound201084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events785.exact201088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201084.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201566.bound, LeftBound201084.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201566.bound, LeftBound201084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201566.actual selector witness, LeftBound201084.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201573

namespace LeftBound201574
def owner : Owner := ⟨.program ⟨257⟩, ⟨20718⟩⟩
def transferEvent : Nat := 201574
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201570 .summary, .result 201088 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201570 .summary)
      LeftBound201569.bound (LeftBound201569.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17820⟩⟩) (rawTerms := some (Proof.Events787.exact201570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201088 .summary)
      LeftBound201087.bound (LeftBound201087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20717⟩⟩) (rawTerms := some (Proof.Events785.exact201088RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201569.bound, LeftBound201087.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201569.bound, LeftBound201087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201569.actual selector witness, LeftBound201087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201574

namespace LeftBound201578
def owner : Owner := ⟨.program ⟨257⟩, ⟨23938⟩⟩
def transferEvent : Nat := 201578
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201576 .coefficient, .predecessor 1 201577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201576 .coefficient)
      LeftBound201573.bound (LeftBound201573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201577 .coefficient)
      LeftBound200602.bound (LeftBound200602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200602.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201573.bound, LeftBound200602.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201573.bound, LeftBound200602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201573.actual selector witness, LeftBound200602.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201578

namespace LeftBound201579
def owner : Owner := ⟨.program ⟨257⟩, ⟨23938⟩⟩
def transferEvent : Nat := 201579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201575 .summary, .result 200606 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201575 .summary)
      LeftBound201574.bound (LeftBound201574.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20718⟩⟩) (rawTerms := some (Proof.Events787.exact201575RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200606 .summary)
      LeftBound200605.bound (LeftBound200605.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23937⟩⟩) (rawTerms := some (Proof.Events783.exact200606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201574.bound, LeftBound200605.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201574.bound, LeftBound200605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201574.actual selector witness, LeftBound200605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201579

namespace LeftBound201583
def owner : Owner := ⟨.program ⟨257⟩, ⟨33958⟩⟩
def transferEvent : Nat := 201583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201581 .coefficient, .predecessor 1 201582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201581 .coefficient)
      LeftBound201578.bound (LeftBound201578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201582 .coefficient)
      LeftBound200120.bound (LeftBound200120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events781.exact200124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201578.bound, LeftBound200120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201578.bound, LeftBound200120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201578.actual selector witness, LeftBound200120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201583

namespace LeftBound201584
def owner : Owner := ⟨.program ⟨257⟩, ⟨33958⟩⟩
def transferEvent : Nat := 201584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201580 .summary, .result 200124 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201580 .summary)
      LeftBound201579.bound (LeftBound201579.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23938⟩⟩) (rawTerms := some (Proof.Events787.exact201580RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200124 .summary)
      LeftBound200123.bound (LeftBound200123.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33957⟩⟩) (rawTerms := some (Proof.Events781.exact200124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201579.bound, LeftBound200123.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201579.bound, LeftBound200123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201579.actual selector witness, LeftBound200123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201584

namespace LeftBound201588
def owner : Owner := ⟨.program ⟨257⟩, ⟨53018⟩⟩
def transferEvent : Nat := 201588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201586 .coefficient, .predecessor 1 201587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201586 .coefficient)
      LeftBound201583.bound (LeftBound201583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201587 .coefficient)
      LeftBound199638.bound (LeftBound199638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events779.exact199642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199638.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201583.bound, LeftBound199638.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201583.bound, LeftBound199638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201583.actual selector witness, LeftBound199638.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201588

namespace LeftBound201589
def owner : Owner := ⟨.program ⟨257⟩, ⟨53018⟩⟩
def transferEvent : Nat := 201589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201585 .summary, .result 199642 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201585 .summary)
      LeftBound201584.bound (LeftBound201584.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33958⟩⟩) (rawTerms := some (Proof.Events787.exact201585RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199642 .summary)
      LeftBound199641.bound (LeftBound199641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53017⟩⟩) (rawTerms := some (Proof.Events779.exact199642RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201584.bound, LeftBound199641.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201584.bound, LeftBound199641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201584.actual selector witness, LeftBound199641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201589

namespace LeftBound201593
def owner : Owner := ⟨.program ⟨257⟩, ⟨55998⟩⟩
def transferEvent : Nat := 201593
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201591 .coefficient, .predecessor 1 201592 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201591 .coefficient)
      LeftBound201588.bound (LeftBound201588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201592 .coefficient)
      LeftBound199156.bound (LeftBound199156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events777.exact199160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201588.bound, LeftBound199156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201588.bound, LeftBound199156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201588.actual selector witness, LeftBound199156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201593

namespace LeftBound201594
def owner : Owner := ⟨.program ⟨257⟩, ⟨55998⟩⟩
def transferEvent : Nat := 201594
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201590 .summary, .result 199160 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201590 .summary)
      LeftBound201589.bound (LeftBound201589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53018⟩⟩) (rawTerms := some (Proof.Events787.exact201590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199160 .summary)
      LeftBound199159.bound (LeftBound199159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55997⟩⟩) (rawTerms := some (Proof.Events777.exact199160RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201589.bound, LeftBound199159.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201589.bound, LeftBound199159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201589.actual selector witness, LeftBound199159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201594

namespace LeftBound201598
def owner : Owner := ⟨.program ⟨257⟩, ⟨58978⟩⟩
def transferEvent : Nat := 201598
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201596 .coefficient, .predecessor 1 201597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201596 .coefficient)
      LeftBound201593.bound (LeftBound201593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201597 .coefficient)
      LeftBound198674.bound (LeftBound198674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events776.exact198678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201593.bound, LeftBound198674.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201593.bound, LeftBound198674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201593.actual selector witness, LeftBound198674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201598

namespace LeftBound201599
def owner : Owner := ⟨.program ⟨257⟩, ⟨58978⟩⟩
def transferEvent : Nat := 201599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201595 .summary, .result 198678 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201595 .summary)
      LeftBound201594.bound (LeftBound201594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55998⟩⟩) (rawTerms := some (Proof.Events787.exact201595RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 198678 .summary)
      LeftBound198677.bound (LeftBound198677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58977⟩⟩) (rawTerms := some (Proof.Events776.exact198678RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound198677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201594.bound, LeftBound198677.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201594.bound, LeftBound198677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201594.actual selector witness, LeftBound198677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201599

namespace LeftBound201603
def owner : Owner := ⟨.program ⟨257⟩, ⟨61958⟩⟩
def transferEvent : Nat := 201603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201601 .coefficient, .predecessor 1 201602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201601 .coefficient)
      LeftBound201598.bound (LeftBound201598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201602 .coefficient)
      LeftBound198192.bound (LeftBound198192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events774.exact198196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound198192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound198192.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201598.bound, LeftBound198192.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201598.bound, LeftBound198192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201598.actual selector witness, LeftBound198192.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201603

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
