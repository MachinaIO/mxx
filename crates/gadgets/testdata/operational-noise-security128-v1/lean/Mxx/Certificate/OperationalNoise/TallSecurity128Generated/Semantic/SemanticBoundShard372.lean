import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard275
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard346
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard347
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard348
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard350
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard351
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard352
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard354
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard371

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60857
def owner : Owner := ⟨.program ⟨257⟩, ⟨70802⟩⟩
def transferEvent : Nat := 60857
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60853 .summary, .result 58222 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60853 .summary)
      LeftBound60852.bound (LeftBound60852.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70801⟩⟩) (rawTerms := some (Proof.Events237.exact60853RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58222 .summary)
      LeftBound58217.bound (LeftBound58217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36827⟩⟩) (rawTerms := some (Proof.Events227.exact58222RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58217.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60852.bound, LeftBound58217.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60852.bound, LeftBound58217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60852.actual selector witness, LeftBound58217.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60857

namespace LeftBound60861
def owner : Owner := ⟨.program ⟨257⟩, ⟨70803⟩⟩
def transferEvent : Nat := 60861
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60859 .coefficient, .predecessor 1 60860 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60859 .coefficient)
      LeftBound60856.bound (LeftBound60856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60860 .coefficient)
      LeftBound58003.bound (LeftBound58003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact58010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60856.bound, LeftBound58003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60856.bound, LeftBound58003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60856.actual selector witness, LeftBound58003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60861

namespace LeftBound60862
def owner : Owner := ⟨.program ⟨257⟩, ⟨70803⟩⟩
def transferEvent : Nat := 60862
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60858 .summary, .result 58010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60858 .summary)
      LeftBound60857.bound (LeftBound60857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70802⟩⟩) (rawTerms := some (Proof.Events237.exact60858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 58010 .summary)
      LeftBound58005.bound (LeftBound58005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39507⟩⟩) (rawTerms := some (Proof.Events226.exact58010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60857.bound, LeftBound58005.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60857.bound, LeftBound58005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60857.actual selector witness, LeftBound58005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60862

namespace LeftBound60866
def owner : Owner := ⟨.program ⟨257⟩, ⟨70804⟩⟩
def transferEvent : Nat := 60866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60864 .coefficient, .predecessor 1 60865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60864 .coefficient)
      LeftBound60861.bound (LeftBound60861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60865 .coefficient)
      LeftBound57791.bound (LeftBound57791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events225.exact57798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60861.bound, LeftBound57791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60861.bound, LeftBound57791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60861.actual selector witness, LeftBound57791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60866

namespace LeftBound60867
def owner : Owner := ⟨.program ⟨257⟩, ⟨70804⟩⟩
def transferEvent : Nat := 60867
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60863 .summary, .result 57798 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60863 .summary)
      LeftBound60862.bound (LeftBound60862.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70803⟩⟩) (rawTerms := some (Proof.Events237.exact60863RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57798 .summary)
      LeftBound57793.bound (LeftBound57793.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42187⟩⟩) (rawTerms := some (Proof.Events225.exact57798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60862.bound, LeftBound57793.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60862.bound, LeftBound57793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60862.actual selector witness, LeftBound57793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60867

namespace LeftBound60871
def owner : Owner := ⟨.program ⟨257⟩, ⟨70805⟩⟩
def transferEvent : Nat := 60871
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60869 .coefficient, .predecessor 1 60870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60869 .coefficient)
      LeftBound60866.bound (LeftBound60866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60870 .coefficient)
      LeftBound57579.bound (LeftBound57579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60866.bound, LeftBound57579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60866.bound, LeftBound57579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60866.actual selector witness, LeftBound57579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60871

namespace LeftBound60872
def owner : Owner := ⟨.program ⟨257⟩, ⟨70805⟩⟩
def transferEvent : Nat := 60872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60868 .summary, .result 57586 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60868 .summary)
      LeftBound60867.bound (LeftBound60867.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70804⟩⟩) (rawTerms := some (Proof.Events237.exact60868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57586 .summary)
      LeftBound57581.bound (LeftBound57581.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44867⟩⟩) (rawTerms := some (Proof.Events224.exact57586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57581.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60867.bound, LeftBound57581.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60867.bound, LeftBound57581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60867.actual selector witness, LeftBound57581.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60872

namespace LeftBound60876
def owner : Owner := ⟨.program ⟨257⟩, ⟨70806⟩⟩
def transferEvent : Nat := 60876
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60874 .coefficient, .predecessor 1 60875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60874 .coefficient)
      LeftBound60871.bound (LeftBound60871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60875 .coefficient)
      LeftBound57367.bound (LeftBound57367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57367.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60871.bound, LeftBound57367.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60871.bound, LeftBound57367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60871.actual selector witness, LeftBound57367.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60876

namespace LeftBound60877
def owner : Owner := ⟨.program ⟨257⟩, ⟨70806⟩⟩
def transferEvent : Nat := 60877
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60873 .summary, .result 57374 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60873 .summary)
      LeftBound60872.bound (LeftBound60872.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70805⟩⟩) (rawTerms := some (Proof.Events237.exact60873RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57374 .summary)
      LeftBound57369.bound (LeftBound57369.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47547⟩⟩) (rawTerms := some (Proof.Events224.exact57374RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60872.bound, LeftBound57369.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60872.bound, LeftBound57369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60872.actual selector witness, LeftBound57369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60877

namespace LeftBound60881
def owner : Owner := ⟨.program ⟨257⟩, ⟨70807⟩⟩
def transferEvent : Nat := 60881
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60879 .coefficient, .predecessor 1 60880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60879 .coefficient)
      LeftBound60876.bound (LeftBound60876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60880 .coefficient)
      LeftBound57155.bound (LeftBound57155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60876.bound, LeftBound57155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60876.bound, LeftBound57155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60876.actual selector witness, LeftBound57155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60881

namespace LeftBound60882
def owner : Owner := ⟨.program ⟨257⟩, ⟨70807⟩⟩
def transferEvent : Nat := 60882
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60878 .summary, .result 57162 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60878 .summary)
      LeftBound60877.bound (LeftBound60877.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70806⟩⟩) (rawTerms := some (Proof.Events237.exact60878RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 57162 .summary)
      LeftBound57157.bound (LeftBound57157.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50227⟩⟩) (rawTerms := some (Proof.Events223.exact57162RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60877.bound, LeftBound57157.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60877.bound, LeftBound57157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60877.actual selector witness, LeftBound57157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60882

namespace LeftBound60886
def owner : Owner := ⟨.program ⟨257⟩, ⟨71507⟩⟩
def transferEvent : Nat := 60886
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60884 .coefficient, .predecessor 1 60885 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60884 .coefficient)
      LeftBound60881.bound (LeftBound60881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60885 .coefficient)
      LeftBound56943.bound (LeftBound56943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60881.bound, LeftBound56943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60881.bound, LeftBound56943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60881.actual selector witness, LeftBound56943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60886

namespace LeftBound60887
def owner : Owner := ⟨.program ⟨257⟩, ⟨71507⟩⟩
def transferEvent : Nat := 60887
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60883 .summary, .result 56950 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60883 .summary)
      LeftBound60882.bound (LeftBound60882.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70807⟩⟩) (rawTerms := some (Proof.Events237.exact60883RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 56950 .summary)
      LeftBound56945.bound (LeftBound56945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71505⟩⟩) (rawTerms := some (Proof.Events222.exact56950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60882.bound, LeftBound56945.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60882.bound, LeftBound56945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60882.actual selector witness, LeftBound56945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60887

namespace LeftBound60893
def owner : Owner := ⟨.program ⟨257⟩, ⟨7403⟩⟩
def transferEvent : Nat := 60893
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 60891 .coefficient) (.predecessor 1 60892 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60891 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60892 .coefficient)
      LeftAuthority16066.bound (LeftAuthority16066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact16067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16066.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16066.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16066.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60893

namespace LeftBound60898
def owner : Owner := ⟨.program ⟨257⟩, ⟨11217⟩⟩
def transferEvent : Nat := 60898
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60896 .coefficient, .predecessor 1 60897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60896 .coefficient)
      LeftBound60893.bound (LeftBound60893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60897 .coefficient)
      LeftBound46651.bound (LeftBound46651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46651.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60893.bound, LeftBound46651.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60893.bound, LeftBound46651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60893.actual selector witness, LeftBound46651.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60898

namespace LeftBound60902
def owner : Owner := ⟨.program ⟨257⟩, ⟨11218⟩⟩
def transferEvent : Nat := 60902
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60900 .coefficient, .predecessor 1 60901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60900 .coefficient)
      LeftBound60898.bound (LeftBound60898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60901 .coefficient)
      LeftAuthority60889.bound (LeftAuthority60889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60898.bound, LeftAuthority60889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60898.bound, LeftAuthority60889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60898.actual selector witness, LeftAuthority60889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60902

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
