import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard619
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard623
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard626
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard630
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard634
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard637
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard644

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99191
def owner : Owner := ⟨.program ⟨257⟩, ⟨17904⟩⟩
def transferEvent : Nat := 99191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99189 .coefficient, .predecessor 1 99190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99189 .coefficient)
      LeftBound99020.bound (LeftBound99020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99190 .coefficient)
      LeftBound99003.bound (LeftBound99003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact99010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99020.bound, LeftBound99003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99020.bound, LeftBound99003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99020.actual selector witness, LeftBound99003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99191

namespace LeftBound99194
def owner : Owner := ⟨.program ⟨257⟩, ⟨17904⟩⟩
def transferEvent : Nat := 99194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99188 .summary, .result 99010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99188 .summary)
      LeftBound99022.bound (LeftBound99022.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16699⟩⟩) (rawTerms := some (Proof.Events387.exact99188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99010 .summary)
      LeftBound99005.bound (LeftBound99005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17903⟩⟩) (rawTerms := some (Proof.Events386.exact99010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99022.bound, LeftBound99005.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99022.bound, LeftBound99005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99022.actual selector witness, LeftBound99005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99194

namespace LeftBound99198
def owner : Owner := ⟨.program ⟨257⟩, ⟨20811⟩⟩
def transferEvent : Nat := 99198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99196 .coefficient, .predecessor 1 99197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99196 .coefficient)
      LeftBound99191.bound (LeftBound99191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99197 .coefficient)
      LeftBound98709.bound (LeftBound98709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99191.bound, LeftBound98709.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99191.bound, LeftBound98709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99191.actual selector witness, LeftBound98709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99198

namespace LeftBound99199
def owner : Owner := ⟨.program ⟨257⟩, ⟨20811⟩⟩
def transferEvent : Nat := 99199
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99195 .summary, .result 98713 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99195 .summary)
      LeftBound99194.bound (LeftBound99194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17904⟩⟩) (rawTerms := some (Proof.Events387.exact99195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98713 .summary)
      LeftBound98712.bound (LeftBound98712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20810⟩⟩) (rawTerms := some (Proof.Events385.exact98713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99194.bound, LeftBound98712.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99194.bound, LeftBound98712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99194.actual selector witness, LeftBound98712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99199

namespace LeftBound99203
def owner : Owner := ⟨.program ⟨257⟩, ⟨24031⟩⟩
def transferEvent : Nat := 99203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99201 .coefficient, .predecessor 1 99202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99201 .coefficient)
      LeftBound99198.bound (LeftBound99198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99202 .coefficient)
      LeftBound98227.bound (LeftBound98227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99198.bound, LeftBound98227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99198.bound, LeftBound98227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99198.actual selector witness, LeftBound98227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99203

namespace LeftBound99204
def owner : Owner := ⟨.program ⟨257⟩, ⟨24031⟩⟩
def transferEvent : Nat := 99204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99200 .summary, .result 98231 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99200 .summary)
      LeftBound99199.bound (LeftBound99199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20811⟩⟩) (rawTerms := some (Proof.Events387.exact99200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98231 .summary)
      LeftBound98230.bound (LeftBound98230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24030⟩⟩) (rawTerms := some (Proof.Events383.exact98231RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99199.bound, LeftBound98230.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99199.bound, LeftBound98230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99199.actual selector witness, LeftBound98230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99204

namespace LeftBound99208
def owner : Owner := ⟨.program ⟨257⟩, ⟨34051⟩⟩
def transferEvent : Nat := 99208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99206 .coefficient, .predecessor 1 99207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99206 .coefficient)
      LeftBound99203.bound (LeftBound99203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99207 .coefficient)
      LeftBound97745.bound (LeftBound97745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99203.bound, LeftBound97745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99203.bound, LeftBound97745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99203.actual selector witness, LeftBound97745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99208

namespace LeftBound99209
def owner : Owner := ⟨.program ⟨257⟩, ⟨34051⟩⟩
def transferEvent : Nat := 99209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99205 .summary, .result 97749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99205 .summary)
      LeftBound99204.bound (LeftBound99204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24031⟩⟩) (rawTerms := some (Proof.Events387.exact99205RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 97749 .summary)
      LeftBound97748.bound (LeftBound97748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34050⟩⟩) (rawTerms := some (Proof.Events381.exact97749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99204.bound, LeftBound97748.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99204.bound, LeftBound97748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99204.actual selector witness, LeftBound97748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99209

namespace LeftBound99213
def owner : Owner := ⟨.program ⟨257⟩, ⟨53111⟩⟩
def transferEvent : Nat := 99213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99211 .coefficient, .predecessor 1 99212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99211 .coefficient)
      LeftBound99208.bound (LeftBound99208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99212 .coefficient)
      LeftBound97263.bound (LeftBound97263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99208.bound, LeftBound97263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99208.bound, LeftBound97263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99208.actual selector witness, LeftBound97263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99213

namespace LeftBound99214
def owner : Owner := ⟨.program ⟨257⟩, ⟨53111⟩⟩
def transferEvent : Nat := 99214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99210 .summary, .result 97267 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99210 .summary)
      LeftBound99209.bound (LeftBound99209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34051⟩⟩) (rawTerms := some (Proof.Events387.exact99210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 97267 .summary)
      LeftBound97266.bound (LeftBound97266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53110⟩⟩) (rawTerms := some (Proof.Events379.exact97267RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99209.bound, LeftBound97266.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99209.bound, LeftBound97266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99209.actual selector witness, LeftBound97266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99214

namespace LeftBound99218
def owner : Owner := ⟨.program ⟨257⟩, ⟨56091⟩⟩
def transferEvent : Nat := 99218
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99216 .coefficient, .predecessor 1 99217 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99216 .coefficient)
      LeftBound99213.bound (LeftBound99213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99217 .coefficient)
      LeftBound96781.bound (LeftBound96781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99213.bound, LeftBound96781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99213.bound, LeftBound96781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99213.actual selector witness, LeftBound96781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99218

namespace LeftBound99219
def owner : Owner := ⟨.program ⟨257⟩, ⟨56091⟩⟩
def transferEvent : Nat := 99219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99215 .summary, .result 96785 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99215 .summary)
      LeftBound99214.bound (LeftBound99214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53111⟩⟩) (rawTerms := some (Proof.Events387.exact99215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96785 .summary)
      LeftBound96784.bound (LeftBound96784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56090⟩⟩) (rawTerms := some (Proof.Events378.exact96785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99214.bound, LeftBound96784.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99214.bound, LeftBound96784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99214.actual selector witness, LeftBound96784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99219

namespace LeftBound99223
def owner : Owner := ⟨.program ⟨257⟩, ⟨59071⟩⟩
def transferEvent : Nat := 99223
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99221 .coefficient, .predecessor 1 99222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99221 .coefficient)
      LeftBound99218.bound (LeftBound99218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99222 .coefficient)
      LeftBound96299.bound (LeftBound96299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99218.bound, LeftBound96299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99218.bound, LeftBound96299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99218.actual selector witness, LeftBound96299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99223

namespace LeftBound99224
def owner : Owner := ⟨.program ⟨257⟩, ⟨59071⟩⟩
def transferEvent : Nat := 99224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99220 .summary, .result 96303 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99220 .summary)
      LeftBound99219.bound (LeftBound99219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56091⟩⟩) (rawTerms := some (Proof.Events387.exact99220RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96303 .summary)
      LeftBound96302.bound (LeftBound96302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59070⟩⟩) (rawTerms := some (Proof.Events376.exact96303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99219.bound, LeftBound96302.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99219.bound, LeftBound96302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99219.actual selector witness, LeftBound96302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99224

namespace LeftBound99228
def owner : Owner := ⟨.program ⟨257⟩, ⟨62051⟩⟩
def transferEvent : Nat := 99228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99226 .coefficient, .predecessor 1 99227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99226 .coefficient)
      LeftBound99223.bound (LeftBound99223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99227 .coefficient)
      LeftBound95817.bound (LeftBound95817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99223.bound, LeftBound95817.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99223.bound, LeftBound95817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99223.actual selector witness, LeftBound95817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99228

namespace LeftBound99229
def owner : Owner := ⟨.program ⟨257⟩, ⟨62051⟩⟩
def transferEvent : Nat := 99229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99225 .summary, .result 95821 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99225 .summary)
      LeftBound99224.bound (LeftBound99224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59071⟩⟩) (rawTerms := some (Proof.Events387.exact99225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 95821 .summary)
      LeftBound95820.bound (LeftBound95820.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62050⟩⟩) (rawTerms := some (Proof.Events374.exact95821RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95820.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99224.bound, LeftBound95820.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99224.bound, LeftBound95820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99224.actual selector witness, LeftBound95820.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99229

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
