import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard767
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard769
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard770
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard771
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard772
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard773
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard774
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard775

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound119284
def owner : Owner := ⟨.program ⟨257⟩, ⟨9402⟩⟩
def transferEvent : Nat := 119284
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩ [⟨.result 31516 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31516 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨118⟩⟩) (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound31515.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound31515.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound119284

namespace LeftBound119289
def owner : Owner := ⟨.program ⟨257⟩, ⟨9480⟩⟩
def transferEvent : Nat := 119289
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119287 .coefficient, .predecessor 1 119288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119287 .coefficient)
      LeftBound119283.bound (LeftBound119283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events465.exact119286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119288 .coefficient)
      LeftBound119283.bound (LeftBound119283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events465.exact119286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119283.bound, LeftBound119283.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119283.bound, LeftBound119283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119283.actual selector witness, LeftBound119283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119289

namespace LeftBound119292
def owner : Owner := ⟨.program ⟨257⟩, ⟨9480⟩⟩
def transferEvent : Nat := 119292
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119286 .summary, .result 119286 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119286 .summary)
      LeftBound119284.bound (LeftBound119284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9402⟩⟩) (rawTerms := some (Proof.Events465.exact119286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119286 .summary)
      LeftBound119284.bound (LeftBound119284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9402⟩⟩) (rawTerms := some (Proof.Events465.exact119286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119284.bound, LeftBound119284.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119284.bound, LeftBound119284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119284.actual selector witness, LeftBound119284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119292

namespace LeftBound119296
def owner : Owner := ⟨.program ⟨257⟩, ⟨17787⟩⟩
def transferEvent : Nat := 119296
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119294 .coefficient, .predecessor 1 119295 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119294 .coefficient)
      LeftBound119289.bound (LeftBound119289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events465.exact119293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119295 .coefficient)
      LeftBound119259.bound (LeftBound119259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events465.exact119266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119289.bound, LeftBound119259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119289.bound, LeftBound119259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119289.actual selector witness, LeftBound119259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119296

namespace LeftBound119297
def owner : Owner := ⟨.program ⟨257⟩, ⟨17787⟩⟩
def transferEvent : Nat := 119297
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119293 .summary, .result 119266 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119293 .summary)
      LeftBound119292.bound (LeftBound119292.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9480⟩⟩) (rawTerms := some (Proof.Events465.exact119293RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119266 .summary)
      LeftBound119261.bound (LeftBound119261.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17786⟩⟩) (rawTerms := some (Proof.Events465.exact119266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119292.bound, LeftBound119261.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119292.bound, LeftBound119261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119292.actual selector witness, LeftBound119261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119297

namespace LeftBound119301
def owner : Owner := ⟨.program ⟨257⟩, ⟨20681⟩⟩
def transferEvent : Nat := 119301
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119299 .coefficient, .predecessor 1 119300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119299 .coefficient)
      LeftBound119296.bound (LeftBound119296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119300 .coefficient)
      LeftBound119047.bound (LeftBound119047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events465.exact119054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119296.bound, LeftBound119047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119296.bound, LeftBound119047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119296.actual selector witness, LeftBound119047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119301

namespace LeftBound119302
def owner : Owner := ⟨.program ⟨257⟩, ⟨20681⟩⟩
def transferEvent : Nat := 119302
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119298 .summary, .result 119054 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119298 .summary)
      LeftBound119297.bound (LeftBound119297.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17787⟩⟩) (rawTerms := some (Proof.Events466.exact119298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119054 .summary)
      LeftBound119049.bound (LeftBound119049.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20680⟩⟩) (rawTerms := some (Proof.Events465.exact119054RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119297.bound, LeftBound119049.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119297.bound, LeftBound119049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119297.actual selector witness, LeftBound119049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119302

namespace LeftBound119306
def owner : Owner := ⟨.program ⟨257⟩, ⟨23901⟩⟩
def transferEvent : Nat := 119306
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119304 .coefficient, .predecessor 1 119305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119304 .coefficient)
      LeftBound119301.bound (LeftBound119301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119305 .coefficient)
      LeftBound118835.bound (LeftBound118835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events464.exact118842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound118835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound118835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119301.bound, LeftBound118835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119301.bound, LeftBound118835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119301.actual selector witness, LeftBound118835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119306

namespace LeftBound119307
def owner : Owner := ⟨.program ⟨257⟩, ⟨23901⟩⟩
def transferEvent : Nat := 119307
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119303 .summary, .result 118842 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119303 .summary)
      LeftBound119302.bound (LeftBound119302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20681⟩⟩) (rawTerms := some (Proof.Events466.exact119303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 118842 .summary)
      LeftBound118837.bound (LeftBound118837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23900⟩⟩) (rawTerms := some (Proof.Events464.exact118842RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound118837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119302.bound, LeftBound118837.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119302.bound, LeftBound118837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119302.actual selector witness, LeftBound118837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119307

namespace LeftBound119311
def owner : Owner := ⟨.program ⟨257⟩, ⟨33921⟩⟩
def transferEvent : Nat := 119311
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119309 .coefficient, .predecessor 1 119310 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119309 .coefficient)
      LeftBound119306.bound (LeftBound119306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119310 .coefficient)
      LeftBound118623.bound (LeftBound118623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events463.exact118630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound118623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound118623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119306.bound, LeftBound118623.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119306.bound, LeftBound118623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119306.actual selector witness, LeftBound118623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119311

namespace LeftBound119312
def owner : Owner := ⟨.program ⟨257⟩, ⟨33921⟩⟩
def transferEvent : Nat := 119312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119308 .summary, .result 118630 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119308 .summary)
      LeftBound119307.bound (LeftBound119307.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23901⟩⟩) (rawTerms := some (Proof.Events466.exact119308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 118630 .summary)
      LeftBound118625.bound (LeftBound118625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33920⟩⟩) (rawTerms := some (Proof.Events463.exact118630RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound118625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119307.bound, LeftBound118625.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119307.bound, LeftBound118625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119307.actual selector witness, LeftBound118625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119312

namespace LeftBound119316
def owner : Owner := ⟨.program ⟨257⟩, ⟨52981⟩⟩
def transferEvent : Nat := 119316
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119314 .coefficient, .predecessor 1 119315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119314 .coefficient)
      LeftBound119311.bound (LeftBound119311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119315 .coefficient)
      LeftBound118411.bound (LeftBound118411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events462.exact118418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound118411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound118411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119311.bound, LeftBound118411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119311.bound, LeftBound118411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119311.actual selector witness, LeftBound118411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119316

namespace LeftBound119317
def owner : Owner := ⟨.program ⟨257⟩, ⟨52981⟩⟩
def transferEvent : Nat := 119317
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119313 .summary, .result 118418 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119313 .summary)
      LeftBound119312.bound (LeftBound119312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33921⟩⟩) (rawTerms := some (Proof.Events466.exact119313RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 118418 .summary)
      LeftBound118413.bound (LeftBound118413.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52980⟩⟩) (rawTerms := some (Proof.Events462.exact118418RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound118413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119312.bound, LeftBound118413.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119312.bound, LeftBound118413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119312.actual selector witness, LeftBound118413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119317

namespace LeftBound119321
def owner : Owner := ⟨.program ⟨257⟩, ⟨55961⟩⟩
def transferEvent : Nat := 119321
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119319 .coefficient, .predecessor 1 119320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119319 .coefficient)
      LeftBound119316.bound (LeftBound119316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119320 .coefficient)
      LeftBound118199.bound (LeftBound118199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events461.exact118206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound118199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound118199.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119316.bound, LeftBound118199.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119316.bound, LeftBound118199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119316.actual selector witness, LeftBound118199.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119321

namespace LeftBound119322
def owner : Owner := ⟨.program ⟨257⟩, ⟨55961⟩⟩
def transferEvent : Nat := 119322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119318 .summary, .result 118206 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119318 .summary)
      LeftBound119317.bound (LeftBound119317.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52981⟩⟩) (rawTerms := some (Proof.Events466.exact119318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 118206 .summary)
      LeftBound118201.bound (LeftBound118201.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55960⟩⟩) (rawTerms := some (Proof.Events461.exact118206RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound118201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119317.bound, LeftBound118201.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119317.bound, LeftBound118201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119317.actual selector witness, LeftBound118201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119322

namespace LeftBound119326
def owner : Owner := ⟨.program ⟨257⟩, ⟨58941⟩⟩
def transferEvent : Nat := 119326
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119324 .coefficient, .predecessor 1 119325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119324 .coefficient)
      LeftBound119321.bound (LeftBound119321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119325 .coefficient)
      LeftBound117987.bound (LeftBound117987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events460.exact117994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound117987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound117987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119321.bound, LeftBound117987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119321.bound, LeftBound117987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119321.actual selector witness, LeftBound117987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119326

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
