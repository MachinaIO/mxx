import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1192
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1195
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1199
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1202
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1206
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1210
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1213
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1254

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound186998
def owner : Owner := ⟨.program ⟨257⟩, ⟨70420⟩⟩
def transferEvent : Nat := 186998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186996 .coefficient, .predecessor 1 186997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186996 .coefficient)
      LeftBound186993.bound (LeftBound186993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186997 .coefficient)
      LeftBound181639.bound (LeftBound181639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events709.exact181643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186993.bound, LeftBound181639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186993.bound, LeftBound181639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186993.actual selector witness, LeftBound181639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186998

namespace LeftBound186999
def owner : Owner := ⟨.program ⟨257⟩, ⟨70420⟩⟩
def transferEvent : Nat := 186999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186995 .summary, .result 181643 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186995 .summary)
      LeftBound186994.bound (LeftBound186994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70419⟩⟩) (rawTerms := some (Proof.Events730.exact186995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181643 .summary)
      LeftBound181642.bound (LeftBound181642.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31047⟩⟩) (rawTerms := some (Proof.Events709.exact181643RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound181642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186994.bound, LeftBound181642.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186994.bound, LeftBound181642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186994.actual selector witness, LeftBound181642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186999

namespace LeftBound187003
def owner : Owner := ⟨.program ⟨257⟩, ⟨70421⟩⟩
def transferEvent : Nat := 187003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187001 .coefficient, .predecessor 1 187002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187001 .coefficient)
      LeftBound186998.bound (LeftBound186998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187002 .coefficient)
      LeftBound181157.bound (LeftBound181157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events707.exact181161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186998.bound, LeftBound181157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186998.bound, LeftBound181157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186998.actual selector witness, LeftBound181157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187003

namespace LeftBound187004
def owner : Owner := ⟨.program ⟨257⟩, ⟨70421⟩⟩
def transferEvent : Nat := 187004
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187000 .summary, .result 181161 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187000 .summary)
      LeftBound186999.bound (LeftBound186999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70420⟩⟩) (rawTerms := some (Proof.Events730.exact187000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181161 .summary)
      LeftBound181160.bound (LeftBound181160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36707⟩⟩) (rawTerms := some (Proof.Events707.exact181161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound181160.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186999.bound, LeftBound181160.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186999.bound, LeftBound181160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186999.actual selector witness, LeftBound181160.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187004

namespace LeftBound187008
def owner : Owner := ⟨.program ⟨257⟩, ⟨70422⟩⟩
def transferEvent : Nat := 187008
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187006 .coefficient, .predecessor 1 187007 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187006 .coefficient)
      LeftBound187003.bound (LeftBound187003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187007 .coefficient)
      LeftBound180675.bound (LeftBound180675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events705.exact180679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187003.bound, LeftBound180675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187003.bound, LeftBound180675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187003.actual selector witness, LeftBound180675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187008

namespace LeftBound187009
def owner : Owner := ⟨.program ⟨257⟩, ⟨70422⟩⟩
def transferEvent : Nat := 187009
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187005 .summary, .result 180679 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187005 .summary)
      LeftBound187004.bound (LeftBound187004.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70421⟩⟩) (rawTerms := some (Proof.Events730.exact187005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 180679 .summary)
      LeftBound180678.bound (LeftBound180678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39387⟩⟩) (rawTerms := some (Proof.Events705.exact180679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound180678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187004.bound, LeftBound180678.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187004.bound, LeftBound180678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187004.actual selector witness, LeftBound180678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187009

namespace LeftBound187013
def owner : Owner := ⟨.program ⟨257⟩, ⟨70423⟩⟩
def transferEvent : Nat := 187013
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187011 .coefficient, .predecessor 1 187012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187011 .coefficient)
      LeftBound187008.bound (LeftBound187008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187012 .coefficient)
      LeftBound180193.bound (LeftBound180193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events703.exact180197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187008.bound, LeftBound180193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187008.bound, LeftBound180193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187008.actual selector witness, LeftBound180193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187013

namespace LeftBound187014
def owner : Owner := ⟨.program ⟨257⟩, ⟨70423⟩⟩
def transferEvent : Nat := 187014
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187010 .summary, .result 180197 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187010 .summary)
      LeftBound187009.bound (LeftBound187009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70422⟩⟩) (rawTerms := some (Proof.Events730.exact187010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 180197 .summary)
      LeftBound180196.bound (LeftBound180196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42067⟩⟩) (rawTerms := some (Proof.Events703.exact180197RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound180196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187009.bound, LeftBound180196.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187009.bound, LeftBound180196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187009.actual selector witness, LeftBound180196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187014

namespace LeftBound187018
def owner : Owner := ⟨.program ⟨257⟩, ⟨70424⟩⟩
def transferEvent : Nat := 187018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187016 .coefficient, .predecessor 1 187017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187016 .coefficient)
      LeftBound187013.bound (LeftBound187013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187013.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187017 .coefficient)
      LeftBound179711.bound (LeftBound179711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events702.exact179715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187013.bound, LeftBound179711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187013.bound, LeftBound179711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187013.actual selector witness, LeftBound179711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187018

namespace LeftBound187019
def owner : Owner := ⟨.program ⟨257⟩, ⟨70424⟩⟩
def transferEvent : Nat := 187019
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187015 .summary, .result 179715 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187015 .summary)
      LeftBound187014.bound (LeftBound187014.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70423⟩⟩) (rawTerms := some (Proof.Events730.exact187015RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 179715 .summary)
      LeftBound179714.bound (LeftBound179714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44747⟩⟩) (rawTerms := some (Proof.Events702.exact179715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound179714.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187014.bound, LeftBound179714.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187014.bound, LeftBound179714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187014.actual selector witness, LeftBound179714.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187019

namespace LeftBound187023
def owner : Owner := ⟨.program ⟨257⟩, ⟨70425⟩⟩
def transferEvent : Nat := 187023
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187021 .coefficient, .predecessor 1 187022 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187021 .coefficient)
      LeftBound187018.bound (LeftBound187018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187022 .coefficient)
      LeftBound179229.bound (LeftBound179229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events700.exact179233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound179229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound179229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187018.bound, LeftBound179229.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187018.bound, LeftBound179229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187018.actual selector witness, LeftBound179229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187023

namespace LeftBound187024
def owner : Owner := ⟨.program ⟨257⟩, ⟨70425⟩⟩
def transferEvent : Nat := 187024
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187020 .summary, .result 179233 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187020 .summary)
      LeftBound187019.bound (LeftBound187019.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70424⟩⟩) (rawTerms := some (Proof.Events730.exact187020RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 179233 .summary)
      LeftBound179232.bound (LeftBound179232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47427⟩⟩) (rawTerms := some (Proof.Events700.exact179233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound179232.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187019.bound, LeftBound179232.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187019.bound, LeftBound179232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187019.actual selector witness, LeftBound179232.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187024

namespace LeftBound187028
def owner : Owner := ⟨.program ⟨257⟩, ⟨70426⟩⟩
def transferEvent : Nat := 187028
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 187026 .coefficient, .predecessor 1 187027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187026 .coefficient)
      LeftBound187023.bound (LeftBound187023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187027 .coefficient)
      LeftBound178747.bound (LeftBound178747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events698.exact178751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187023.bound, LeftBound178747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187023.bound, LeftBound178747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187023.actual selector witness, LeftBound178747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187028

namespace LeftBound187029
def owner : Owner := ⟨.program ⟨257⟩, ⟨70426⟩⟩
def transferEvent : Nat := 187029
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 187025 .summary, .result 178751 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187025 .summary)
      LeftBound187024.bound (LeftBound187024.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70425⟩⟩) (rawTerms := some (Proof.Events730.exact187025RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178751 .summary)
      LeftBound178750.bound (LeftBound178750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50107⟩⟩) (rawTerms := some (Proof.Events698.exact178751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound187024.bound, LeftBound178750.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187024.bound, LeftBound178750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound187024.actual selector witness, LeftBound178750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound187029

namespace LeftBound187033
def owner : Owner := ⟨.program ⟨257⟩, ⟨71331⟩⟩
def transferEvent : Nat := 187033
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 187031 .coefficient) (.predecessor 1 187032 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187031 .coefficient)
      LeftBound187028.bound (LeftBound187028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187032 .coefficient)
      LeftAuthority178252.bound (LeftAuthority178252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority178252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority178252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound187028.bound LeftAuthority178252.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187028.bound, LeftAuthority178252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound187028.actual selector witness) * (LeftAuthority178252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound187033

namespace LeftBound187034
def owner : Owner := ⟨.program ⟨257⟩, ⟨71331⟩⟩
def transferEvent : Nat := 187034
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩ [⟨.result 178253 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178253 .coefficient)
      LeftAuthority178252.bound (LeftAuthority178252.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71329⟩⟩) (rawTerms := some (Proof.Events696.exact178253RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority178252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority178252.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority178252.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority178252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority178252.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound187034

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
