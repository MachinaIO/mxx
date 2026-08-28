import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1573
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1575
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1577
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1578
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1579
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1580
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1583
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1584
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1587

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound236307
def owner : Owner := ⟨.program ⟨257⟩, ⟨23839⟩⟩
def transferEvent : Nat := 236307
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236303 .summary, .result 235842 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236303 .summary)
      LeftBound236302.bound (LeftBound236302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20619⟩⟩) (rawTerms := some (Proof.Events923.exact236303RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235842 .summary)
      LeftBound235837.bound (LeftBound235837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23838⟩⟩) (rawTerms := some (Proof.Events921.exact235842RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound235837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236302.bound, LeftBound235837.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236302.bound, LeftBound235837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236302.actual selector witness, LeftBound235837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236307

namespace LeftBound236311
def owner : Owner := ⟨.program ⟨257⟩, ⟨33859⟩⟩
def transferEvent : Nat := 236311
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236309 .coefficient, .predecessor 1 236310 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236309 .coefficient)
      LeftBound236306.bound (LeftBound236306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236310 .coefficient)
      LeftBound235623.bound (LeftBound235623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236306.bound, LeftBound235623.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236306.bound, LeftBound235623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236306.actual selector witness, LeftBound235623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236311

namespace LeftBound236312
def owner : Owner := ⟨.program ⟨257⟩, ⟨33859⟩⟩
def transferEvent : Nat := 236312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236308 .summary, .result 235630 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236308 .summary)
      LeftBound236307.bound (LeftBound236307.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23839⟩⟩) (rawTerms := some (Proof.Events923.exact236308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235630 .summary)
      LeftBound235625.bound (LeftBound235625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33858⟩⟩) (rawTerms := some (Proof.Events920.exact235630RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound235625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236307.bound, LeftBound235625.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236307.bound, LeftBound235625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236307.actual selector witness, LeftBound235625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236312

namespace LeftBound236316
def owner : Owner := ⟨.program ⟨257⟩, ⟨52919⟩⟩
def transferEvent : Nat := 236316
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236314 .coefficient, .predecessor 1 236315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236314 .coefficient)
      LeftBound236311.bound (LeftBound236311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236315 .coefficient)
      LeftBound235411.bound (LeftBound235411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events919.exact235418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236311.bound, LeftBound235411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236311.bound, LeftBound235411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236311.actual selector witness, LeftBound235411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236316

namespace LeftBound236317
def owner : Owner := ⟨.program ⟨257⟩, ⟨52919⟩⟩
def transferEvent : Nat := 236317
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236313 .summary, .result 235418 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236313 .summary)
      LeftBound236312.bound (LeftBound236312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33859⟩⟩) (rawTerms := some (Proof.Events923.exact236313RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235418 .summary)
      LeftBound235413.bound (LeftBound235413.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52918⟩⟩) (rawTerms := some (Proof.Events919.exact235418RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound235413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236312.bound, LeftBound235413.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236312.bound, LeftBound235413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236312.actual selector witness, LeftBound235413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236317

namespace LeftBound236321
def owner : Owner := ⟨.program ⟨257⟩, ⟨55899⟩⟩
def transferEvent : Nat := 236321
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236319 .coefficient, .predecessor 1 236320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236319 .coefficient)
      LeftBound236316.bound (LeftBound236316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236320 .coefficient)
      LeftBound235199.bound (LeftBound235199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events918.exact235206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235199.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236316.bound, LeftBound235199.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236316.bound, LeftBound235199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236316.actual selector witness, LeftBound235199.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236321

namespace LeftBound236322
def owner : Owner := ⟨.program ⟨257⟩, ⟨55899⟩⟩
def transferEvent : Nat := 236322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236318 .summary, .result 235206 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236318 .summary)
      LeftBound236317.bound (LeftBound236317.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52919⟩⟩) (rawTerms := some (Proof.Events923.exact236318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235206 .summary)
      LeftBound235201.bound (LeftBound235201.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55898⟩⟩) (rawTerms := some (Proof.Events918.exact235206RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound235201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236317.bound, LeftBound235201.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236317.bound, LeftBound235201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236317.actual selector witness, LeftBound235201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236322

namespace LeftBound236326
def owner : Owner := ⟨.program ⟨257⟩, ⟨58879⟩⟩
def transferEvent : Nat := 236326
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236324 .coefficient, .predecessor 1 236325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236324 .coefficient)
      LeftBound236321.bound (LeftBound236321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236325 .coefficient)
      LeftBound234987.bound (LeftBound234987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events917.exact234994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound234987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound234987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236321.bound, LeftBound234987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236321.bound, LeftBound234987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236321.actual selector witness, LeftBound234987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236326

namespace LeftBound236327
def owner : Owner := ⟨.program ⟨257⟩, ⟨58879⟩⟩
def transferEvent : Nat := 236327
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236323 .summary, .result 234994 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236323 .summary)
      LeftBound236322.bound (LeftBound236322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55899⟩⟩) (rawTerms := some (Proof.Events923.exact236323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 234994 .summary)
      LeftBound234989.bound (LeftBound234989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58878⟩⟩) (rawTerms := some (Proof.Events917.exact234994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound234989.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236322.bound, LeftBound234989.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236322.bound, LeftBound234989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236322.actual selector witness, LeftBound234989.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236327

namespace LeftBound236331
def owner : Owner := ⟨.program ⟨257⟩, ⟨61859⟩⟩
def transferEvent : Nat := 236331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236329 .coefficient, .predecessor 1 236330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236329 .coefficient)
      LeftBound236326.bound (LeftBound236326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236330 .coefficient)
      LeftBound234775.bound (LeftBound234775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events917.exact234782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound234775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound234775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236326.bound, LeftBound234775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236326.bound, LeftBound234775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236326.actual selector witness, LeftBound234775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236331

namespace LeftBound236332
def owner : Owner := ⟨.program ⟨257⟩, ⟨61859⟩⟩
def transferEvent : Nat := 236332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236328 .summary, .result 234782 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236328 .summary)
      LeftBound236327.bound (LeftBound236327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58879⟩⟩) (rawTerms := some (Proof.Events923.exact236328RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 234782 .summary)
      LeftBound234777.bound (LeftBound234777.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61858⟩⟩) (rawTerms := some (Proof.Events917.exact234782RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound234777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236327.bound, LeftBound234777.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236327.bound, LeftBound234777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236327.actual selector witness, LeftBound234777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236332

namespace LeftBound236336
def owner : Owner := ⟨.program ⟨257⟩, ⟨64839⟩⟩
def transferEvent : Nat := 236336
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236334 .coefficient, .predecessor 1 236335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236334 .coefficient)
      LeftBound236331.bound (LeftBound236331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236335 .coefficient)
      LeftBound234563.bound (LeftBound234563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events916.exact234570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound234563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound234563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236331.bound, LeftBound234563.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236331.bound, LeftBound234563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236331.actual selector witness, LeftBound234563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236336

namespace LeftBound236337
def owner : Owner := ⟨.program ⟨257⟩, ⟨64839⟩⟩
def transferEvent : Nat := 236337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236333 .summary, .result 234570 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236333 .summary)
      LeftBound236332.bound (LeftBound236332.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61859⟩⟩) (rawTerms := some (Proof.Events923.exact236333RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 234570 .summary)
      LeftBound234565.bound (LeftBound234565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64838⟩⟩) (rawTerms := some (Proof.Events916.exact234570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound234565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236332.bound, LeftBound234565.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236332.bound, LeftBound234565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236332.actual selector witness, LeftBound234565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236337

namespace LeftBound236341
def owner : Owner := ⟨.program ⟨257⟩, ⟨70088⟩⟩
def transferEvent : Nat := 236341
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236339 .coefficient, .predecessor 1 236340 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236339 .coefficient)
      LeftBound236336.bound (LeftBound236336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236340 .coefficient)
      LeftBound234351.bound (LeftBound234351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events915.exact234358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound234351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound234351.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236336.bound, LeftBound234351.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236336.bound, LeftBound234351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236336.actual selector witness, LeftBound234351.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236341

namespace LeftBound236342
def owner : Owner := ⟨.program ⟨257⟩, ⟨70088⟩⟩
def transferEvent : Nat := 236342
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 236338 .summary, .result 234358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236338 .summary)
      LeftBound236337.bound (LeftBound236337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64839⟩⟩) (rawTerms := some (Proof.Events923.exact236338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 234358 .summary)
      LeftBound234353.bound (LeftBound234353.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70087⟩⟩) (rawTerms := some (Proof.Events915.exact234358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound234353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236337.bound, LeftBound234353.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236337.bound, LeftBound234353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236337.actual selector witness, LeftBound234353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236342

namespace LeftBound236346
def owner : Owner := ⟨.program ⟨257⟩, ⟨70089⟩⟩
def transferEvent : Nat := 236346
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 236344 .coefficient, .predecessor 1 236345 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 236344 .coefficient)
      LeftBound236341.bound (LeftBound236341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events923.exact236343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 236345 .coefficient)
      LeftBound234139.bound (LeftBound234139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events914.exact234146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound234139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound234139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound236341.bound, LeftBound234139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236341.bound, LeftBound234139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound236341.actual selector witness, LeftBound234139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound236346

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
