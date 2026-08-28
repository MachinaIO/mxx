import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1867
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1869
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1870
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1893

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound280251
def owner : Owner := ⟨.program ⟨257⟩, ⟨69517⟩⟩
def transferEvent : Nat := 280251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280249 .coefficient, .predecessor 1 280250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280249 .coefficient)
      LeftBound280246.bound (LeftBound280246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280250 .coefficient)
      LeftBound276742.bound (LeftBound276742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1081.exact276749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound276742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound276742.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280246.bound, LeftBound276742.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280246.bound, LeftBound276742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280246.actual selector witness, LeftBound276742.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280251

namespace LeftBound280252
def owner : Owner := ⟨.program ⟨257⟩, ⟨69517⟩⟩
def transferEvent : Nat := 280252
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280248 .summary, .result 276749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280248 .summary)
      LeftBound280247.bound (LeftBound280247.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69516⟩⟩) (rawTerms := some (Proof.Events1094.exact280248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 276749 .summary)
      LeftBound276744.bound (LeftBound276744.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47140⟩⟩) (rawTerms := some (Proof.Events1081.exact276749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound276744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280247.bound, LeftBound276744.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280247.bound, LeftBound276744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280247.actual selector witness, LeftBound276744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280252

namespace LeftBound280256
def owner : Owner := ⟨.program ⟨257⟩, ⟨69518⟩⟩
def transferEvent : Nat := 280256
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280254 .coefficient, .predecessor 1 280255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280254 .coefficient)
      LeftBound280251.bound (LeftBound280251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280255 .coefficient)
      LeftBound276530.bound (LeftBound276530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1080.exact276537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound276530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound276530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280251.bound, LeftBound276530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280251.bound, LeftBound276530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280251.actual selector witness, LeftBound276530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280256

namespace LeftBound280257
def owner : Owner := ⟨.program ⟨257⟩, ⟨69518⟩⟩
def transferEvent : Nat := 280257
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280253 .summary, .result 276537 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280253 .summary)
      LeftBound280252.bound (LeftBound280252.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69517⟩⟩) (rawTerms := some (Proof.Events1094.exact280253RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 276537 .summary)
      LeftBound276532.bound (LeftBound276532.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49820⟩⟩) (rawTerms := some (Proof.Events1080.exact276537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound276532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280252.bound, LeftBound276532.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280252.bound, LeftBound276532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280252.actual selector witness, LeftBound276532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280257

namespace LeftBound280261
def owner : Owner := ⟨.program ⟨257⟩, ⟨70985⟩⟩
def transferEvent : Nat := 280261
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280259 .coefficient, .predecessor 1 280260 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280259 .coefficient)
      LeftBound280256.bound (LeftBound280256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280260 .coefficient)
      LeftBound276318.bound (LeftBound276318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1079.exact276325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound276318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound276318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280256.bound, LeftBound276318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280256.bound, LeftBound276318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280256.actual selector witness, LeftBound276318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280261

namespace LeftBound280262
def owner : Owner := ⟨.program ⟨257⟩, ⟨70985⟩⟩
def transferEvent : Nat := 280262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280258 .summary, .result 276325 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280258 .summary)
      LeftBound280257.bound (LeftBound280257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69518⟩⟩) (rawTerms := some (Proof.Events1094.exact280258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 276325 .summary)
      LeftBound276320.bound (LeftBound276320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70983⟩⟩) (rawTerms := some (Proof.Events1079.exact276325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound276320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280257.bound, LeftBound276320.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280257.bound, LeftBound276320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280257.actual selector witness, LeftBound276320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280262

namespace LeftBound280268
def owner : Owner := ⟨.program ⟨257⟩, ⟨7418⟩⟩
def transferEvent : Nat := 280268
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 280266 .coefficient) (.predecessor 1 280267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280266 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280267 .coefficient)
      LeftAuthority16666.bound (LeftAuthority16666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16666.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280268

namespace LeftBound280273
def owner : Owner := ⟨.program ⟨257⟩, ⟨9239⟩⟩
def transferEvent : Nat := 280273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280271 .coefficient, .predecessor 1 280272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280271 .coefficient)
      LeftBound280268.bound (LeftBound280268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280272 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280268.bound, LeftBound266026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280268.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280268.actual selector witness, LeftBound266026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280273

namespace LeftBound280277
def owner : Owner := ⟨.program ⟨257⟩, ⟨9240⟩⟩
def transferEvent : Nat := 280277
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280275 .coefficient, .predecessor 1 280276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280275 .coefficient)
      LeftBound280273.bound (LeftBound280273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280276 .coefficient)
      LeftAuthority280264.bound (LeftAuthority280264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280273.bound, LeftAuthority280264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280273.bound, LeftAuthority280264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280273.actual selector witness, LeftAuthority280264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280277

namespace LeftBound280278
def owner : Owner := ⟨.program ⟨257⟩, ⟨9240⟩⟩
def transferEvent : Nat := 280278
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨55⟩⟩]⟩ [⟨.result 280265 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280265 .coefficient)
      LeftAuthority280264.bound (LeftAuthority280264.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨55⟩⟩) (rawTerms := some (Proof.Events1094.exact280265RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280264.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority280264.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority280264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority280264.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound280278

namespace LeftBound280283
def owner : Owner := ⟨.program ⟨257⟩, ⟨9630⟩⟩
def transferEvent : Nat := 280283
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 280281 .coefficient) (.predecessor 1 280282 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280281 .coefficient)
      LeftBound280277.bound (LeftBound280277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280282 .coefficient)
      LeftBound15983.bound (LeftBound15983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280277.bound LeftBound15983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280277.bound, LeftBound15983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280277.actual selector witness) * (LeftBound15983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280283

namespace LeftBound280284
def owner : Owner := ⟨.program ⟨257⟩, ⟨9630⟩⟩
def transferEvent : Nat := 280284
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩ [⟨.result 15980 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15980 .coefficient)
      LeftAuthority15979.bound (LeftAuthority15979.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9583⟩⟩) (rawTerms := some (Proof.Events062.exact15980RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15979.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15979.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15979.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound280284

namespace LeftBound280285
def owner : Owner := ⟨.program ⟨257⟩, ⟨9630⟩⟩
def transferEvent : Nat := 280285
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280280 .summary) (.transfer 280284) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280280 .summary)
      LeftBound280278.bound (LeftBound280278.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9240⟩⟩) (rawTerms := some (Proof.Events1094.exact280280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 280284)
      LeftBound280284.bound (LeftBound280284.actual selector witness) := by
  exact .transfer (LeftBound280284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280278.bound LeftBound280284.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280278.bound, LeftBound280284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280278.actual selector witness) * (LeftBound280284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280285

namespace LeftBound280311
def owner : Owner := ⟨.program ⟨257⟩, ⟨70986⟩⟩
def transferEvent : Nat := 280311
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 280309 .coefficient, .predecessor 1 280310 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280309 .coefficient)
      LeftBound280283.bound (LeftBound280283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280310 .coefficient)
      LeftBound280261.bound (LeftBound280261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1094.exact280263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280261.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280283.bound, LeftBound280261.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280283.bound, LeftBound280261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280283.actual selector witness, LeftBound280261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280311

namespace LeftBound280331
def owner : Owner := ⟨.program ⟨257⟩, ⟨70986⟩⟩
def transferEvent : Nat := 280331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 280308 .summary, .result 280263 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280308 .summary)
      LeftBound280285.bound (LeftBound280285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9630⟩⟩) (rawTerms := some (Proof.Events1094.exact280308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280263 .summary)
      LeftBound280262.bound (LeftBound280262.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70985⟩⟩) (rawTerms := some (Proof.Events1094.exact280263RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound280285.bound, LeftBound280262.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280285.bound, LeftBound280262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound280285.actual selector witness, LeftBound280262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound280331

namespace LeftBound280335
def owner : Owner := ⟨.program ⟨257⟩, ⟨70987⟩⟩
def transferEvent : Nat := 280335
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 280333 .coefficient) (.predecessor 1 280334 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 280333 .coefficient)
      LeftBound280311.bound (LeftBound280311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 280334 .coefficient)
      LeftBound16663.bound (LeftBound16663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound280311.bound LeftBound16663.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280311.bound, LeftBound16663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound280311.actual selector witness) * (LeftBound16663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound280335

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
