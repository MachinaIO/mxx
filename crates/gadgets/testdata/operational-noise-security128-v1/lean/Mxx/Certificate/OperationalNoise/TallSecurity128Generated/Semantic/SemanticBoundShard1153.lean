import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1101
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1105
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1108
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1112
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1115
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1116
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1119
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1123
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1152

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound172349
def owner : Owner := ⟨.program ⟨257⟩, ⟨59040⟩⟩
def transferEvent : Nat := 172349
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172345 .summary, .result 169428 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172345 .summary)
      LeftBound172344.bound (LeftBound172344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56060⟩⟩) (rawTerms := some (Proof.Events673.exact172345RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 169428 .summary)
      LeftBound169427.bound (LeftBound169427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59039⟩⟩) (rawTerms := some (Proof.Events661.exact169428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound169427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172344.bound, LeftBound169427.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172344.bound, LeftBound169427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172344.actual selector witness, LeftBound169427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172349

namespace LeftBound172353
def owner : Owner := ⟨.program ⟨257⟩, ⟨62020⟩⟩
def transferEvent : Nat := 172353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172351 .coefficient, .predecessor 1 172352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172351 .coefficient)
      LeftBound172348.bound (LeftBound172348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172352 .coefficient)
      LeftBound168942.bound (LeftBound168942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168942.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172348.bound, LeftBound168942.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172348.bound, LeftBound168942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172348.actual selector witness, LeftBound168942.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172353

namespace LeftBound172354
def owner : Owner := ⟨.program ⟨257⟩, ⟨62020⟩⟩
def transferEvent : Nat := 172354
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172350 .summary, .result 168946 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172350 .summary)
      LeftBound172349.bound (LeftBound172349.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59040⟩⟩) (rawTerms := some (Proof.Events673.exact172350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168946 .summary)
      LeftBound168945.bound (LeftBound168945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62019⟩⟩) (rawTerms := some (Proof.Events659.exact168946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound168945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172349.bound, LeftBound168945.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172349.bound, LeftBound168945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172349.actual selector witness, LeftBound168945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172354

namespace LeftBound172358
def owner : Owner := ⟨.program ⟨257⟩, ⟨65000⟩⟩
def transferEvent : Nat := 172358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172356 .coefficient, .predecessor 1 172357 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172356 .coefficient)
      LeftBound172353.bound (LeftBound172353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172357 .coefficient)
      LeftBound168460.bound (LeftBound168460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172353.bound, LeftBound168460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172353.bound, LeftBound168460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172353.actual selector witness, LeftBound168460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172358

namespace LeftBound172359
def owner : Owner := ⟨.program ⟨257⟩, ⟨65000⟩⟩
def transferEvent : Nat := 172359
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172355 .summary, .result 168464 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172355 .summary)
      LeftBound172354.bound (LeftBound172354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62020⟩⟩) (rawTerms := some (Proof.Events673.exact172355RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168464 .summary)
      LeftBound168463.bound (LeftBound168463.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64999⟩⟩) (rawTerms := some (Proof.Events658.exact168464RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound168463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172354.bound, LeftBound168463.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172354.bound, LeftBound168463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172354.actual selector witness, LeftBound168463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172359

namespace LeftBound172363
def owner : Owner := ⟨.program ⟨257⟩, ⟨70497⟩⟩
def transferEvent : Nat := 172363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172361 .coefficient, .predecessor 1 172362 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172361 .coefficient)
      LeftBound172358.bound (LeftBound172358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172362 .coefficient)
      LeftBound167978.bound (LeftBound167978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events656.exact167982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172358.bound, LeftBound167978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172358.bound, LeftBound167978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172358.actual selector witness, LeftBound167978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172363

namespace LeftBound172364
def owner : Owner := ⟨.program ⟨257⟩, ⟨70497⟩⟩
def transferEvent : Nat := 172364
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172360 .summary, .result 167982 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172360 .summary)
      LeftBound172359.bound (LeftBound172359.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65000⟩⟩) (rawTerms := some (Proof.Events673.exact172360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167982 .summary)
      LeftBound167981.bound (LeftBound167981.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70496⟩⟩) (rawTerms := some (Proof.Events656.exact167982RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167981.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172359.bound, LeftBound167981.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172359.bound, LeftBound167981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172359.actual selector witness, LeftBound167981.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172364

namespace LeftBound172368
def owner : Owner := ⟨.program ⟨257⟩, ⟨70498⟩⟩
def transferEvent : Nat := 172368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172366 .coefficient, .predecessor 1 172367 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172366 .coefficient)
      LeftBound172363.bound (LeftBound172363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172367 .coefficient)
      LeftBound167496.bound (LeftBound167496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events654.exact167500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172363.bound, LeftBound167496.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172363.bound, LeftBound167496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172363.actual selector witness, LeftBound167496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172368

namespace LeftBound172369
def owner : Owner := ⟨.program ⟨257⟩, ⟨70498⟩⟩
def transferEvent : Nat := 172369
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172365 .summary, .result 167500 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172365 .summary)
      LeftBound172364.bound (LeftBound172364.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70497⟩⟩) (rawTerms := some (Proof.Events673.exact172365RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167500 .summary)
      LeftBound167499.bound (LeftBound167499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28392⟩⟩) (rawTerms := some (Proof.Events654.exact167500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167499.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172364.bound, LeftBound167499.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172364.bound, LeftBound167499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172364.actual selector witness, LeftBound167499.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172369

namespace LeftBound172373
def owner : Owner := ⟨.program ⟨257⟩, ⟨70499⟩⟩
def transferEvent : Nat := 172373
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172371 .coefficient, .predecessor 1 172372 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172371 .coefficient)
      LeftBound172368.bound (LeftBound172368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172372 .coefficient)
      LeftBound167014.bound (LeftBound167014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events652.exact167018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound167014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound167014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172368.bound, LeftBound167014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172368.bound, LeftBound167014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172368.actual selector witness, LeftBound167014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172373

namespace LeftBound172374
def owner : Owner := ⟨.program ⟨257⟩, ⟨70499⟩⟩
def transferEvent : Nat := 172374
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172370 .summary, .result 167018 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172370 .summary)
      LeftBound172369.bound (LeftBound172369.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70498⟩⟩) (rawTerms := some (Proof.Events673.exact172370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 167018 .summary)
      LeftBound167017.bound (LeftBound167017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31072⟩⟩) (rawTerms := some (Proof.Events652.exact167018RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound167017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172369.bound, LeftBound167017.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172369.bound, LeftBound167017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172369.actual selector witness, LeftBound167017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172374

namespace LeftBound172378
def owner : Owner := ⟨.program ⟨257⟩, ⟨70500⟩⟩
def transferEvent : Nat := 172378
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172376 .coefficient, .predecessor 1 172377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172376 .coefficient)
      LeftBound172373.bound (LeftBound172373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172377 .coefficient)
      LeftBound166532.bound (LeftBound166532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events650.exact166536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound166532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound166532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172373.bound, LeftBound166532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172373.bound, LeftBound166532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172373.actual selector witness, LeftBound166532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172378

namespace LeftBound172379
def owner : Owner := ⟨.program ⟨257⟩, ⟨70500⟩⟩
def transferEvent : Nat := 172379
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172375 .summary, .result 166536 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172375 .summary)
      LeftBound172374.bound (LeftBound172374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70499⟩⟩) (rawTerms := some (Proof.Events673.exact172375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 166536 .summary)
      LeftBound166535.bound (LeftBound166535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36732⟩⟩) (rawTerms := some (Proof.Events650.exact166536RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound166535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172374.bound, LeftBound166535.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172374.bound, LeftBound166535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172374.actual selector witness, LeftBound166535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172379

namespace LeftBound172383
def owner : Owner := ⟨.program ⟨257⟩, ⟨70501⟩⟩
def transferEvent : Nat := 172383
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172381 .coefficient, .predecessor 1 172382 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172381 .coefficient)
      LeftBound172378.bound (LeftBound172378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172382 .coefficient)
      LeftBound166050.bound (LeftBound166050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events648.exact166054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound166050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound166050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172378.bound, LeftBound166050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172378.bound, LeftBound166050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172378.actual selector witness, LeftBound166050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172383

namespace LeftBound172384
def owner : Owner := ⟨.program ⟨257⟩, ⟨70501⟩⟩
def transferEvent : Nat := 172384
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 172380 .summary, .result 166054 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 172380 .summary)
      LeftBound172379.bound (LeftBound172379.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70500⟩⟩) (rawTerms := some (Proof.Events673.exact172380RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound172379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 166054 .summary)
      LeftBound166053.bound (LeftBound166053.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39412⟩⟩) (rawTerms := some (Proof.Events648.exact166054RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound166053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172379.bound, LeftBound166053.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172379.bound, LeftBound166053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172379.actual selector witness, LeftBound166053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172384

namespace LeftBound172388
def owner : Owner := ⟨.program ⟨257⟩, ⟨70502⟩⟩
def transferEvent : Nat := 172388
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 172386 .coefficient, .predecessor 1 172387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 172386 .coefficient)
      LeftBound172383.bound (LeftBound172383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events673.exact172385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound172383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound172383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 172387 .coefficient)
      LeftBound165568.bound (LeftBound165568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events646.exact165572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound165568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound165568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound172383.bound, LeftBound165568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound172383.bound, LeftBound165568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound172383.actual selector witness, LeftBound165568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound172388

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
