import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1300
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1304
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1307
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1308
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1311
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1315
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1318
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1322
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1326
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1329
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1355

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound201604
def owner : Owner := ⟨.program ⟨257⟩, ⟨61958⟩⟩
def transferEvent : Nat := 201604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201600 .summary, .result 198196 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201600 .summary)
      LeftBound201599.bound (LeftBound201599.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58978⟩⟩) (rawTerms := some (Proof.Events787.exact201600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 198196 .summary)
      LeftBound198195.bound (LeftBound198195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61957⟩⟩) (rawTerms := some (Proof.Events774.exact198196RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound198195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201599.bound, LeftBound198195.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201599.bound, LeftBound198195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201599.actual selector witness, LeftBound198195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201604

namespace LeftBound201608
def owner : Owner := ⟨.program ⟨257⟩, ⟨64938⟩⟩
def transferEvent : Nat := 201608
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201606 .coefficient, .predecessor 1 201607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201606 .coefficient)
      LeftBound201603.bound (LeftBound201603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201607 .coefficient)
      LeftBound197710.bound (LeftBound197710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events772.exact197714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201603.bound, LeftBound197710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201603.bound, LeftBound197710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201603.actual selector witness, LeftBound197710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201608

namespace LeftBound201609
def owner : Owner := ⟨.program ⟨257⟩, ⟨64938⟩⟩
def transferEvent : Nat := 201609
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201605 .summary, .result 197714 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201605 .summary)
      LeftBound201604.bound (LeftBound201604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61958⟩⟩) (rawTerms := some (Proof.Events787.exact201605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 197714 .summary)
      LeftBound197713.bound (LeftBound197713.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64937⟩⟩) (rawTerms := some (Proof.Events772.exact197714RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound197713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201604.bound, LeftBound197713.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201604.bound, LeftBound197713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201604.actual selector witness, LeftBound197713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201609

namespace LeftBound201613
def owner : Owner := ⟨.program ⟨257⟩, ⟨70339⟩⟩
def transferEvent : Nat := 201613
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201611 .coefficient, .predecessor 1 201612 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201611 .coefficient)
      LeftBound201608.bound (LeftBound201608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201612 .coefficient)
      LeftBound197228.bound (LeftBound197228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201608.bound, LeftBound197228.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201608.bound, LeftBound197228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201608.actual selector witness, LeftBound197228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201613

namespace LeftBound201614
def owner : Owner := ⟨.program ⟨257⟩, ⟨70339⟩⟩
def transferEvent : Nat := 201614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201610 .summary, .result 197232 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201610 .summary)
      LeftBound201609.bound (LeftBound201609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64938⟩⟩) (rawTerms := some (Proof.Events787.exact201610RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 197232 .summary)
      LeftBound197231.bound (LeftBound197231.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70338⟩⟩) (rawTerms := some (Proof.Events770.exact197232RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound197231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201609.bound, LeftBound197231.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201609.bound, LeftBound197231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201609.actual selector witness, LeftBound197231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201614

namespace LeftBound201618
def owner : Owner := ⟨.program ⟨257⟩, ⟨70340⟩⟩
def transferEvent : Nat := 201618
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201616 .coefficient, .predecessor 1 201617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201616 .coefficient)
      LeftBound201613.bound (LeftBound201613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201613.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201617 .coefficient)
      LeftBound196746.bound (LeftBound196746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events768.exact196750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201613.bound, LeftBound196746.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201613.bound, LeftBound196746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201613.actual selector witness, LeftBound196746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201618

namespace LeftBound201619
def owner : Owner := ⟨.program ⟨257⟩, ⟨70340⟩⟩
def transferEvent : Nat := 201619
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201615 .summary, .result 196750 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201615 .summary)
      LeftBound201614.bound (LeftBound201614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70339⟩⟩) (rawTerms := some (Proof.Events787.exact201615RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196750 .summary)
      LeftBound196749.bound (LeftBound196749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28342⟩⟩) (rawTerms := some (Proof.Events768.exact196750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound196749.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201614.bound, LeftBound196749.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201614.bound, LeftBound196749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201614.actual selector witness, LeftBound196749.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201619

namespace LeftBound201623
def owner : Owner := ⟨.program ⟨257⟩, ⟨70341⟩⟩
def transferEvent : Nat := 201623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201621 .coefficient, .predecessor 1 201622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201621 .coefficient)
      LeftBound201618.bound (LeftBound201618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201622 .coefficient)
      LeftBound196264.bound (LeftBound196264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events766.exact196268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201618.bound, LeftBound196264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201618.bound, LeftBound196264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201618.actual selector witness, LeftBound196264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201623

namespace LeftBound201624
def owner : Owner := ⟨.program ⟨257⟩, ⟨70341⟩⟩
def transferEvent : Nat := 201624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201620 .summary, .result 196268 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201620 .summary)
      LeftBound201619.bound (LeftBound201619.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70340⟩⟩) (rawTerms := some (Proof.Events787.exact201620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196268 .summary)
      LeftBound196267.bound (LeftBound196267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31022⟩⟩) (rawTerms := some (Proof.Events766.exact196268RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound196267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201619.bound, LeftBound196267.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201619.bound, LeftBound196267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201619.actual selector witness, LeftBound196267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201624

namespace LeftBound201628
def owner : Owner := ⟨.program ⟨257⟩, ⟨70342⟩⟩
def transferEvent : Nat := 201628
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201626 .coefficient, .predecessor 1 201627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201626 .coefficient)
      LeftBound201623.bound (LeftBound201623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201627 .coefficient)
      LeftBound195782.bound (LeftBound195782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events764.exact195786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201623.bound, LeftBound195782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201623.bound, LeftBound195782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201623.actual selector witness, LeftBound195782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201628

namespace LeftBound201629
def owner : Owner := ⟨.program ⟨257⟩, ⟨70342⟩⟩
def transferEvent : Nat := 201629
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201625 .summary, .result 195786 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201625 .summary)
      LeftBound201624.bound (LeftBound201624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70341⟩⟩) (rawTerms := some (Proof.Events787.exact201625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 195786 .summary)
      LeftBound195785.bound (LeftBound195785.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36682⟩⟩) (rawTerms := some (Proof.Events764.exact195786RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound195785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201624.bound, LeftBound195785.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201624.bound, LeftBound195785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201624.actual selector witness, LeftBound195785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201629

namespace LeftBound201633
def owner : Owner := ⟨.program ⟨257⟩, ⟨70343⟩⟩
def transferEvent : Nat := 201633
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201631 .coefficient, .predecessor 1 201632 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201631 .coefficient)
      LeftBound201628.bound (LeftBound201628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201632 .coefficient)
      LeftBound195300.bound (LeftBound195300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events762.exact195304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201628.bound, LeftBound195300.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201628.bound, LeftBound195300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201628.actual selector witness, LeftBound195300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201633

namespace LeftBound201634
def owner : Owner := ⟨.program ⟨257⟩, ⟨70343⟩⟩
def transferEvent : Nat := 201634
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201630 .summary, .result 195304 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201630 .summary)
      LeftBound201629.bound (LeftBound201629.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70342⟩⟩) (rawTerms := some (Proof.Events787.exact201630RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 195304 .summary)
      LeftBound195303.bound (LeftBound195303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39362⟩⟩) (rawTerms := some (Proof.Events762.exact195304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound195303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201629.bound, LeftBound195303.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201629.bound, LeftBound195303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201629.actual selector witness, LeftBound195303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201634

namespace LeftBound201638
def owner : Owner := ⟨.program ⟨257⟩, ⟨70344⟩⟩
def transferEvent : Nat := 201638
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201636 .coefficient, .predecessor 1 201637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201636 .coefficient)
      LeftBound201633.bound (LeftBound201633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201637 .coefficient)
      LeftBound194818.bound (LeftBound194818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201633.bound, LeftBound194818.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201633.bound, LeftBound194818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201633.actual selector witness, LeftBound194818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201638

namespace LeftBound201639
def owner : Owner := ⟨.program ⟨257⟩, ⟨70344⟩⟩
def transferEvent : Nat := 201639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 201635 .summary, .result 194822 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 201635 .summary)
      LeftBound201634.bound (LeftBound201634.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70343⟩⟩) (rawTerms := some (Proof.Events787.exact201635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound201634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 194822 .summary)
      LeftBound194821.bound (LeftBound194821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42042⟩⟩) (rawTerms := some (Proof.Events761.exact194822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201634.bound, LeftBound194821.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201634.bound, LeftBound194821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201634.actual selector witness, LeftBound194821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201639

namespace LeftBound201643
def owner : Owner := ⟨.program ⟨257⟩, ⟨70345⟩⟩
def transferEvent : Nat := 201643
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 201641 .coefficient, .predecessor 1 201642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 201641 .coefficient)
      LeftBound201638.bound (LeftBound201638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events787.exact201640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound201638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 201642 .coefficient)
      LeftBound194336.bound (LeftBound194336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events759.exact194340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound201638.bound, LeftBound194336.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound201638.bound, LeftBound194336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound201638.actual selector witness, LeftBound194336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound201643

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
