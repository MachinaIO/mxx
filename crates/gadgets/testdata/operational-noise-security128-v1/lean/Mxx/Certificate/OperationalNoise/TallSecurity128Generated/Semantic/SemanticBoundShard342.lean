import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard279
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard282
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard290
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard293
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard297
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard300
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard304
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard341

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55369
def owner : Owner := ⟨.program ⟨257⟩, ⟨70814⟩⟩
def transferEvent : Nat := 55369
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55365 .summary, .result 50500 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55365 .summary)
      LeftBound55364.bound (LeftBound55364.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70813⟩⟩) (rawTerms := some (Proof.Events216.exact55365RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 50500 .summary)
      LeftBound50499.bound (LeftBound50499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28492⟩⟩) (rawTerms := some (Proof.Events197.exact50500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50499.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55364.bound, LeftBound50499.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55364.bound, LeftBound50499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55364.actual selector witness, LeftBound50499.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55369

namespace LeftBound55373
def owner : Owner := ⟨.program ⟨257⟩, ⟨70815⟩⟩
def transferEvent : Nat := 55373
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55371 .coefficient, .predecessor 1 55372 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55371 .coefficient)
      LeftBound55368.bound (LeftBound55368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55372 .coefficient)
      LeftBound50014.bound (LeftBound50014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact50018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55368.bound, LeftBound50014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55368.bound, LeftBound50014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55368.actual selector witness, LeftBound50014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55373

namespace LeftBound55374
def owner : Owner := ⟨.program ⟨257⟩, ⟨70815⟩⟩
def transferEvent : Nat := 55374
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55370 .summary, .result 50018 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55370 .summary)
      LeftBound55369.bound (LeftBound55369.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70814⟩⟩) (rawTerms := some (Proof.Events216.exact55370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 50018 .summary)
      LeftBound50017.bound (LeftBound50017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31172⟩⟩) (rawTerms := some (Proof.Events195.exact50018RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55369.bound, LeftBound50017.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55369.bound, LeftBound50017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55369.actual selector witness, LeftBound50017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55374

namespace LeftBound55378
def owner : Owner := ⟨.program ⟨257⟩, ⟨70816⟩⟩
def transferEvent : Nat := 55378
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55376 .coefficient, .predecessor 1 55377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55376 .coefficient)
      LeftBound55373.bound (LeftBound55373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55377 .coefficient)
      LeftBound49532.bound (LeftBound49532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55373.bound, LeftBound49532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55373.bound, LeftBound49532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55373.actual selector witness, LeftBound49532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55378

namespace LeftBound55379
def owner : Owner := ⟨.program ⟨257⟩, ⟨70816⟩⟩
def transferEvent : Nat := 55379
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55375 .summary, .result 49536 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55375 .summary)
      LeftBound55374.bound (LeftBound55374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70815⟩⟩) (rawTerms := some (Proof.Events216.exact55375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 49536 .summary)
      LeftBound49535.bound (LeftBound49535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36832⟩⟩) (rawTerms := some (Proof.Events193.exact49536RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55374.bound, LeftBound49535.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55374.bound, LeftBound49535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55374.actual selector witness, LeftBound49535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55379

namespace LeftBound55383
def owner : Owner := ⟨.program ⟨257⟩, ⟨70817⟩⟩
def transferEvent : Nat := 55383
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55381 .coefficient, .predecessor 1 55382 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55381 .coefficient)
      LeftBound55378.bound (LeftBound55378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55382 .coefficient)
      LeftBound49050.bound (LeftBound49050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55378.bound, LeftBound49050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55378.bound, LeftBound49050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55378.actual selector witness, LeftBound49050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55383

namespace LeftBound55384
def owner : Owner := ⟨.program ⟨257⟩, ⟨70817⟩⟩
def transferEvent : Nat := 55384
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55380 .summary, .result 49054 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55380 .summary)
      LeftBound55379.bound (LeftBound55379.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70816⟩⟩) (rawTerms := some (Proof.Events216.exact55380RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 49054 .summary)
      LeftBound49053.bound (LeftBound49053.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39512⟩⟩) (rawTerms := some (Proof.Events191.exact49054RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55379.bound, LeftBound49053.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55379.bound, LeftBound49053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55379.actual selector witness, LeftBound49053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55384

namespace LeftBound55388
def owner : Owner := ⟨.program ⟨257⟩, ⟨70818⟩⟩
def transferEvent : Nat := 55388
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55386 .coefficient, .predecessor 1 55387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55386 .coefficient)
      LeftBound55383.bound (LeftBound55383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55387 .coefficient)
      LeftBound48568.bound (LeftBound48568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55383.bound, LeftBound48568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55383.bound, LeftBound48568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55383.actual selector witness, LeftBound48568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55388

namespace LeftBound55389
def owner : Owner := ⟨.program ⟨257⟩, ⟨70818⟩⟩
def transferEvent : Nat := 55389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55385 .summary, .result 48572 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55385 .summary)
      LeftBound55384.bound (LeftBound55384.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70817⟩⟩) (rawTerms := some (Proof.Events216.exact55385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 48572 .summary)
      LeftBound48571.bound (LeftBound48571.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42192⟩⟩) (rawTerms := some (Proof.Events189.exact48572RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55384.bound, LeftBound48571.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55384.bound, LeftBound48571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55384.actual selector witness, LeftBound48571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55389

namespace LeftBound55393
def owner : Owner := ⟨.program ⟨257⟩, ⟨70819⟩⟩
def transferEvent : Nat := 55393
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55391 .coefficient, .predecessor 1 55392 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55391 .coefficient)
      LeftBound55388.bound (LeftBound55388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55392 .coefficient)
      LeftBound48086.bound (LeftBound48086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55388.bound, LeftBound48086.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55388.bound, LeftBound48086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55388.actual selector witness, LeftBound48086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55393

namespace LeftBound55394
def owner : Owner := ⟨.program ⟨257⟩, ⟨70819⟩⟩
def transferEvent : Nat := 55394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55390 .summary, .result 48090 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55390 .summary)
      LeftBound55389.bound (LeftBound55389.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70818⟩⟩) (rawTerms := some (Proof.Events216.exact55390RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 48090 .summary)
      LeftBound48089.bound (LeftBound48089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44872⟩⟩) (rawTerms := some (Proof.Events187.exact48090RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48089.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55389.bound, LeftBound48089.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55389.bound, LeftBound48089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55389.actual selector witness, LeftBound48089.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55394

namespace LeftBound55398
def owner : Owner := ⟨.program ⟨257⟩, ⟨70820⟩⟩
def transferEvent : Nat := 55398
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55396 .coefficient, .predecessor 1 55397 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55396 .coefficient)
      LeftBound55393.bound (LeftBound55393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55397 .coefficient)
      LeftBound47604.bound (LeftBound47604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55393.bound, LeftBound47604.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55393.bound, LeftBound47604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55393.actual selector witness, LeftBound47604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55398

namespace LeftBound55399
def owner : Owner := ⟨.program ⟨257⟩, ⟨70820⟩⟩
def transferEvent : Nat := 55399
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55395 .summary, .result 47608 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55395 .summary)
      LeftBound55394.bound (LeftBound55394.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70819⟩⟩) (rawTerms := some (Proof.Events216.exact55395RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47608 .summary)
      LeftBound47607.bound (LeftBound47607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47552⟩⟩) (rawTerms := some (Proof.Events185.exact47608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55394.bound, LeftBound47607.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55394.bound, LeftBound47607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55394.actual selector witness, LeftBound47607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55399

namespace LeftBound55403
def owner : Owner := ⟨.program ⟨257⟩, ⟨70821⟩⟩
def transferEvent : Nat := 55403
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55401 .coefficient, .predecessor 1 55402 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55401 .coefficient)
      LeftBound55398.bound (LeftBound55398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55402 .coefficient)
      LeftBound47122.bound (LeftBound47122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55398.bound, LeftBound47122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55398.bound, LeftBound47122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55398.actual selector witness, LeftBound47122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55403

namespace LeftBound55404
def owner : Owner := ⟨.program ⟨257⟩, ⟨70821⟩⟩
def transferEvent : Nat := 55404
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55400 .summary, .result 47126 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55400 .summary)
      LeftBound55399.bound (LeftBound55399.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70820⟩⟩) (rawTerms := some (Proof.Events216.exact55400RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 47126 .summary)
      LeftBound47125.bound (LeftBound47125.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50232⟩⟩) (rawTerms := some (Proof.Events184.exact47126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47125.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55399.bound, LeftBound47125.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55399.bound, LeftBound47125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55399.actual selector witness, LeftBound47125.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55404

namespace LeftBound55408
def owner : Owner := ⟨.program ⟨257⟩, ⟨71503⟩⟩
def transferEvent : Nat := 55408
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55406 .coefficient) (.predecessor 1 55407 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55406 .coefficient)
      LeftBound55403.bound (LeftBound55403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55407 .coefficient)
      LeftAuthority46627.bound (LeftAuthority46627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55403.bound LeftAuthority46627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55403.bound, LeftAuthority46627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55403.actual selector witness) * (LeftAuthority46627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55408

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
