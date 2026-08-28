import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1499
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1503
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1507
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1510
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1514
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1518
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1521
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1525
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1529
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1558

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound230859
def owner : Owner := ⟨.program ⟨257⟩, ⟨64845⟩⟩
def transferEvent : Nat := 230859
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230855 .summary, .result 226964 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230855 .summary)
      LeftBound230854.bound (LeftBound230854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61865⟩⟩) (rawTerms := some (Proof.Events901.exact230855RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226964 .summary)
      LeftBound226963.bound (LeftBound226963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64844⟩⟩) (rawTerms := some (Proof.Events886.exact226964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound226963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230854.bound, LeftBound226963.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230854.bound, LeftBound226963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230854.actual selector witness, LeftBound226963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230859

namespace LeftBound230863
def owner : Owner := ⟨.program ⟨257⟩, ⟨70102⟩⟩
def transferEvent : Nat := 230863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230861 .coefficient, .predecessor 1 230862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230861 .coefficient)
      LeftBound230858.bound (LeftBound230858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230862 .coefficient)
      LeftBound226478.bound (LeftBound226478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events884.exact226482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230858.bound, LeftBound226478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230858.bound, LeftBound226478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230858.actual selector witness, LeftBound226478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230863

namespace LeftBound230864
def owner : Owner := ⟨.program ⟨257⟩, ⟨70102⟩⟩
def transferEvent : Nat := 230864
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230860 .summary, .result 226482 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230860 .summary)
      LeftBound230859.bound (LeftBound230859.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64845⟩⟩) (rawTerms := some (Proof.Events901.exact230860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226482 .summary)
      LeftBound226481.bound (LeftBound226481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70101⟩⟩) (rawTerms := some (Proof.Events884.exact226482RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound226481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230859.bound, LeftBound226481.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230859.bound, LeftBound226481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230859.actual selector witness, LeftBound226481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230864

namespace LeftBound230868
def owner : Owner := ⟨.program ⟨257⟩, ⟨70103⟩⟩
def transferEvent : Nat := 230868
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230866 .coefficient, .predecessor 1 230867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230866 .coefficient)
      LeftBound230863.bound (LeftBound230863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230867 .coefficient)
      LeftBound225996.bound (LeftBound225996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact226000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230863.bound, LeftBound225996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230863.bound, LeftBound225996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230863.actual selector witness, LeftBound225996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230868

namespace LeftBound230869
def owner : Owner := ⟨.program ⟨257⟩, ⟨70103⟩⟩
def transferEvent : Nat := 230869
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230865 .summary, .result 226000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230865 .summary)
      LeftBound230864.bound (LeftBound230864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70102⟩⟩) (rawTerms := some (Proof.Events901.exact230865RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 226000 .summary)
      LeftBound225999.bound (LeftBound225999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28267⟩⟩) (rawTerms := some (Proof.Events882.exact226000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound225999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230864.bound, LeftBound225999.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230864.bound, LeftBound225999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230864.actual selector witness, LeftBound225999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230869

namespace LeftBound230873
def owner : Owner := ⟨.program ⟨257⟩, ⟨70104⟩⟩
def transferEvent : Nat := 230873
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230871 .coefficient, .predecessor 1 230872 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230871 .coefficient)
      LeftBound230868.bound (LeftBound230868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230872 .coefficient)
      LeftBound225514.bound (LeftBound225514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events880.exact225518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230868.bound, LeftBound225514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230868.bound, LeftBound225514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230868.actual selector witness, LeftBound225514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230873

namespace LeftBound230874
def owner : Owner := ⟨.program ⟨257⟩, ⟨70104⟩⟩
def transferEvent : Nat := 230874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230870 .summary, .result 225518 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230870 .summary)
      LeftBound230869.bound (LeftBound230869.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70103⟩⟩) (rawTerms := some (Proof.Events901.exact230870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225518 .summary)
      LeftBound225517.bound (LeftBound225517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30947⟩⟩) (rawTerms := some (Proof.Events880.exact225518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound225517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230869.bound, LeftBound225517.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230869.bound, LeftBound225517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230869.actual selector witness, LeftBound225517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230874

namespace LeftBound230878
def owner : Owner := ⟨.program ⟨257⟩, ⟨70105⟩⟩
def transferEvent : Nat := 230878
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230876 .coefficient, .predecessor 1 230877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230876 .coefficient)
      LeftBound230873.bound (LeftBound230873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230877 .coefficient)
      LeftBound225032.bound (LeftBound225032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events879.exact225036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230873.bound, LeftBound225032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230873.bound, LeftBound225032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230873.actual selector witness, LeftBound225032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230878

namespace LeftBound230879
def owner : Owner := ⟨.program ⟨257⟩, ⟨70105⟩⟩
def transferEvent : Nat := 230879
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230875 .summary, .result 225036 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230875 .summary)
      LeftBound230874.bound (LeftBound230874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70104⟩⟩) (rawTerms := some (Proof.Events901.exact230875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225036 .summary)
      LeftBound225035.bound (LeftBound225035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36607⟩⟩) (rawTerms := some (Proof.Events879.exact225036RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound225035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230874.bound, LeftBound225035.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230874.bound, LeftBound225035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230874.actual selector witness, LeftBound225035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230879

namespace LeftBound230883
def owner : Owner := ⟨.program ⟨257⟩, ⟨70106⟩⟩
def transferEvent : Nat := 230883
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230881 .coefficient, .predecessor 1 230882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230881 .coefficient)
      LeftBound230878.bound (LeftBound230878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230882 .coefficient)
      LeftBound224550.bound (LeftBound224550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events877.exact224554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230878.bound, LeftBound224550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230878.bound, LeftBound224550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230878.actual selector witness, LeftBound224550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230883

namespace LeftBound230884
def owner : Owner := ⟨.program ⟨257⟩, ⟨70106⟩⟩
def transferEvent : Nat := 230884
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230880 .summary, .result 224554 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230880 .summary)
      LeftBound230879.bound (LeftBound230879.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70105⟩⟩) (rawTerms := some (Proof.Events901.exact230880RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224554 .summary)
      LeftBound224553.bound (LeftBound224553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39287⟩⟩) (rawTerms := some (Proof.Events877.exact224554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230879.bound, LeftBound224553.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230879.bound, LeftBound224553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230879.actual selector witness, LeftBound224553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230884

namespace LeftBound230888
def owner : Owner := ⟨.program ⟨257⟩, ⟨70107⟩⟩
def transferEvent : Nat := 230888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230886 .coefficient, .predecessor 1 230887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230886 .coefficient)
      LeftBound230883.bound (LeftBound230883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230887 .coefficient)
      LeftBound224068.bound (LeftBound224068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events875.exact224072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound224068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound224068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230883.bound, LeftBound224068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230883.bound, LeftBound224068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230883.actual selector witness, LeftBound224068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230888

namespace LeftBound230889
def owner : Owner := ⟨.program ⟨257⟩, ⟨70107⟩⟩
def transferEvent : Nat := 230889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230885 .summary, .result 224072 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230885 .summary)
      LeftBound230884.bound (LeftBound230884.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70106⟩⟩) (rawTerms := some (Proof.Events901.exact230885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 224072 .summary)
      LeftBound224071.bound (LeftBound224071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41967⟩⟩) (rawTerms := some (Proof.Events875.exact224072RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound224071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230884.bound, LeftBound224071.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230884.bound, LeftBound224071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230884.actual selector witness, LeftBound224071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230889

namespace LeftBound230893
def owner : Owner := ⟨.program ⟨257⟩, ⟨70108⟩⟩
def transferEvent : Nat := 230893
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230891 .coefficient, .predecessor 1 230892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230891 .coefficient)
      LeftBound230888.bound (LeftBound230888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230892 .coefficient)
      LeftBound223586.bound (LeftBound223586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events873.exact223590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound223586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound223586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230888.bound, LeftBound223586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230888.bound, LeftBound223586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230888.actual selector witness, LeftBound223586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230893

namespace LeftBound230894
def owner : Owner := ⟨.program ⟨257⟩, ⟨70108⟩⟩
def transferEvent : Nat := 230894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 230890 .summary, .result 223590 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 230890 .summary)
      LeftBound230889.bound (LeftBound230889.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70107⟩⟩) (rawTerms := some (Proof.Events901.exact230890RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound230889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 223590 .summary)
      LeftBound223589.bound (LeftBound223589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44647⟩⟩) (rawTerms := some (Proof.Events873.exact223590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound223589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230889.bound, LeftBound223589.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230889.bound, LeftBound223589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230889.actual selector witness, LeftBound223589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230894

namespace LeftBound230898
def owner : Owner := ⟨.program ⟨257⟩, ⟨70109⟩⟩
def transferEvent : Nat := 230898
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 230896 .coefficient, .predecessor 1 230897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 230896 .coefficient)
      LeftBound230893.bound (LeftBound230893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events901.exact230895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound230893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 230897 .coefficient)
      LeftBound223104.bound (LeftBound223104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events871.exact223108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound223104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound223104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound230893.bound, LeftBound223104.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound230893.bound, LeftBound223104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound230893.actual selector witness, LeftBound223104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound230898

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
