import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard590
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard594
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard597
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard601
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard605
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard608
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard612
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard616
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard645

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99233
def owner : Owner := ⟨.program ⟨257⟩, ⟨65031⟩⟩
def transferEvent : Nat := 99233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99231 .coefficient, .predecessor 1 99232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99231 .coefficient)
      LeftBound99228.bound (LeftBound99228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99232 .coefficient)
      LeftBound95335.bound (LeftBound95335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99228.bound, LeftBound95335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99228.bound, LeftBound95335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99228.actual selector witness, LeftBound95335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99233

namespace LeftBound99234
def owner : Owner := ⟨.program ⟨257⟩, ⟨65031⟩⟩
def transferEvent : Nat := 99234
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99230 .summary, .result 95339 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99230 .summary)
      LeftBound99229.bound (LeftBound99229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62051⟩⟩) (rawTerms := some (Proof.Events387.exact99230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 95339 .summary)
      LeftBound95338.bound (LeftBound95338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65030⟩⟩) (rawTerms := some (Proof.Events372.exact95339RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99229.bound, LeftBound95338.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99229.bound, LeftBound95338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99229.actual selector witness, LeftBound95338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99234

namespace LeftBound99238
def owner : Owner := ⟨.program ⟨257⟩, ⟨70576⟩⟩
def transferEvent : Nat := 99238
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99236 .coefficient, .predecessor 1 99237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99236 .coefficient)
      LeftBound99233.bound (LeftBound99233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99237 .coefficient)
      LeftBound94853.bound (LeftBound94853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99233.bound, LeftBound94853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99233.bound, LeftBound94853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99233.actual selector witness, LeftBound94853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99238

namespace LeftBound99239
def owner : Owner := ⟨.program ⟨257⟩, ⟨70576⟩⟩
def transferEvent : Nat := 99239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99235 .summary, .result 94857 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99235 .summary)
      LeftBound99234.bound (LeftBound99234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65031⟩⟩) (rawTerms := some (Proof.Events387.exact99235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 94857 .summary)
      LeftBound94856.bound (LeftBound94856.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70575⟩⟩) (rawTerms := some (Proof.Events370.exact94857RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99234.bound, LeftBound94856.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99234.bound, LeftBound94856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99234.actual selector witness, LeftBound94856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99239

namespace LeftBound99243
def owner : Owner := ⟨.program ⟨257⟩, ⟨70577⟩⟩
def transferEvent : Nat := 99243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99241 .coefficient, .predecessor 1 99242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99241 .coefficient)
      LeftBound99238.bound (LeftBound99238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99242 .coefficient)
      LeftBound94371.bound (LeftBound94371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99238.bound, LeftBound94371.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99238.bound, LeftBound94371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99238.actual selector witness, LeftBound94371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99243

namespace LeftBound99244
def owner : Owner := ⟨.program ⟨257⟩, ⟨70577⟩⟩
def transferEvent : Nat := 99244
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99240 .summary, .result 94375 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99240 .summary)
      LeftBound99239.bound (LeftBound99239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70576⟩⟩) (rawTerms := some (Proof.Events387.exact99240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 94375 .summary)
      LeftBound94374.bound (LeftBound94374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28417⟩⟩) (rawTerms := some (Proof.Events368.exact94375RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99239.bound, LeftBound94374.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99239.bound, LeftBound94374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99239.actual selector witness, LeftBound94374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99244

namespace LeftBound99248
def owner : Owner := ⟨.program ⟨257⟩, ⟨70578⟩⟩
def transferEvent : Nat := 99248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99246 .coefficient, .predecessor 1 99247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99246 .coefficient)
      LeftBound99243.bound (LeftBound99243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99247 .coefficient)
      LeftBound93889.bound (LeftBound93889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99243.bound, LeftBound93889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99243.bound, LeftBound93889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99243.actual selector witness, LeftBound93889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99248

namespace LeftBound99249
def owner : Owner := ⟨.program ⟨257⟩, ⟨70578⟩⟩
def transferEvent : Nat := 99249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99245 .summary, .result 93893 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99245 .summary)
      LeftBound99244.bound (LeftBound99244.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70577⟩⟩) (rawTerms := some (Proof.Events387.exact99245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 93893 .summary)
      LeftBound93892.bound (LeftBound93892.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31097⟩⟩) (rawTerms := some (Proof.Events366.exact93893RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99244.bound, LeftBound93892.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99244.bound, LeftBound93892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99244.actual selector witness, LeftBound93892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99249

namespace LeftBound99253
def owner : Owner := ⟨.program ⟨257⟩, ⟨70579⟩⟩
def transferEvent : Nat := 99253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99251 .coefficient, .predecessor 1 99252 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99251 .coefficient)
      LeftBound99248.bound (LeftBound99248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99252 .coefficient)
      LeftBound93407.bound (LeftBound93407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events364.exact93411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99248.bound, LeftBound93407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99248.bound, LeftBound93407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99248.actual selector witness, LeftBound93407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99253

namespace LeftBound99254
def owner : Owner := ⟨.program ⟨257⟩, ⟨70579⟩⟩
def transferEvent : Nat := 99254
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99250 .summary, .result 93411 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99250 .summary)
      LeftBound99249.bound (LeftBound99249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70578⟩⟩) (rawTerms := some (Proof.Events387.exact99250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 93411 .summary)
      LeftBound93410.bound (LeftBound93410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36757⟩⟩) (rawTerms := some (Proof.Events364.exact93411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99249.bound, LeftBound93410.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99249.bound, LeftBound93410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99249.actual selector witness, LeftBound93410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99254

namespace LeftBound99258
def owner : Owner := ⟨.program ⟨257⟩, ⟨70580⟩⟩
def transferEvent : Nat := 99258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99256 .coefficient, .predecessor 1 99257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99256 .coefficient)
      LeftBound99253.bound (LeftBound99253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99257 .coefficient)
      LeftBound92925.bound (LeftBound92925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99253.bound, LeftBound92925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99253.bound, LeftBound92925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99253.actual selector witness, LeftBound92925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99258

namespace LeftBound99259
def owner : Owner := ⟨.program ⟨257⟩, ⟨70580⟩⟩
def transferEvent : Nat := 99259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99255 .summary, .result 92929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99255 .summary)
      LeftBound99254.bound (LeftBound99254.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70579⟩⟩) (rawTerms := some (Proof.Events387.exact99255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92929 .summary)
      LeftBound92928.bound (LeftBound92928.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39437⟩⟩) (rawTerms := some (Proof.Events363.exact92929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99254.bound, LeftBound92928.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99254.bound, LeftBound92928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99254.actual selector witness, LeftBound92928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99259

namespace LeftBound99263
def owner : Owner := ⟨.program ⟨257⟩, ⟨70581⟩⟩
def transferEvent : Nat := 99263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99261 .coefficient, .predecessor 1 99262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99261 .coefficient)
      LeftBound99258.bound (LeftBound99258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99262 .coefficient)
      LeftBound92443.bound (LeftBound92443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99258.bound, LeftBound92443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99258.bound, LeftBound92443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99258.actual selector witness, LeftBound92443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99263

namespace LeftBound99264
def owner : Owner := ⟨.program ⟨257⟩, ⟨70581⟩⟩
def transferEvent : Nat := 99264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99260 .summary, .result 92447 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99260 .summary)
      LeftBound99259.bound (LeftBound99259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70580⟩⟩) (rawTerms := some (Proof.Events387.exact99260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92447 .summary)
      LeftBound92446.bound (LeftBound92446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42117⟩⟩) (rawTerms := some (Proof.Events361.exact92447RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99259.bound, LeftBound92446.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99259.bound, LeftBound92446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99259.actual selector witness, LeftBound92446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99264

namespace LeftBound99268
def owner : Owner := ⟨.program ⟨257⟩, ⟨70582⟩⟩
def transferEvent : Nat := 99268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99266 .coefficient, .predecessor 1 99267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 99266 .coefficient)
      LeftBound99263.bound (LeftBound99263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 99267 .coefficient)
      LeftBound91961.bound (LeftBound91961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact91965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99263.bound, LeftBound91961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99263.bound, LeftBound91961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99263.actual selector witness, LeftBound91961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99268

namespace LeftBound99269
def owner : Owner := ⟨.program ⟨257⟩, ⟨70582⟩⟩
def transferEvent : Nat := 99269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99265 .summary, .result 91965 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99265 .summary)
      LeftBound99264.bound (LeftBound99264.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70581⟩⟩) (rawTerms := some (Proof.Events387.exact99265RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91965 .summary)
      LeftBound91964.bound (LeftBound91964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44797⟩⟩) (rawTerms := some (Proof.Events359.exact91965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99264.bound, LeftBound91964.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99264.bound, LeftBound91964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99264.actual selector witness, LeftBound91964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99269

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
