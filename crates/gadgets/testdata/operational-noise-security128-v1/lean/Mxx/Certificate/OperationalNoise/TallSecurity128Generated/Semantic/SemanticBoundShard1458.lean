import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1394
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1398
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1402
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1405
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1409
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1413
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1457

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound216253
def owner : Owner := ⟨.program ⟨257⟩, ⟨70184⟩⟩
def transferEvent : Nat := 216253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216251 .coefficient, .predecessor 1 216252 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216251 .coefficient)
      LeftBound216248.bound (LeftBound216248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216252 .coefficient)
      LeftBound210407.bound (LeftBound210407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events821.exact210411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound210407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound210407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216248.bound, LeftBound210407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216248.bound, LeftBound210407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216248.actual selector witness, LeftBound210407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216253

namespace LeftBound216254
def owner : Owner := ⟨.program ⟨257⟩, ⟨70184⟩⟩
def transferEvent : Nat := 216254
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216250 .summary, .result 210411 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216250 .summary)
      LeftBound216249.bound (LeftBound216249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70183⟩⟩) (rawTerms := some (Proof.Events844.exact216250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 210411 .summary)
      LeftBound210410.bound (LeftBound210410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36632⟩⟩) (rawTerms := some (Proof.Events821.exact210411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound210410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216249.bound, LeftBound210410.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216249.bound, LeftBound210410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216249.actual selector witness, LeftBound210410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216254

namespace LeftBound216258
def owner : Owner := ⟨.program ⟨257⟩, ⟨70185⟩⟩
def transferEvent : Nat := 216258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216256 .coefficient, .predecessor 1 216257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216256 .coefficient)
      LeftBound216253.bound (LeftBound216253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216257 .coefficient)
      LeftBound209925.bound (LeftBound209925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events820.exact209929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216253.bound, LeftBound209925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216253.bound, LeftBound209925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216253.actual selector witness, LeftBound209925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216258

namespace LeftBound216259
def owner : Owner := ⟨.program ⟨257⟩, ⟨70185⟩⟩
def transferEvent : Nat := 216259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216255 .summary, .result 209929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216255 .summary)
      LeftBound216254.bound (LeftBound216254.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70184⟩⟩) (rawTerms := some (Proof.Events844.exact216255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209929 .summary)
      LeftBound209928.bound (LeftBound209928.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39312⟩⟩) (rawTerms := some (Proof.Events820.exact209929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216254.bound, LeftBound209928.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216254.bound, LeftBound209928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216254.actual selector witness, LeftBound209928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216259

namespace LeftBound216263
def owner : Owner := ⟨.program ⟨257⟩, ⟨70186⟩⟩
def transferEvent : Nat := 216263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216261 .coefficient, .predecessor 1 216262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216261 .coefficient)
      LeftBound216258.bound (LeftBound216258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216262 .coefficient)
      LeftBound209443.bound (LeftBound209443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events818.exact209447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound209443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound209443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216258.bound, LeftBound209443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216258.bound, LeftBound209443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216258.actual selector witness, LeftBound209443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216263

namespace LeftBound216264
def owner : Owner := ⟨.program ⟨257⟩, ⟨70186⟩⟩
def transferEvent : Nat := 216264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216260 .summary, .result 209447 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216260 .summary)
      LeftBound216259.bound (LeftBound216259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70185⟩⟩) (rawTerms := some (Proof.Events844.exact216260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 209447 .summary)
      LeftBound209446.bound (LeftBound209446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41992⟩⟩) (rawTerms := some (Proof.Events818.exact209447RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound209446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216259.bound, LeftBound209446.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216259.bound, LeftBound209446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216259.actual selector witness, LeftBound209446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216264

namespace LeftBound216268
def owner : Owner := ⟨.program ⟨257⟩, ⟨70187⟩⟩
def transferEvent : Nat := 216268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216266 .coefficient, .predecessor 1 216267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216266 .coefficient)
      LeftBound216263.bound (LeftBound216263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216267 .coefficient)
      LeftBound208961.bound (LeftBound208961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events816.exact208965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216263.bound, LeftBound208961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216263.bound, LeftBound208961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216263.actual selector witness, LeftBound208961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216268

namespace LeftBound216269
def owner : Owner := ⟨.program ⟨257⟩, ⟨70187⟩⟩
def transferEvent : Nat := 216269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216265 .summary, .result 208965 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216265 .summary)
      LeftBound216264.bound (LeftBound216264.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70186⟩⟩) (rawTerms := some (Proof.Events844.exact216265RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208965 .summary)
      LeftBound208964.bound (LeftBound208964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44672⟩⟩) (rawTerms := some (Proof.Events816.exact208965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216264.bound, LeftBound208964.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216264.bound, LeftBound208964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216264.actual selector witness, LeftBound208964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216269

namespace LeftBound216273
def owner : Owner := ⟨.program ⟨257⟩, ⟨70188⟩⟩
def transferEvent : Nat := 216273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216271 .coefficient, .predecessor 1 216272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216271 .coefficient)
      LeftBound216268.bound (LeftBound216268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216272 .coefficient)
      LeftBound208479.bound (LeftBound208479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events814.exact208483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound208479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound208479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216268.bound, LeftBound208479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216268.bound, LeftBound208479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216268.actual selector witness, LeftBound208479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216273

namespace LeftBound216274
def owner : Owner := ⟨.program ⟨257⟩, ⟨70188⟩⟩
def transferEvent : Nat := 216274
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216270 .summary, .result 208483 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216270 .summary)
      LeftBound216269.bound (LeftBound216269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70187⟩⟩) (rawTerms := some (Proof.Events844.exact216270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208483 .summary)
      LeftBound208482.bound (LeftBound208482.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47352⟩⟩) (rawTerms := some (Proof.Events814.exact208483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208482.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216269.bound, LeftBound208482.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216269.bound, LeftBound208482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216269.actual selector witness, LeftBound208482.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216274

namespace LeftBound216278
def owner : Owner := ⟨.program ⟨257⟩, ⟨70189⟩⟩
def transferEvent : Nat := 216278
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216276 .coefficient, .predecessor 1 216277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216276 .coefficient)
      LeftBound216273.bound (LeftBound216273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216277 .coefficient)
      LeftBound207997.bound (LeftBound207997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events812.exact208001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207997.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216273.bound, LeftBound207997.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216273.bound, LeftBound207997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216273.actual selector witness, LeftBound207997.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216278

namespace LeftBound216279
def owner : Owner := ⟨.program ⟨257⟩, ⟨70189⟩⟩
def transferEvent : Nat := 216279
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216275 .summary, .result 208001 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216275 .summary)
      LeftBound216274.bound (LeftBound216274.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70188⟩⟩) (rawTerms := some (Proof.Events844.exact216275RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 208001 .summary)
      LeftBound208000.bound (LeftBound208000.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50032⟩⟩) (rawTerms := some (Proof.Events812.exact208001RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound208000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216274.bound, LeftBound208000.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216274.bound, LeftBound208000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216274.actual selector witness, LeftBound208000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216279

namespace LeftBound216283
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def transferEvent : Nat := 216283
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 216281 .coefficient) (.predecessor 1 216282 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216281 .coefficient)
      LeftBound216278.bound (LeftBound216278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216282 .coefficient)
      LeftAuthority207502.bound (LeftAuthority207502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207502.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound216278.bound LeftAuthority207502.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216278.bound, LeftAuthority207502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound216278.actual selector witness) * (LeftAuthority207502.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216283

namespace LeftBound216284
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def transferEvent : Nat := 216284
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩ [⟨.result 207503 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207503 .coefficient)
      LeftAuthority207502.bound (LeftAuthority207502.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71236⟩⟩) (rawTerms := some (Proof.Events810.exact207503RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207502.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority207502.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority207502.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound216284

namespace LeftBound216285
def owner : Owner := ⟨.program ⟨257⟩, ⟨71238⟩⟩
def transferEvent : Nat := 216285
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 216280 .summary) (.transfer 216284) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216280 .summary)
      LeftBound216279.bound (LeftBound216279.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70189⟩⟩) (rawTerms := some (Proof.Events844.exact216280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 216284)
      LeftBound216284.bound (LeftBound216284.actual selector witness) := by
  exact .transfer (LeftBound216284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound216279.bound LeftBound216284.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216279.bound, LeftBound216284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound216279.actual selector witness) * (LeftBound216284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216285

namespace LeftBound216364
def owner : Owner := ⟨.program ⟨257⟩, ⟨68372⟩⟩
def transferEvent : Nat := 216364
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 216362 .coefficient) (.value (.predecessor 1 216363 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216362 .coefficient)
      LeftAuthority216360.bound (LeftAuthority216360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events845.exact216361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216363 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority216360.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216360.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority216360.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound216364

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
