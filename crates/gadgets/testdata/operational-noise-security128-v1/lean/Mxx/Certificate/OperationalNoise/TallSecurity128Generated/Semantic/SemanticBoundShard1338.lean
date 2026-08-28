import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1337

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound199233
def owner : Owner := ⟨.program ⟨257⟩, ⟨50606⟩⟩
def transferEvent : Nat := 199233
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 199228 .summary) (.transfer 199232) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199228 .summary)
      LeftBound199226.bound (LeftBound199226.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50605⟩⟩) (rawTerms := some (Proof.Events778.exact199228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 199232)
      LeftBound199232.bound (LeftBound199232.actual selector witness) := by
  exact .transfer (LeftBound199232.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound199226.bound LeftBound199232.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199226.bound, LeftBound199232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound199226.actual selector witness) * (LeftBound199232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199233

namespace LeftBound199241
def owner : Owner := ⟨.program ⟨257⟩, ⟨50607⟩⟩
def transferEvent : Nat := 199241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 199239 .coefficient, .predecessor 1 199240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199239 .coefficient)
      LeftBound199231.bound (LeftBound199231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199240 .coefficient)
      LeftBound199203.bound (LeftBound199203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound199231.bound, LeftBound199203.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199231.bound, LeftBound199203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound199231.actual selector witness, LeftBound199203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound199241

namespace LeftBound199243
def owner : Owner := ⟨.program ⟨257⟩, ⟨50607⟩⟩
def transferEvent : Nat := 199243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 199238 .summary, .result 199208 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199238 .summary)
      LeftBound199233.bound (LeftBound199233.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50606⟩⟩) (rawTerms := some (Proof.Events778.exact199238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199208 .summary)
      LeftBound199205.bound (LeftBound199205.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50602⟩⟩) (rawTerms := some (Proof.Events778.exact199208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199205.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound199233.bound, LeftBound199205.bound]
def bound : CoeffClass := .finite ⟨279181393920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199233.bound, LeftBound199205.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound199233.actual selector witness, LeftBound199205.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound199243

namespace LeftBound199247
def owner : Owner := ⟨.program ⟨257⟩, ⟨52542⟩⟩
def transferEvent : Nat := 199247
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 199245 .coefficient) (.predecessor 1 199246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199245 .coefficient)
      LeftBound199241.bound (LeftBound199241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199246 .coefficient)
      LeftAuthority199179.bound (LeftAuthority199179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199179.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound199241.bound LeftAuthority199179.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199241.bound, LeftAuthority199179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound199241.actual selector witness) * (LeftAuthority199179.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199247

namespace LeftBound199248
def owner : Owner := ⟨.program ⟨257⟩, ⟨52542⟩⟩
def transferEvent : Nat := 199248
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52541⟩⟩]⟩ [⟨.result 199180 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199180 .coefficient)
      LeftAuthority199179.bound (LeftAuthority199179.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52541⟩⟩) (rawTerms := some (Proof.Events778.exact199180RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199179.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority199179.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority199179.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound199248

namespace LeftBound199249
def owner : Owner := ⟨.program ⟨257⟩, ⟨52542⟩⟩
def transferEvent : Nat := 199249
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 199244 .summary) (.transfer 199248) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199244 .summary)
      LeftBound199243.bound (LeftBound199243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50607⟩⟩) (rawTerms := some (Proof.Events778.exact199244RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound199243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 199248)
      LeftBound199248.bound (LeftBound199248.actual selector witness) := by
  exact .transfer (LeftBound199248.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound199243.bound LeftBound199248.bound
def bound : CoeffClass := .finite ⟨2997687391345233100800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199243.bound, LeftBound199248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound199243.actual selector witness) * (LeftBound199248.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199249

namespace LeftBound199260
def owner : Owner := ⟨.program ⟨257⟩, ⟨51471⟩⟩
def transferEvent : Nat := 199260
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 199258 .coefficient) (.value (.predecessor 1 199259 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199258 .coefficient)
      LeftAuthority199256.bound (LeftAuthority199256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199259 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority199256.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199256.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority199256.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound199260

namespace LeftBound199264
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def transferEvent : Nat := 199264
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 199262 .coefficient) (.predecessor 1 199263 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199262 .coefficient)
      LeftBound192992.bound (LeftBound192992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199263 .coefficient)
      LeftBound199260.bound (LeftBound199260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199260.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192992.bound LeftBound199260.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192992.bound, LeftBound199260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192992.actual selector witness) * (LeftBound199260.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199264

namespace LeftBound199265
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def transferEvent : Nat := 199265
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩ [⟨.result 199257 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 199257 .coefficient)
      LeftAuthority199256.bound (LeftAuthority199256.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51469⟩⟩) (rawTerms := some (Proof.Events778.exact199257RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199256.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority199256.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority199256.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound199265

namespace LeftBound199266
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def transferEvent : Nat := 199266
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 199265) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 199265)
      LeftBound199265.bound (LeftBound199265.actual selector witness) := by
  exact .transfer (LeftBound199265.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound199265.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound199265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound199265.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199266

namespace LeftBound199345
def owner : Owner := ⟨.program ⟨257⟩, ⟨50600⟩⟩
def transferEvent : Nat := 199345
def frameStart : Nat := 199316
def rule : BoundRule := .product (.predecessor 0 199343 .coefficient) (.predecessor 1 199344 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199343 .coefficient)
      LeftAuthority199341.bound (LeftAuthority199341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199344 .coefficient)
      LeftAuthority199338.bound (LeftAuthority199338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority199341.bound LeftAuthority199338.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199341.bound, LeftAuthority199338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority199341.actual selector witness) * (LeftAuthority199338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199345

namespace LeftBound199349
def owner : Owner := ⟨.program ⟨257⟩, ⟨50601⟩⟩
def transferEvent : Nat := 199349
def frameStart : Nat := 199316
def rule : BoundRule := .identity (.predecessor 0 199348 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199348 .coefficient)
      LeftBound199345.bound (LeftBound199345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199345.derived selector witness)

def rawBound : CoeffClass := LeftBound199345.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound199345.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound199349

namespace LeftBound199366
def owner : Owner := ⟨.program ⟨257⟩, ⟨52294⟩⟩
def transferEvent : Nat := 199366
def frameStart : Nat := 199316
def rule : BoundRule := .sum [.predecessor 0 199364 .coefficient, .predecessor 1 199365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199364 .coefficient)
      LeftBound199349.bound (LeftBound199349.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound199349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199365 .coefficient)
      LeftAuthority199362.bound (LeftAuthority199362.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority199362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound199349.bound, LeftAuthority199362.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199349.bound, LeftAuthority199362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound199349.actual selector witness, LeftAuthority199362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound199366

namespace LeftBound199369
def owner : Owner := ⟨.program ⟨257⟩, ⟨52295⟩⟩
def transferEvent : Nat := 199369
def frameStart : Nat := 199316
def rule : BoundRule := .identity (.predecessor 0 199368 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199368 .coefficient)
      LeftBound199366.bound (LeftBound199366.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound199366.derived selector witness)

def rawBound : CoeffClass := LeftBound199366.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound199366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound199366.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound199369

namespace LeftBound199375
def owner : Owner := ⟨.program ⟨257⟩, ⟨52296⟩⟩
def transferEvent : Nat := 199375
def frameStart : Nat := 199316
def rule : BoundRule := .product (.predecessor 0 199373 .coefficient) (.predecessor 1 199374 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199373 .coefficient)
      LeftAuthority199371.bound (LeftAuthority199371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199374 .coefficient)
      LeftBound199369.bound (LeftBound199369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound199369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound199369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority199371.bound LeftBound199369.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199371.bound, LeftBound199369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority199371.actual selector witness) * (LeftBound199369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound199375

namespace LeftBound199391
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 199391
def frameStart : Nat := 199316
def rule : BoundRule := .scale (.predecessor 0 199389 .coefficient) (.value (.predecessor 1 199390 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 199389 .coefficient)
      LeftAuthority199387.bound (LeftAuthority199387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events778.exact199388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority199387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority199387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 199390 .coefficient)
      LeftAuthority199378.bound (LeftAuthority199378.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority199378.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority199387.bound LeftAuthority199378.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority199387.bound, LeftAuthority199378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority199387.actual selector witness) * (LeftAuthority199378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound199391

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
