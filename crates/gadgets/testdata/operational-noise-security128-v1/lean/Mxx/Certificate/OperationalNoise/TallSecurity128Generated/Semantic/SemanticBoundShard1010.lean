import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1009

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound152203
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def transferEvent : Nat := 152203
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 152198 .summary) (.transfer 152202) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152198 .summary)
      LeftBound152197.bound (LeftBound152197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30568⟩⟩) (rawTerms := some (Proof.Events594.exact152198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 152202)
      LeftBound152202.bound (LeftBound152202.actual selector witness) := by
  exact .transfer (LeftBound152202.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152197.bound LeftBound152202.bound
def bound : CoeffClass := .finite ⟨32192146870060190229763897425920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152197.bound, LeftBound152202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152197.actual selector witness) * (LeftBound152202.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152203

namespace LeftBound152214
def owner : Owner := ⟨.program ⟨257⟩, ⟨29778⟩⟩
def transferEvent : Nat := 152214
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 152212 .coefficient) (.value (.predecessor 1 152213 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152212 .coefficient)
      LeftAuthority152210.bound (LeftAuthority152210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152213 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority152210.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152210.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152210.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound152214

namespace LeftBound152218
def owner : Owner := ⟨.program ⟨257⟩, ⟨29779⟩⟩
def transferEvent : Nat := 152218
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 152216 .coefficient) (.predecessor 1 152217 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152216 .coefficient)
      LeftBound149117.bound (LeftBound149117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152217 .coefficient)
      LeftBound152214.bound (LeftBound152214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149117.bound LeftBound152214.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149117.bound, LeftBound152214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149117.actual selector witness) * (LeftBound152214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152218

namespace LeftBound152219
def owner : Owner := ⟨.program ⟨257⟩, ⟨29779⟩⟩
def transferEvent : Nat := 152219
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨29776⟩⟩]⟩ [⟨.result 152211 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152211 .coefficient)
      LeftAuthority152210.bound (LeftAuthority152210.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29776⟩⟩) (rawTerms := some (Proof.Events594.exact152211RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152210.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority152210.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152210.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound152219

namespace LeftBound152220
def owner : Owner := ⟨.program ⟨257⟩, ⟨29779⟩⟩
def transferEvent : Nat := 152220
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 152219) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 152219)
      LeftBound152219.bound (LeftBound152219.actual selector witness) := by
  exact .transfer (LeftBound152219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound152219.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound152219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound152219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152220

namespace LeftBound152315
def owner : Owner := ⟨.program ⟨257⟩, ⟨29065⟩⟩
def transferEvent : Nat := 152315
def frameStart : Nat := 152276
def rule : BoundRule := .identity (.predecessor 0 152314 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152314 .coefficient)
      LeftAuthority152312.bound (LeftAuthority152312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152312.derived selector witness)

def rawBound : CoeffClass := LeftAuthority152312.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority152312.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152315

namespace LeftBound152332
def owner : Owner := ⟨.program ⟨257⟩, ⟨30434⟩⟩
def transferEvent : Nat := 152332
def frameStart : Nat := 152276
def rule : BoundRule := .sum [.predecessor 0 152330 .coefficient, .predecessor 1 152331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152330 .coefficient)
      LeftBound152315.bound (LeftBound152315.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152331 .coefficient)
      LeftAuthority152328.bound (LeftAuthority152328.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority152328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152315.bound, LeftAuthority152328.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152315.bound, LeftAuthority152328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152315.actual selector witness, LeftAuthority152328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152332

namespace LeftBound152335
def owner : Owner := ⟨.program ⟨257⟩, ⟨30435⟩⟩
def transferEvent : Nat := 152335
def frameStart : Nat := 152276
def rule : BoundRule := .identity (.predecessor 0 152334 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152334 .coefficient)
      LeftBound152332.bound (LeftBound152332.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152332.derived selector witness)

def rawBound : CoeffClass := LeftBound152332.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound152332.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152335

namespace LeftBound152341
def owner : Owner := ⟨.program ⟨257⟩, ⟨30436⟩⟩
def transferEvent : Nat := 152341
def frameStart : Nat := 152276
def rule : BoundRule := .product (.predecessor 0 152339 .coefficient) (.predecessor 1 152340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152339 .coefficient)
      LeftAuthority152337.bound (LeftAuthority152337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152340 .coefficient)
      LeftBound152335.bound (LeftBound152335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152335.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority152337.bound LeftBound152335.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152337.bound, LeftBound152335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority152337.actual selector witness) * (LeftBound152335.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152341

namespace LeftBound152349
def owner : Owner := ⟨.program ⟨257⟩, ⟨30437⟩⟩
def transferEvent : Nat := 152349
def frameStart : Nat := 152276
def rule : BoundRule := .sum [.predecessor 0 152347 .coefficient, .predecessor 1 152348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152347 .coefficient)
      LeftAuthority152345.bound (LeftAuthority152345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152348 .coefficient)
      LeftBound152341.bound (LeftBound152341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority152345.bound, LeftBound152341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152345.bound, LeftBound152341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority152345.actual selector witness, LeftBound152341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152349

namespace LeftBound152353
def owner : Owner := ⟨.program ⟨257⟩, ⟨30895⟩⟩
def transferEvent : Nat := 152353
def frameStart : Nat := 152276
def rule : BoundRule := .product (.predecessor 0 152351 .coefficient) (.predecessor 1 152352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152351 .coefficient)
      LeftBound152349.bound (LeftBound152349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152352 .coefficient)
      LeftAuthority152326.bound (LeftAuthority152326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152349.bound LeftAuthority152326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152349.bound, LeftAuthority152326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152349.actual selector witness) * (LeftAuthority152326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152353

namespace LeftBound152364
def owner : Owner := ⟨.program ⟨257⟩, ⟨29261⟩⟩
def transferEvent : Nat := 152364
def frameStart : Nat := 152276
def rule : BoundRule := .product (.predecessor 0 152362 .coefficient) (.predecessor 1 152363 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152362 .coefficient)
      LeftAuthority152337.bound (LeftAuthority152337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152363 .coefficient)
      LeftAuthority152360.bound (LeftAuthority152360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority152337.bound LeftAuthority152360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152337.bound, LeftAuthority152360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority152337.actual selector witness) * (LeftAuthority152360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152364

namespace LeftBound152372
def owner : Owner := ⟨.program ⟨257⟩, ⟨29262⟩⟩
def transferEvent : Nat := 152372
def frameStart : Nat := 152276
def rule : BoundRule := .sum [.predecessor 0 152370 .coefficient, .predecessor 1 152371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152370 .coefficient)
      LeftAuthority152368.bound (LeftAuthority152368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152371 .coefficient)
      LeftBound152364.bound (LeftBound152364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority152368.bound, LeftBound152364.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152368.bound, LeftBound152364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority152368.actual selector witness, LeftBound152364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152372

namespace LeftBound152376
def owner : Owner := ⟨.program ⟨257⟩, ⟨30898⟩⟩
def transferEvent : Nat := 152376
def frameStart : Nat := 152276
def rule : BoundRule := .sum [.predecessor 0 152374 .coefficient, .predecessor 1 152375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152374 .coefficient)
      LeftBound152372.bound (LeftBound152372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152375 .coefficient)
      LeftBound152353.bound (LeftBound152353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152372.bound, LeftBound152353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152372.bound, LeftBound152353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152372.actual selector witness, LeftBound152353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152376

namespace LeftBound152389
def owner : Owner := ⟨.program ⟨257⟩, ⟨30897⟩⟩
def transferEvent : Nat := 152389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 152387 .coefficient, .predecessor 1 152388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152387 .coefficient)
      LeftBound152218.bound (LeftBound152218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152388 .coefficient)
      LeftBound152201.bound (LeftBound152201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152218.bound, LeftBound152201.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152218.bound, LeftBound152201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152218.actual selector witness, LeftBound152201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152389

namespace LeftBound152392
def owner : Owner := ⟨.program ⟨257⟩, ⟨30897⟩⟩
def transferEvent : Nat := 152392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 152386 .summary, .result 152208 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152386 .summary)
      LeftBound152220.bound (LeftBound152220.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29779⟩⟩) (rawTerms := some (Proof.Events595.exact152386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152208 .summary)
      LeftBound152203.bound (LeftBound152203.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30896⟩⟩) (rawTerms := some (Proof.Events594.exact152208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152220.bound, LeftBound152203.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152220.bound, LeftBound152203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152220.actual selector witness, LeftBound152203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152392

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
