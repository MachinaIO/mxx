import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard115

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23185
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def transferEvent : Nat := 23185
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨54342⟩⟩]⟩ [⟨.result 23177 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23177 .coefficient)
      LeftAuthority23176.bound (LeftAuthority23176.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54342⟩⟩) (rawTerms := some (Proof.Events090.exact23177RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23176.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23176.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority23176.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23185

namespace LeftBound23186
def owner : Owner := ⟨.program ⟨257⟩, ⟨54345⟩⟩
def transferEvent : Nat := 23186
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 23185) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 23185)
      LeftBound23185.bound (LeftBound23185.actual selector witness) := by
  exact .transfer (LeftBound23185.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound23185.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound23185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound23185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23186

namespace LeftBound23265
def owner : Owner := ⟨.program ⟨257⟩, ⟨53292⟩⟩
def transferEvent : Nat := 23265
def frameStart : Nat := 23236
def rule : BoundRule := .product (.predecessor 0 23263 .coefficient) (.predecessor 1 23264 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23263 .coefficient)
      LeftAuthority23261.bound (LeftAuthority23261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23264 .coefficient)
      LeftAuthority23258.bound (LeftAuthority23258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23258.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23261.bound LeftAuthority23258.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23261.bound, LeftAuthority23258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority23261.actual selector witness) * (LeftAuthority23258.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23265

namespace LeftBound23269
def owner : Owner := ⟨.program ⟨257⟩, ⟨53293⟩⟩
def transferEvent : Nat := 23269
def frameStart : Nat := 23236
def rule : BoundRule := .identity (.predecessor 0 23268 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23268 .coefficient)
      LeftBound23265.bound (LeftBound23265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23265.derived selector witness)

def rawBound : CoeffClass := LeftBound23265.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound23265.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23269

namespace LeftBound23286
def owner : Owner := ⟨.program ⟨257⟩, ⟨55230⟩⟩
def transferEvent : Nat := 23286
def frameStart : Nat := 23236
def rule : BoundRule := .sum [.predecessor 0 23284 .coefficient, .predecessor 1 23285 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23284 .coefficient)
      LeftBound23269.bound (LeftBound23269.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23285 .coefficient)
      LeftAuthority23282.bound (LeftAuthority23282.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23269.bound, LeftAuthority23282.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23269.bound, LeftAuthority23282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound23269.actual selector witness, LeftAuthority23282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23286

namespace LeftBound23289
def owner : Owner := ⟨.program ⟨257⟩, ⟨55231⟩⟩
def transferEvent : Nat := 23289
def frameStart : Nat := 23236
def rule : BoundRule := .identity (.predecessor 0 23288 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23288 .coefficient)
      LeftBound23286.bound (LeftBound23286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23286.derived selector witness)

def rawBound : CoeffClass := LeftBound23286.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound23286.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23289

namespace LeftBound23295
def owner : Owner := ⟨.program ⟨257⟩, ⟨55232⟩⟩
def transferEvent : Nat := 23295
def frameStart : Nat := 23236
def rule : BoundRule := .product (.predecessor 0 23293 .coefficient) (.predecessor 1 23294 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23293 .coefficient)
      LeftAuthority23291.bound (LeftAuthority23291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23291.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23294 .coefficient)
      LeftBound23289.bound (LeftBound23289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority23291.bound LeftBound23289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23291.bound, LeftBound23289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority23291.actual selector witness) * (LeftBound23289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23295

namespace LeftBound23311
def owner : Owner := ⟨.program ⟨257⟩, ⟨9530⟩⟩
def transferEvent : Nat := 23311
def frameStart : Nat := 23236
def rule : BoundRule := .scale (.predecessor 0 23309 .coefficient) (.value (.predecessor 1 23310 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23309 .coefficient)
      LeftAuthority23307.bound (LeftAuthority23307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23310 .coefficient)
      LeftAuthority23298.bound (LeftAuthority23298.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23298.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23307.bound LeftAuthority23298.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23307.bound, LeftAuthority23298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority23307.actual selector witness) * (LeftAuthority23298.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23311

namespace LeftBound23314
def owner : Owner := ⟨.program ⟨257⟩, ⟨7289⟩⟩
def transferEvent : Nat := 23314
def frameStart : Nat := 23236
def rule : BoundRule := .identity (.predecessor 0 23313 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23313 .coefficient)
      LeftAuthority23301.bound (LeftAuthority23301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23301.derived selector witness)

def rawBound : CoeffClass := LeftAuthority23301.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority23301.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23314

namespace LeftBound23318
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def transferEvent : Nat := 23318
def frameStart : Nat := 23236
def rule : BoundRule := .product (.predecessor 0 23316 .coefficient) (.predecessor 1 23317 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23316 .coefficient)
      LeftBound23314.bound (LeftBound23314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23317 .coefficient)
      LeftBound23311.bound (LeftBound23311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound23314.bound LeftBound23311.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23314.bound, LeftBound23311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound23314.actual selector witness) * (LeftBound23311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23318

namespace LeftBound23323
def owner : Owner := ⟨.program ⟨257⟩, ⟨55233⟩⟩
def transferEvent : Nat := 23323
def frameStart : Nat := 23236
def rule : BoundRule := .sum [.predecessor 0 23321 .coefficient, .predecessor 1 23322 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23321 .coefficient)
      LeftBound23318.bound (LeftBound23318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23322 .coefficient)
      LeftBound23295.bound (LeftBound23295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23295.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23318.bound, LeftBound23295.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23318.bound, LeftBound23295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound23318.actual selector witness, LeftBound23295.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23323

namespace LeftBound23327
def owner : Owner := ⟨.program ⟨257⟩, ⟨55406⟩⟩
def transferEvent : Nat := 23327
def frameStart : Nat := 23236
def rule : BoundRule := .product (.predecessor 0 23325 .coefficient) (.predecessor 1 23326 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23325 .coefficient)
      LeftBound23323.bound (LeftBound23323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23326 .coefficient)
      LeftAuthority23280.bound (LeftAuthority23280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound23323.bound LeftAuthority23280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23323.bound, LeftAuthority23280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound23323.actual selector witness) * (LeftAuthority23280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23327

namespace LeftBound23338
def owner : Owner := ⟨.program ⟨257⟩, ⟨53800⟩⟩
def transferEvent : Nat := 23338
def frameStart : Nat := 23236
def rule : BoundRule := .product (.predecessor 0 23336 .coefficient) (.predecessor 1 23337 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23336 .coefficient)
      LeftAuthority23291.bound (LeftAuthority23291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23291.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23337 .coefficient)
      LeftAuthority23334.bound (LeftAuthority23334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23334.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23291.bound LeftAuthority23334.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23291.bound, LeftAuthority23334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority23291.actual selector witness) * (LeftAuthority23334.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23338

namespace LeftBound23346
def owner : Owner := ⟨.program ⟨257⟩, ⟨53801⟩⟩
def transferEvent : Nat := 23346
def frameStart : Nat := 23236
def rule : BoundRule := .sum [.predecessor 0 23344 .coefficient, .predecessor 1 23345 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23344 .coefficient)
      LeftAuthority23342.bound (LeftAuthority23342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23345 .coefficient)
      LeftBound23338.bound (LeftBound23338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23342.bound, LeftBound23338.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23342.bound, LeftBound23338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority23342.actual selector witness, LeftBound23338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23346

namespace LeftBound23350
def owner : Owner := ⟨.program ⟨257⟩, ⟨55407⟩⟩
def transferEvent : Nat := 23350
def frameStart : Nat := 23236
def rule : BoundRule := .sum [.predecessor 0 23348 .coefficient, .predecessor 1 23349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23348 .coefficient)
      LeftBound23346.bound (LeftBound23346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23349 .coefficient)
      LeftBound23327.bound (LeftBound23327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23346.bound, LeftBound23327.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23346.bound, LeftBound23327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound23346.actual selector witness, LeftBound23327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23350

namespace LeftBound23363
def owner : Owner := ⟨.program ⟨257⟩, ⟨55405⟩⟩
def transferEvent : Nat := 23363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23361 .coefficient, .predecessor 1 23362 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 23361 .coefficient)
      LeftBound23184.bound (LeftBound23184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 23362 .coefficient)
      LeftBound23167.bound (LeftBound23167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23184.bound, LeftBound23167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23184.bound, LeftBound23167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound23184.actual selector witness, LeftBound23167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23363

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
