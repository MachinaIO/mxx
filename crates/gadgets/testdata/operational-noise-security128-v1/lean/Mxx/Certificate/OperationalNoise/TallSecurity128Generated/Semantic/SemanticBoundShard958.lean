import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard897
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard957

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound145153
def owner : Owner := ⟨.program ⟨257⟩, ⟨43395⟩⟩
def transferEvent : Nat := 145153
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 134495 .summary) (.transfer 145152) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134495 .summary)
      LeftBound134493.bound (LeftBound134493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5473⟩⟩) (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 145152)
      LeftBound145152.bound (LeftBound145152.actual selector witness) := by
  exact .transfer (LeftBound145152.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134493.bound LeftBound145152.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134493.bound, LeftBound145152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134493.actual selector witness) * (LeftBound145152.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145153

namespace LeftBound145248
def owner : Owner := ⟨.program ⟨257⟩, ⟨42733⟩⟩
def transferEvent : Nat := 145248
def frameStart : Nat := 145209
def rule : BoundRule := .identity (.predecessor 0 145247 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145247 .coefficient)
      LeftAuthority145245.bound (LeftAuthority145245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145245.derived selector witness)

def rawBound : CoeffClass := LeftAuthority145245.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority145245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority145245.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound145248

namespace LeftBound145265
def owner : Owner := ⟨.program ⟨257⟩, ⟨44118⟩⟩
def transferEvent : Nat := 145265
def frameStart : Nat := 145209
def rule : BoundRule := .sum [.predecessor 0 145263 .coefficient, .predecessor 1 145264 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145263 .coefficient)
      LeftBound145248.bound (LeftBound145248.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound145248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145264 .coefficient)
      LeftAuthority145261.bound (LeftAuthority145261.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority145261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound145248.bound, LeftAuthority145261.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145248.bound, LeftAuthority145261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound145248.actual selector witness, LeftAuthority145261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145265

namespace LeftBound145268
def owner : Owner := ⟨.program ⟨257⟩, ⟨44119⟩⟩
def transferEvent : Nat := 145268
def frameStart : Nat := 145209
def rule : BoundRule := .identity (.predecessor 0 145267 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145267 .coefficient)
      LeftBound145265.bound (LeftBound145265.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound145265.derived selector witness)

def rawBound : CoeffClass := LeftBound145265.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound145265.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound145268

namespace LeftBound145274
def owner : Owner := ⟨.program ⟨257⟩, ⟨44120⟩⟩
def transferEvent : Nat := 145274
def frameStart : Nat := 145209
def rule : BoundRule := .product (.predecessor 0 145272 .coefficient) (.predecessor 1 145273 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145272 .coefficient)
      LeftAuthority145270.bound (LeftAuthority145270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145273 .coefficient)
      LeftBound145268.bound (LeftBound145268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145268.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority145270.bound LeftBound145268.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority145270.bound, LeftBound145268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority145270.actual selector witness) * (LeftBound145268.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145274

namespace LeftBound145282
def owner : Owner := ⟨.program ⟨257⟩, ⟨44121⟩⟩
def transferEvent : Nat := 145282
def frameStart : Nat := 145209
def rule : BoundRule := .sum [.predecessor 0 145280 .coefficient, .predecessor 1 145281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145280 .coefficient)
      LeftAuthority145278.bound (LeftAuthority145278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145281 .coefficient)
      LeftBound145274.bound (LeftBound145274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority145278.bound, LeftBound145274.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority145278.bound, LeftBound145274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority145278.actual selector witness, LeftBound145274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145282

namespace LeftBound145286
def owner : Owner := ⟨.program ⟨257⟩, ⟨44489⟩⟩
def transferEvent : Nat := 145286
def frameStart : Nat := 145209
def rule : BoundRule := .product (.predecessor 0 145284 .coefficient) (.predecessor 1 145285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145284 .coefficient)
      LeftBound145282.bound (LeftBound145282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145285 .coefficient)
      LeftAuthority145259.bound (LeftAuthority145259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145259.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound145282.bound LeftAuthority145259.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145282.bound, LeftAuthority145259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound145282.actual selector witness) * (LeftAuthority145259.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145286

namespace LeftBound145297
def owner : Owner := ⟨.program ⟨257⟩, ⟨42913⟩⟩
def transferEvent : Nat := 145297
def frameStart : Nat := 145209
def rule : BoundRule := .product (.predecessor 0 145295 .coefficient) (.predecessor 1 145296 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145295 .coefficient)
      LeftAuthority145270.bound (LeftAuthority145270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145296 .coefficient)
      LeftAuthority145293.bound (LeftAuthority145293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority145270.bound LeftAuthority145293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority145270.bound, LeftAuthority145293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority145270.actual selector witness) * (LeftAuthority145293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145297

namespace LeftBound145305
def owner : Owner := ⟨.program ⟨257⟩, ⟨42914⟩⟩
def transferEvent : Nat := 145305
def frameStart : Nat := 145209
def rule : BoundRule := .sum [.predecessor 0 145303 .coefficient, .predecessor 1 145304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145303 .coefficient)
      LeftAuthority145301.bound (LeftAuthority145301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145304 .coefficient)
      LeftBound145297.bound (LeftBound145297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority145301.bound, LeftBound145297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority145301.bound, LeftBound145297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority145301.actual selector witness, LeftBound145297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145305

namespace LeftBound145309
def owner : Owner := ⟨.program ⟨257⟩, ⟨44493⟩⟩
def transferEvent : Nat := 145309
def frameStart : Nat := 145209
def rule : BoundRule := .sum [.predecessor 0 145307 .coefficient, .predecessor 1 145308 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145307 .coefficient)
      LeftBound145305.bound (LeftBound145305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145308 .coefficient)
      LeftBound145286.bound (LeftBound145286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound145305.bound, LeftBound145286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145305.bound, LeftBound145286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound145305.actual selector witness, LeftBound145286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145309

namespace LeftBound145322
def owner : Owner := ⟨.program ⟨257⟩, ⟨44491⟩⟩
def transferEvent : Nat := 145322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 145320 .coefficient, .predecessor 1 145321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145320 .coefficient)
      LeftBound145151.bound (LeftBound145151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145321 .coefficient)
      LeftBound145134.bound (LeftBound145134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events566.exact145141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound145151.bound, LeftBound145134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145151.bound, LeftBound145134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound145151.actual selector witness, LeftBound145134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145322

namespace LeftBound145325
def owner : Owner := ⟨.program ⟨257⟩, ⟨44491⟩⟩
def transferEvent : Nat := 145325
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 145319 .summary, .result 145141 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145319 .summary)
      LeftBound145153.bound (LeftBound145153.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43395⟩⟩) (rawTerms := some (Proof.Events567.exact145319RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145141 .summary)
      LeftBound145136.bound (LeftBound145136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44490⟩⟩) (rawTerms := some (Proof.Events566.exact145141RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound145153.bound, LeftBound145136.bound]
def bound : CoeffClass := .finite ⟨32193718473625891320532869316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145153.bound, LeftBound145136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound145153.actual selector witness, LeftBound145136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound145325

namespace LeftBound145329
def owner : Owner := ⟨.program ⟨257⟩, ⟨44492⟩⟩
def transferEvent : Nat := 145329
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 145327 .coefficient) (.predecessor 1 145328 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145327 .coefficient)
      LeftBound145322.bound (LeftBound145322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145328 .coefficient)
      LeftBound15581.bound (LeftBound15581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15581.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound145322.bound LeftBound15581.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145322.bound, LeftBound15581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound145322.actual selector witness) * (LeftBound15581.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145329

namespace LeftBound145330
def owner : Owner := ⟨.program ⟨257⟩, ⟨44492⟩⟩
def transferEvent : Nat := 145330
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩ [⟨.result 15578 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15578 .coefficient)
      LeftAuthority15577.bound (LeftAuthority15577.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7153⟩⟩) (rawTerms := some (Proof.Events060.exact15578RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15577.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15577.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15577.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound145330

namespace LeftBound145331
def owner : Owner := ⟨.program ⟨257⟩, ⟨44492⟩⟩
def transferEvent : Nat := 145331
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 145326 .summary) (.transfer 145330) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145326 .summary)
      LeftBound145325.bound (LeftBound145325.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44491⟩⟩) (rawTerms := some (Proof.Events567.exact145326RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 145330)
      LeftBound145330.bound (LeftBound145330.actual selector witness) := by
  exact .transfer (LeftBound145330.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound145325.bound LeftBound145330.bound
def bound : CoeffClass := .finite ⟨345677419952135604401347317519683074129920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound145325.bound, LeftBound145330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound145325.actual selector witness) * (LeftBound145330.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145331

namespace LeftBound145346
def owner : Owner := ⟨.program ⟨257⟩, ⟨41810⟩⟩
def transferEvent : Nat := 145346
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 145344 .coefficient) (.predecessor 1 145345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 145344 .coefficient)
      LeftBound136123.bound (LeftBound136123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events531.exact136127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound136123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound136123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 145345 .coefficient)
      LeftAuthority145342.bound (LeftAuthority145342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority145342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority145342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound136123.bound LeftAuthority145342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound136123.bound, LeftAuthority145342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound136123.actual selector witness) * (LeftAuthority145342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound145346

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
