import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard390
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard450

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72143
def owner : Owner := ⟨.program ⟨257⟩, ⟨44175⟩⟩
def transferEvent : Nat := 72143
def frameStart : Nat := 72084
def rule : BoundRule := .identity (.predecessor 0 72142 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72142 .coefficient)
      LeftBound72140.bound (LeftBound72140.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72140.derived selector witness)

def rawBound : CoeffClass := LeftBound72140.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound72140.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72143

namespace LeftBound72149
def owner : Owner := ⟨.program ⟨257⟩, ⟨44176⟩⟩
def transferEvent : Nat := 72149
def frameStart : Nat := 72084
def rule : BoundRule := .product (.predecessor 0 72147 .coefficient) (.predecessor 1 72148 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72147 .coefficient)
      LeftAuthority72145.bound (LeftAuthority72145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72148 .coefficient)
      LeftBound72143.bound (LeftBound72143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority72145.bound LeftBound72143.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72145.bound, LeftBound72143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority72145.actual selector witness) * (LeftBound72143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72149

namespace LeftBound72157
def owner : Owner := ⟨.program ⟨257⟩, ⟨44177⟩⟩
def transferEvent : Nat := 72157
def frameStart : Nat := 72084
def rule : BoundRule := .sum [.predecessor 0 72155 .coefficient, .predecessor 1 72156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72155 .coefficient)
      LeftAuthority72153.bound (LeftAuthority72153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72156 .coefficient)
      LeftBound72149.bound (LeftBound72149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72153.bound, LeftBound72149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72153.bound, LeftBound72149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority72153.actual selector witness, LeftBound72149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72157

namespace LeftBound72161
def owner : Owner := ⟨.program ⟨257⟩, ⟨44839⟩⟩
def transferEvent : Nat := 72161
def frameStart : Nat := 72084
def rule : BoundRule := .product (.predecessor 0 72159 .coefficient) (.predecessor 1 72160 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72159 .coefficient)
      LeftBound72157.bound (LeftBound72157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72160 .coefficient)
      LeftAuthority72134.bound (LeftAuthority72134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72134.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound72157.bound LeftAuthority72134.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72157.bound, LeftAuthority72134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound72157.actual selector witness) * (LeftAuthority72134.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72161

namespace LeftBound72172
def owner : Owner := ⟨.program ⟨257⟩, ⟨43095⟩⟩
def transferEvent : Nat := 72172
def frameStart : Nat := 72084
def rule : BoundRule := .product (.predecessor 0 72170 .coefficient) (.predecessor 1 72171 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72170 .coefficient)
      LeftAuthority72145.bound (LeftAuthority72145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72171 .coefficient)
      LeftAuthority72168.bound (LeftAuthority72168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72168.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72145.bound LeftAuthority72168.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72145.bound, LeftAuthority72168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority72145.actual selector witness) * (LeftAuthority72168.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72172

namespace LeftBound72180
def owner : Owner := ⟨.program ⟨257⟩, ⟨43096⟩⟩
def transferEvent : Nat := 72180
def frameStart : Nat := 72084
def rule : BoundRule := .sum [.predecessor 0 72178 .coefficient, .predecessor 1 72179 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72178 .coefficient)
      LeftAuthority72176.bound (LeftAuthority72176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72179 .coefficient)
      LeftBound72172.bound (LeftBound72172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72172.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72176.bound, LeftBound72172.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72176.bound, LeftBound72172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority72176.actual selector witness, LeftBound72172.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72180

namespace LeftBound72184
def owner : Owner := ⟨.program ⟨257⟩, ⟨44843⟩⟩
def transferEvent : Nat := 72184
def frameStart : Nat := 72084
def rule : BoundRule := .sum [.predecessor 0 72182 .coefficient, .predecessor 1 72183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72182 .coefficient)
      LeftBound72180.bound (LeftBound72180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72183 .coefficient)
      LeftBound72161.bound (LeftBound72161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72180.bound, LeftBound72161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72180.bound, LeftBound72161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound72180.actual selector witness, LeftBound72161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72184

namespace LeftBound72197
def owner : Owner := ⟨.program ⟨257⟩, ⟨44841⟩⟩
def transferEvent : Nat := 72197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72195 .coefficient, .predecessor 1 72196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72195 .coefficient)
      LeftBound72026.bound (LeftBound72026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72196 .coefficient)
      LeftBound72009.bound (LeftBound72009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72026.bound, LeftBound72009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72026.bound, LeftBound72009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound72026.actual selector witness, LeftBound72009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72197

namespace LeftBound72200
def owner : Owner := ⟨.program ⟨257⟩, ⟨44841⟩⟩
def transferEvent : Nat := 72200
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72194 .summary, .result 72016 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72194 .summary)
      LeftBound72028.bound (LeftBound72028.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43675⟩⟩) (rawTerms := some (Proof.Events282.exact72194RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72016 .summary)
      LeftBound72011.bound (LeftBound72011.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44840⟩⟩) (rawTerms := some (Proof.Events281.exact72016RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72028.bound, LeftBound72011.bound]
def bound : CoeffClass := .finite ⟨32193718473625891320532869316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72028.bound, LeftBound72011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound72028.actual selector witness, LeftBound72011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72200

namespace LeftBound72204
def owner : Owner := ⟨.program ⟨257⟩, ⟨44842⟩⟩
def transferEvent : Nat := 72204
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72202 .coefficient) (.predecessor 1 72203 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72202 .coefficient)
      LeftBound72197.bound (LeftBound72197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72203 .coefficient)
      LeftBound15581.bound (LeftBound15581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15581.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound72197.bound LeftBound15581.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72197.bound, LeftBound15581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound72197.actual selector witness) * (LeftBound15581.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72204

namespace LeftBound72205
def owner : Owner := ⟨.program ⟨257⟩, ⟨44842⟩⟩
def transferEvent : Nat := 72205
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
end LeftBound72205

namespace LeftBound72206
def owner : Owner := ⟨.program ⟨257⟩, ⟨44842⟩⟩
def transferEvent : Nat := 72206
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72201 .summary) (.transfer 72205) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72201 .summary)
      LeftBound72200.bound (LeftBound72200.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44841⟩⟩) (rawTerms := some (Proof.Events282.exact72201RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 72205)
      LeftBound72205.bound (LeftBound72205.actual selector witness) := by
  exact .transfer (LeftBound72205.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound72200.bound LeftBound72205.bound
def bound : CoeffClass := .finite ⟨345677419952135604401347317519683074129920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72200.bound, LeftBound72205.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound72200.actual selector witness) * (LeftBound72205.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72206

namespace LeftBound72221
def owner : Owner := ⟨.program ⟨257⟩, ⟨42160⟩⟩
def transferEvent : Nat := 72221
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72219 .coefficient) (.predecessor 1 72220 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72219 .coefficient)
      LeftBound62998.bound (LeftBound62998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72220 .coefficient)
      LeftAuthority72217.bound (LeftAuthority72217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound62998.bound LeftAuthority72217.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62998.bound, LeftAuthority72217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound62998.actual selector witness) * (LeftAuthority72217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72221

namespace LeftBound72222
def owner : Owner := ⟨.program ⟨257⟩, ⟨42160⟩⟩
def transferEvent : Nat := 72222
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨42158⟩⟩]⟩ [⟨.result 72218 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 72218 .coefficient)
      LeftAuthority72217.bound (LeftAuthority72217.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42158⟩⟩) (rawTerms := some (Proof.Events282.exact72218RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72217.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72217.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority72217.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72222

namespace LeftBound72223
def owner : Owner := ⟨.program ⟨257⟩, ⟨42160⟩⟩
def transferEvent : Nat := 72223
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63002 .summary) (.transfer 72222) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63002 .summary)
      LeftBound63001.bound (LeftBound63001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41698⟩⟩) (rawTerms := some (Proof.Events246.exact63002RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 72222)
      LeftBound72222.bound (LeftBound72222.actual selector witness) := by
  exact .transfer (LeftBound72222.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63001.bound LeftBound72222.bound
def bound : CoeffClass := .finite ⟨32193129122288627115968346193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63001.bound, LeftBound72222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63001.actual selector witness) * (LeftBound72222.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72223

namespace LeftBound72234
def owner : Owner := ⟨.program ⟨257⟩, ⟨40994⟩⟩
def transferEvent : Nat := 72234
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 72232 .coefficient) (.value (.predecessor 1 72233 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 72232 .coefficient)
      LeftAuthority72230.bound (LeftAuthority72230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 72233 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72230.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72230.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority72230.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72234

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
