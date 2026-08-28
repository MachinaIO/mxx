import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1805

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound267172
def owner : Owner := ⟨.program ⟨257⟩, ⟨42276⟩⟩
def transferEvent : Nat := 267172
def frameStart : Nat := 267139
def rule : BoundRule := .identity (.predecessor 0 267171 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267171 .coefficient)
      LeftBound267168.bound (LeftBound267168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267168.derived selector witness)

def rawBound : CoeffClass := LeftBound267168.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound267168.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound267172

namespace LeftBound267189
def owner : Owner := ⟨.program ⟨257⟩, ⟨44034⟩⟩
def transferEvent : Nat := 267189
def frameStart : Nat := 267139
def rule : BoundRule := .sum [.predecessor 0 267187 .coefficient, .predecessor 1 267188 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267187 .coefficient)
      LeftBound267172.bound (LeftBound267172.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound267172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267188 .coefficient)
      LeftAuthority267185.bound (LeftAuthority267185.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority267185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267172.bound, LeftAuthority267185.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267172.bound, LeftAuthority267185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267172.actual selector witness, LeftAuthority267185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267189

namespace LeftBound267192
def owner : Owner := ⟨.program ⟨257⟩, ⟨44035⟩⟩
def transferEvent : Nat := 267192
def frameStart : Nat := 267139
def rule : BoundRule := .identity (.predecessor 0 267191 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267191 .coefficient)
      LeftBound267189.bound (LeftBound267189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound267189.derived selector witness)

def rawBound : CoeffClass := LeftBound267189.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound267189.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound267192

namespace LeftBound267198
def owner : Owner := ⟨.program ⟨257⟩, ⟨44036⟩⟩
def transferEvent : Nat := 267198
def frameStart : Nat := 267139
def rule : BoundRule := .product (.predecessor 0 267196 .coefficient) (.predecessor 1 267197 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267196 .coefficient)
      LeftAuthority267194.bound (LeftAuthority267194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267197 .coefficient)
      LeftBound267192.bound (LeftBound267192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority267194.bound LeftBound267192.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267194.bound, LeftBound267192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority267194.actual selector witness) * (LeftBound267192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267198

namespace LeftBound267214
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 267214
def frameStart : Nat := 267139
def rule : BoundRule := .scale (.predecessor 0 267212 .coefficient) (.value (.predecessor 1 267213 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267212 .coefficient)
      LeftAuthority267210.bound (LeftAuthority267210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267213 .coefficient)
      LeftAuthority267201.bound (LeftAuthority267201.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority267201.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority267210.bound LeftAuthority267201.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267210.bound, LeftAuthority267201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority267210.actual selector witness) * (LeftAuthority267201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound267214

namespace LeftBound267217
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 267217
def frameStart : Nat := 267139
def rule : BoundRule := .identity (.predecessor 0 267216 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267216 .coefficient)
      LeftAuthority267204.bound (LeftAuthority267204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267204.derived selector witness)

def rawBound : CoeffClass := LeftAuthority267204.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority267204.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound267217

namespace LeftBound267221
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 267221
def frameStart : Nat := 267139
def rule : BoundRule := .product (.predecessor 0 267219 .coefficient) (.predecessor 1 267220 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267219 .coefficient)
      LeftBound267217.bound (LeftBound267217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267220 .coefficient)
      LeftBound267214.bound (LeftBound267214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound267217.bound LeftBound267214.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267217.bound, LeftBound267214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound267217.actual selector witness) * (LeftBound267214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267221

namespace LeftBound267226
def owner : Owner := ⟨.program ⟨257⟩, ⟨44037⟩⟩
def transferEvent : Nat := 267226
def frameStart : Nat := 267139
def rule : BoundRule := .sum [.predecessor 0 267224 .coefficient, .predecessor 1 267225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267224 .coefficient)
      LeftBound267221.bound (LeftBound267221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267225 .coefficient)
      LeftBound267198.bound (LeftBound267198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267198.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267221.bound, LeftBound267198.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267221.bound, LeftBound267198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267221.actual selector witness, LeftBound267198.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267226

namespace LeftBound267230
def owner : Owner := ⟨.program ⟨257⟩, ⟨44211⟩⟩
def transferEvent : Nat := 267230
def frameStart : Nat := 267139
def rule : BoundRule := .product (.predecessor 0 267228 .coefficient) (.predecessor 1 267229 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267228 .coefficient)
      LeftBound267226.bound (LeftBound267226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267229 .coefficient)
      LeftAuthority267183.bound (LeftAuthority267183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound267226.bound LeftAuthority267183.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267226.bound, LeftAuthority267183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound267226.actual selector witness) * (LeftAuthority267183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267230

namespace LeftBound267241
def owner : Owner := ⟨.program ⟨257⟩, ⟨42724⟩⟩
def transferEvent : Nat := 267241
def frameStart : Nat := 267139
def rule : BoundRule := .product (.predecessor 0 267239 .coefficient) (.predecessor 1 267240 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267239 .coefficient)
      LeftAuthority267194.bound (LeftAuthority267194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267240 .coefficient)
      LeftAuthority267237.bound (LeftAuthority267237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267237.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority267194.bound LeftAuthority267237.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267194.bound, LeftAuthority267237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority267194.actual selector witness) * (LeftAuthority267237.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267241

namespace LeftBound267249
def owner : Owner := ⟨.program ⟨257⟩, ⟨42725⟩⟩
def transferEvent : Nat := 267249
def frameStart : Nat := 267139
def rule : BoundRule := .sum [.predecessor 0 267247 .coefficient, .predecessor 1 267248 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267247 .coefficient)
      LeftAuthority267245.bound (LeftAuthority267245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267245.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267248 .coefficient)
      LeftBound267241.bound (LeftBound267241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267241.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority267245.bound, LeftBound267241.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267245.bound, LeftBound267241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority267245.actual selector witness, LeftBound267241.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267249

namespace LeftBound267253
def owner : Owner := ⟨.program ⟨257⟩, ⟨44212⟩⟩
def transferEvent : Nat := 267253
def frameStart : Nat := 267139
def rule : BoundRule := .sum [.predecessor 0 267251 .coefficient, .predecessor 1 267252 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267251 .coefficient)
      LeftBound267249.bound (LeftBound267249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267252 .coefficient)
      LeftBound267230.bound (LeftBound267230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267249.bound, LeftBound267230.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267249.bound, LeftBound267230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267249.actual selector witness, LeftBound267230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267253

namespace LeftBound267266
def owner : Owner := ⟨.program ⟨257⟩, ⟨44210⟩⟩
def transferEvent : Nat := 267266
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 267264 .coefficient, .predecessor 1 267265 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267264 .coefficient)
      LeftBound267087.bound (LeftBound267087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267265 .coefficient)
      LeftBound267070.bound (LeftBound267070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1043.exact267077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267087.bound, LeftBound267070.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267087.bound, LeftBound267070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267087.actual selector witness, LeftBound267070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267266

namespace LeftBound267269
def owner : Owner := ⟨.program ⟨257⟩, ⟨44210⟩⟩
def transferEvent : Nat := 267269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 267263 .summary, .result 267077 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267263 .summary)
      LeftBound267089.bound (LeftBound267089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43149⟩⟩) (rawTerms := some (Proof.Events1043.exact267263RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound267089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267077 .summary)
      LeftBound267072.bound (LeftBound267072.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44209⟩⟩) (rawTerms := some (Proof.Events1043.exact267077RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound267072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267089.bound, LeftBound267072.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267089.bound, LeftBound267072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267089.actual selector witness, LeftBound267072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267269

namespace LeftBound267273
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def transferEvent : Nat := 267273
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 267271 .coefficient) (.predecessor 1 267272 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267271 .coefficient)
      LeftBound267266.bound (LeftBound267266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267272 .coefficient)
      LeftAuthority266992.bound (LeftAuthority266992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1042.exact266993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority266992.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority266992.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound267266.bound LeftAuthority266992.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267266.bound, LeftAuthority266992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound267266.actual selector witness) * (LeftAuthority266992.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267273

namespace LeftBound267274
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def transferEvent : Nat := 267274
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩ [⟨.result 266993 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266993 .coefficient)
      LeftAuthority266992.bound (LeftAuthority266992.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44462⟩⟩) (rawTerms := some (Proof.Events1042.exact266993RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority266992.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority266992.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority266992.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority266992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority266992.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound267274

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
