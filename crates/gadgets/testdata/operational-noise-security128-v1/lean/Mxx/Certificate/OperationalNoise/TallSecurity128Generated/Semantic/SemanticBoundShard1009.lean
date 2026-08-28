import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1008

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound152100
def owner : Owner := ⟨.program ⟨257⟩, ⟨28704⟩⟩
def transferEvent : Nat := 152100
def frameStart : Nat := 152067
def rule : BoundRule := .identity (.predecessor 0 152099 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152099 .coefficient)
      LeftBound152096.bound (LeftBound152096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152096.derived selector witness)

def rawBound : CoeffClass := LeftBound152096.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound152096.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152100

namespace LeftBound152117
def owner : Owner := ⟨.program ⟨257⟩, ⟨30354⟩⟩
def transferEvent : Nat := 152117
def frameStart : Nat := 152067
def rule : BoundRule := .sum [.predecessor 0 152115 .coefficient, .predecessor 1 152116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152115 .coefficient)
      LeftBound152100.bound (LeftBound152100.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152116 .coefficient)
      LeftAuthority152113.bound (LeftAuthority152113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority152113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152100.bound, LeftAuthority152113.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152100.bound, LeftAuthority152113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152100.actual selector witness, LeftAuthority152113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152117

namespace LeftBound152120
def owner : Owner := ⟨.program ⟨257⟩, ⟨30355⟩⟩
def transferEvent : Nat := 152120
def frameStart : Nat := 152067
def rule : BoundRule := .identity (.predecessor 0 152119 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152119 .coefficient)
      LeftBound152117.bound (LeftBound152117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152117.derived selector witness)

def rawBound : CoeffClass := LeftBound152117.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound152117.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152120

namespace LeftBound152126
def owner : Owner := ⟨.program ⟨257⟩, ⟨30356⟩⟩
def transferEvent : Nat := 152126
def frameStart : Nat := 152067
def rule : BoundRule := .product (.predecessor 0 152124 .coefficient) (.predecessor 1 152125 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152124 .coefficient)
      LeftAuthority152122.bound (LeftAuthority152122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152125 .coefficient)
      LeftBound152120.bound (LeftBound152120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority152122.bound LeftBound152120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152122.bound, LeftBound152120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority152122.actual selector witness) * (LeftBound152120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152126

namespace LeftBound152142
def owner : Owner := ⟨.program ⟨257⟩, ⟨9548⟩⟩
def transferEvent : Nat := 152142
def frameStart : Nat := 152067
def rule : BoundRule := .scale (.predecessor 0 152140 .coefficient) (.value (.predecessor 1 152141 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152140 .coefficient)
      LeftAuthority152138.bound (LeftAuthority152138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152141 .coefficient)
      LeftAuthority152129.bound (LeftAuthority152129.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority152129.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority152138.bound LeftAuthority152129.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152138.bound, LeftAuthority152129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152138.actual selector witness) * (LeftAuthority152129.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound152142

namespace LeftBound152145
def owner : Owner := ⟨.program ⟨257⟩, ⟨7296⟩⟩
def transferEvent : Nat := 152145
def frameStart : Nat := 152067
def rule : BoundRule := .identity (.predecessor 0 152144 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152144 .coefficient)
      LeftAuthority152132.bound (LeftAuthority152132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152132.derived selector witness)

def rawBound : CoeffClass := LeftAuthority152132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority152132.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152145

namespace LeftBound152149
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def transferEvent : Nat := 152149
def frameStart : Nat := 152067
def rule : BoundRule := .product (.predecessor 0 152147 .coefficient) (.predecessor 1 152148 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152147 .coefficient)
      LeftBound152145.bound (LeftBound152145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152148 .coefficient)
      LeftBound152142.bound (LeftBound152142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152145.bound LeftBound152142.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152145.bound, LeftBound152142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152145.actual selector witness) * (LeftBound152142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152149

namespace LeftBound152154
def owner : Owner := ⟨.program ⟨257⟩, ⟨30357⟩⟩
def transferEvent : Nat := 152154
def frameStart : Nat := 152067
def rule : BoundRule := .sum [.predecessor 0 152152 .coefficient, .predecessor 1 152153 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152152 .coefficient)
      LeftBound152149.bound (LeftBound152149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152153 .coefficient)
      LeftBound152126.bound (LeftBound152126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152149.bound, LeftBound152126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152149.bound, LeftBound152126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152149.actual selector witness, LeftBound152126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152154

namespace LeftBound152158
def owner : Owner := ⟨.program ⟨257⟩, ⟨30569⟩⟩
def transferEvent : Nat := 152158
def frameStart : Nat := 152067
def rule : BoundRule := .product (.predecessor 0 152156 .coefficient) (.predecessor 1 152157 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152156 .coefficient)
      LeftBound152154.bound (LeftBound152154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152154.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152157 .coefficient)
      LeftAuthority152111.bound (LeftAuthority152111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152154.bound LeftAuthority152111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152154.bound, LeftAuthority152111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152154.actual selector witness) * (LeftAuthority152111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152158

namespace LeftBound152169
def owner : Owner := ⟨.program ⟨257⟩, ⟨29066⟩⟩
def transferEvent : Nat := 152169
def frameStart : Nat := 152067
def rule : BoundRule := .product (.predecessor 0 152167 .coefficient) (.predecessor 1 152168 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152167 .coefficient)
      LeftAuthority152122.bound (LeftAuthority152122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152168 .coefficient)
      LeftAuthority152165.bound (LeftAuthority152165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority152122.bound LeftAuthority152165.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152122.bound, LeftAuthority152165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority152122.actual selector witness) * (LeftAuthority152165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152169

namespace LeftBound152177
def owner : Owner := ⟨.program ⟨257⟩, ⟨29067⟩⟩
def transferEvent : Nat := 152177
def frameStart : Nat := 152067
def rule : BoundRule := .sum [.predecessor 0 152175 .coefficient, .predecessor 1 152176 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152175 .coefficient)
      LeftAuthority152173.bound (LeftAuthority152173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152176 .coefficient)
      LeftBound152169.bound (LeftBound152169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority152173.bound, LeftBound152169.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152173.bound, LeftBound152169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority152173.actual selector witness, LeftBound152169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152177

namespace LeftBound152181
def owner : Owner := ⟨.program ⟨257⟩, ⟨30570⟩⟩
def transferEvent : Nat := 152181
def frameStart : Nat := 152067
def rule : BoundRule := .sum [.predecessor 0 152179 .coefficient, .predecessor 1 152180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152179 .coefficient)
      LeftBound152177.bound (LeftBound152177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152180 .coefficient)
      LeftBound152158.bound (LeftBound152158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152177.bound, LeftBound152158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152177.bound, LeftBound152158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152177.actual selector witness, LeftBound152158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152181

namespace LeftBound152194
def owner : Owner := ⟨.program ⟨257⟩, ⟨30568⟩⟩
def transferEvent : Nat := 152194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 152192 .coefficient, .predecessor 1 152193 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152192 .coefficient)
      LeftBound152015.bound (LeftBound152015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152193 .coefficient)
      LeftBound151998.bound (LeftBound151998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact152005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152015.bound, LeftBound151998.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152015.bound, LeftBound151998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152015.actual selector witness, LeftBound151998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152194

namespace LeftBound152197
def owner : Owner := ⟨.program ⟨257⟩, ⟨30568⟩⟩
def transferEvent : Nat := 152197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 152191 .summary, .result 152005 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152191 .summary)
      LeftBound152017.bound (LeftBound152017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29502⟩⟩) (rawTerms := some (Proof.Events594.exact152191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152017.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152005 .summary)
      LeftBound152000.bound (LeftBound152000.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30567⟩⟩) (rawTerms := some (Proof.Events593.exact152005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152017.bound, LeftBound152000.bound]
def bound : CoeffClass := .finite ⟨2998127310542407467008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152017.bound, LeftBound152000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152017.actual selector witness, LeftBound152000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152197

namespace LeftBound152201
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def transferEvent : Nat := 152201
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 152199 .coefficient) (.predecessor 1 152200 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152199 .coefficient)
      LeftBound152194.bound (LeftBound152194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events594.exact152198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152200 .coefficient)
      LeftAuthority151920.bound (LeftAuthority151920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152194.bound LeftAuthority151920.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152194.bound, LeftAuthority151920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152194.actual selector witness) * (LeftAuthority151920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152201

namespace LeftBound152202
def owner : Owner := ⟨.program ⟨257⟩, ⟨30896⟩⟩
def transferEvent : Nat := 152202
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨30894⟩⟩]⟩ [⟨.result 151921 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151921 .coefficient)
      LeftAuthority151920.bound (LeftAuthority151920.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨30894⟩⟩) (rawTerms := some (Proof.Events593.exact151921RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151920.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority151920.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority151920.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound152202

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
