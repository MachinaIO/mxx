import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1850

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound273197
def owner : Owner := ⟨.program ⟨257⟩, ⟨33276⟩⟩
def transferEvent : Nat := 273197
def frameStart : Nat := 273132
def rule : BoundRule := .product (.predecessor 0 273195 .coefficient) (.predecessor 1 273196 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273195 .coefficient)
      LeftAuthority273193.bound (LeftAuthority273193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273196 .coefficient)
      LeftBound273191.bound (LeftBound273191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273191.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority273193.bound LeftBound273191.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority273193.bound, LeftBound273191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority273193.actual selector witness) * (LeftBound273191.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273197

namespace LeftBound273205
def owner : Owner := ⟨.program ⟨257⟩, ⟨33277⟩⟩
def transferEvent : Nat := 273205
def frameStart : Nat := 273132
def rule : BoundRule := .sum [.predecessor 0 273203 .coefficient, .predecessor 1 273204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273203 .coefficient)
      LeftAuthority273201.bound (LeftAuthority273201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273201.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273204 .coefficient)
      LeftBound273197.bound (LeftBound273197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority273201.bound, LeftBound273197.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority273201.bound, LeftBound273197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority273201.actual selector witness, LeftBound273197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273205

namespace LeftBound273209
def owner : Owner := ⟨.program ⟨257⟩, ⟨33636⟩⟩
def transferEvent : Nat := 273209
def frameStart : Nat := 273132
def rule : BoundRule := .product (.predecessor 0 273207 .coefficient) (.predecessor 1 273208 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273207 .coefficient)
      LeftBound273205.bound (LeftBound273205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273205.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273208 .coefficient)
      LeftAuthority273182.bound (LeftAuthority273182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound273205.bound LeftAuthority273182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273205.bound, LeftAuthority273182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound273205.actual selector witness) * (LeftAuthority273182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273209

namespace LeftBound273220
def owner : Owner := ⟨.program ⟨257⟩, ⟨31951⟩⟩
def transferEvent : Nat := 273220
def frameStart : Nat := 273132
def rule : BoundRule := .product (.predecessor 0 273218 .coefficient) (.predecessor 1 273219 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273218 .coefficient)
      LeftAuthority273193.bound (LeftAuthority273193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273219 .coefficient)
      LeftAuthority273216.bound (LeftAuthority273216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273216.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority273193.bound LeftAuthority273216.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority273193.bound, LeftAuthority273216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority273193.actual selector witness) * (LeftAuthority273216.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273220

namespace LeftBound273228
def owner : Owner := ⟨.program ⟨257⟩, ⟨31952⟩⟩
def transferEvent : Nat := 273228
def frameStart : Nat := 273132
def rule : BoundRule := .sum [.predecessor 0 273226 .coefficient, .predecessor 1 273227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273226 .coefficient)
      LeftAuthority273224.bound (LeftAuthority273224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority273224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority273224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273227 .coefficient)
      LeftBound273220.bound (LeftBound273220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority273224.bound, LeftBound273220.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority273224.bound, LeftBound273220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority273224.actual selector witness, LeftBound273220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273228

namespace LeftBound273232
def owner : Owner := ⟨.program ⟨257⟩, ⟨33640⟩⟩
def transferEvent : Nat := 273232
def frameStart : Nat := 273132
def rule : BoundRule := .sum [.predecessor 0 273230 .coefficient, .predecessor 1 273231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273230 .coefficient)
      LeftBound273228.bound (LeftBound273228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273231 .coefficient)
      LeftBound273209.bound (LeftBound273209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273228.bound, LeftBound273209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273228.bound, LeftBound273209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273228.actual selector witness, LeftBound273209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273232

namespace LeftBound273245
def owner : Owner := ⟨.program ⟨257⟩, ⟨33638⟩⟩
def transferEvent : Nat := 273245
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273243 .coefficient, .predecessor 1 273244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273243 .coefficient)
      LeftBound273074.bound (LeftBound273074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273244 .coefficient)
      LeftBound273057.bound (LeftBound273057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1066.exact273064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273074.bound, LeftBound273057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273074.bound, LeftBound273057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273074.actual selector witness, LeftBound273057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273245

namespace LeftBound273248
def owner : Owner := ⟨.program ⟨257⟩, ⟨33638⟩⟩
def transferEvent : Nat := 273248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 273242 .summary, .result 273064 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273242 .summary)
      LeftBound273076.bound (LeftBound273076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32533⟩⟩) (rawTerms := some (Proof.Events1067.exact273242RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273064 .summary)
      LeftBound273059.bound (LeftBound273059.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33637⟩⟩) (rawTerms := some (Proof.Events1066.exact273064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273076.bound, LeftBound273059.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273076.bound, LeftBound273059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273076.actual selector witness, LeftBound273059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273248

namespace LeftBound273272
def owner : Owner := ⟨.program ⟨257⟩, ⟨21297⟩⟩
def transferEvent : Nat := 273272
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 273270 .coefficient) (.predecessor 1 273271 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273270 .coefficient)
      LeftAuthority13154.bound (LeftAuthority13154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273271 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13154.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13154.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13154.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound273272

namespace LeftBound273277
def owner : Owner := ⟨.program ⟨257⟩, ⟨7662⟩⟩
def transferEvent : Nat := 273277
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273275 .coefficient) (.predecessor 1 273276 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273275 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273276 .coefficient)
      LeftBound24594.bound (LeftBound24594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound24594.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound24594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound24594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273277

namespace LeftBound273282
def owner : Owner := ⟨.program ⟨257⟩, ⟨21298⟩⟩
def transferEvent : Nat := 273282
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273280 .coefficient, .predecessor 1 273281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273280 .coefficient)
      LeftBound273277.bound (LeftBound273277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273281 .coefficient)
      LeftBound273272.bound (LeftBound273272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273277.bound, LeftBound273272.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273277.bound, LeftBound273272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273277.actual selector witness, LeftBound273272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273282

namespace LeftBound273286
def owner : Owner := ⟨.program ⟨257⟩, ⟨21299⟩⟩
def transferEvent : Nat := 273286
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273284 .coefficient, .predecessor 1 273285 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273284 .coefficient)
      LeftBound273282.bound (LeftBound273282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273285 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273282.bound, LeftBound24586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273282.bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273282.actual selector witness, LeftBound24586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273286

namespace LeftBound273287
def owner : Owner := ⟨.program ⟨257⟩, ⟨21299⟩⟩
def transferEvent : Nat := 273287
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩ [⟨.result 24587 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24587 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨132⟩⟩) (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24586.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24586.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound273287

namespace LeftBound273292
def owner : Owner := ⟨.program ⟨257⟩, ⟨21300⟩⟩
def transferEvent : Nat := 273292
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273290 .coefficient) (.predecessor 1 273291 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273290 .coefficient)
      LeftBound273286.bound (LeftBound273286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1067.exact273289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273291 .coefficient)
      LeftAuthority13157.bound (LeftAuthority13157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13157.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound273286.bound LeftAuthority13157.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273286.bound, LeftAuthority13157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound273286.actual selector witness) * (LeftAuthority13157.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273292

namespace LeftBound273293
def owner : Owner := ⟨.program ⟨257⟩, ⟨21300⟩⟩
def transferEvent : Nat := 273293
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨20976⟩⟩], []⟩ [⟨.result 13158 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 13158 .coefficient)
      LeftAuthority13157.bound (LeftAuthority13157.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20976⟩⟩) (rawTerms := some (Proof.Events051.exact13158RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13157.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13157.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority13157.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound273293

namespace LeftBound273294
def owner : Owner := ⟨.program ⟨257⟩, ⟨21300⟩⟩
def transferEvent : Nat := 273294
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 273289 .summary) (.transfer 273293) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273289 .summary)
      LeftBound273287.bound (LeftBound273287.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21299⟩⟩) (rawTerms := some (Proof.Events1067.exact273289RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 273293)
      LeftBound273293.bound (LeftBound273293.actual selector witness) := by
  exact .transfer (LeftBound273293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound273287.bound LeftBound273293.bound
def bound : CoeffClass := .finite ⟨3407872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273287.bound, LeftBound273293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound273287.actual selector witness) * (LeftBound273293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273294

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
