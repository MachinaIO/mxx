import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard133
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1798
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1857

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound274040
def owner : Owner := ⟨.program ⟨257⟩, ⟨19293⟩⟩
def transferEvent : Nat := 274040
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 266120 .summary) (.transfer 274039) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266120 .summary)
      LeftBound266118.bound (LeftBound266118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5449⟩⟩) (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound266118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 274039)
      LeftBound274039.bound (LeftBound274039.actual selector witness) := by
  exact .transfer (LeftBound274039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266118.bound LeftBound274039.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266118.bound, LeftBound274039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266118.actual selector witness) * (LeftBound274039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274040

namespace LeftBound274135
def owner : Owner := ⟨.program ⟨257⟩, ⟨18523⟩⟩
def transferEvent : Nat := 274135
def frameStart : Nat := 274096
def rule : BoundRule := .identity (.predecessor 0 274134 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274134 .coefficient)
      LeftAuthority274132.bound (LeftAuthority274132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274132.derived selector witness)

def rawBound : CoeffClass := LeftAuthority274132.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority274132.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound274135

namespace LeftBound274152
def owner : Owner := ⟨.program ⟨257⟩, ⟨20034⟩⟩
def transferEvent : Nat := 274152
def frameStart : Nat := 274096
def rule : BoundRule := .sum [.predecessor 0 274150 .coefficient, .predecessor 1 274151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274150 .coefficient)
      LeftBound274135.bound (LeftBound274135.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound274135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274151 .coefficient)
      LeftAuthority274148.bound (LeftAuthority274148.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority274148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274135.bound, LeftAuthority274148.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274135.bound, LeftAuthority274148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274135.actual selector witness, LeftAuthority274148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274152

namespace LeftBound274155
def owner : Owner := ⟨.program ⟨257⟩, ⟨20035⟩⟩
def transferEvent : Nat := 274155
def frameStart : Nat := 274096
def rule : BoundRule := .identity (.predecessor 0 274154 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274154 .coefficient)
      LeftBound274152.bound (LeftBound274152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound274152.derived selector witness)

def rawBound : CoeffClass := LeftBound274152.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound274152.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound274155

namespace LeftBound274161
def owner : Owner := ⟨.program ⟨257⟩, ⟨20036⟩⟩
def transferEvent : Nat := 274161
def frameStart : Nat := 274096
def rule : BoundRule := .product (.predecessor 0 274159 .coefficient) (.predecessor 1 274160 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274159 .coefficient)
      LeftAuthority274157.bound (LeftAuthority274157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274160 .coefficient)
      LeftBound274155.bound (LeftBound274155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority274157.bound LeftBound274155.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274157.bound, LeftBound274155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority274157.actual selector witness) * (LeftBound274155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274161

namespace LeftBound274169
def owner : Owner := ⟨.program ⟨257⟩, ⟨20037⟩⟩
def transferEvent : Nat := 274169
def frameStart : Nat := 274096
def rule : BoundRule := .sum [.predecessor 0 274167 .coefficient, .predecessor 1 274168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274167 .coefficient)
      LeftAuthority274165.bound (LeftAuthority274165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274168 .coefficient)
      LeftBound274161.bound (LeftBound274161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority274165.bound, LeftBound274161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274165.bound, LeftBound274161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority274165.actual selector witness, LeftBound274161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274169

namespace LeftBound274173
def owner : Owner := ⟨.program ⟨257⟩, ⟨20396⟩⟩
def transferEvent : Nat := 274173
def frameStart : Nat := 274096
def rule : BoundRule := .product (.predecessor 0 274171 .coefficient) (.predecessor 1 274172 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274171 .coefficient)
      LeftBound274169.bound (LeftBound274169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274169.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274172 .coefficient)
      LeftAuthority274146.bound (LeftAuthority274146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound274169.bound LeftAuthority274146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274169.bound, LeftAuthority274146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound274169.actual selector witness) * (LeftAuthority274146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274173

namespace LeftBound274184
def owner : Owner := ⟨.program ⟨257⟩, ⟨18711⟩⟩
def transferEvent : Nat := 274184
def frameStart : Nat := 274096
def rule : BoundRule := .product (.predecessor 0 274182 .coefficient) (.predecessor 1 274183 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274182 .coefficient)
      LeftAuthority274157.bound (LeftAuthority274157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274183 .coefficient)
      LeftAuthority274180.bound (LeftAuthority274180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274180.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority274157.bound LeftAuthority274180.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274157.bound, LeftAuthority274180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority274157.actual selector witness) * (LeftAuthority274180.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274184

namespace LeftBound274192
def owner : Owner := ⟨.program ⟨257⟩, ⟨18712⟩⟩
def transferEvent : Nat := 274192
def frameStart : Nat := 274096
def rule : BoundRule := .sum [.predecessor 0 274190 .coefficient, .predecessor 1 274191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274190 .coefficient)
      LeftAuthority274188.bound (LeftAuthority274188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority274188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority274188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274191 .coefficient)
      LeftBound274184.bound (LeftBound274184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority274188.bound, LeftBound274184.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority274188.bound, LeftBound274184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority274188.actual selector witness, LeftBound274184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274192

namespace LeftBound274196
def owner : Owner := ⟨.program ⟨257⟩, ⟨20400⟩⟩
def transferEvent : Nat := 274196
def frameStart : Nat := 274096
def rule : BoundRule := .sum [.predecessor 0 274194 .coefficient, .predecessor 1 274195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274194 .coefficient)
      LeftBound274192.bound (LeftBound274192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274195 .coefficient)
      LeftBound274173.bound (LeftBound274173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274173.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274192.bound, LeftBound274173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274192.bound, LeftBound274173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274192.actual selector witness, LeftBound274173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274196

namespace LeftBound274209
def owner : Owner := ⟨.program ⟨257⟩, ⟨20398⟩⟩
def transferEvent : Nat := 274209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274207 .coefficient, .predecessor 1 274208 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274207 .coefficient)
      LeftBound274038.bound (LeftBound274038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274208 .coefficient)
      LeftBound274021.bound (LeftBound274021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1070.exact274028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274038.bound, LeftBound274021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274038.bound, LeftBound274021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274038.actual selector witness, LeftBound274021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274209

namespace LeftBound274212
def owner : Owner := ⟨.program ⟨257⟩, ⟨20398⟩⟩
def transferEvent : Nat := 274212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 274206 .summary, .result 274028 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274206 .summary)
      LeftBound274040.bound (LeftBound274040.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19293⟩⟩) (rawTerms := some (Proof.Events1071.exact274206RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 274028 .summary)
      LeftBound274023.bound (LeftBound274023.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20397⟩⟩) (rawTerms := some (Proof.Events1070.exact274028RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound274023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274040.bound, LeftBound274023.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274040.bound, LeftBound274023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274040.actual selector witness, LeftBound274023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274212

namespace LeftBound274236
def owner : Owner := ⟨.program ⟨257⟩, ⟨15277⟩⟩
def transferEvent : Nat := 274236
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 274234 .coefficient) (.predecessor 1 274235 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274234 .coefficient)
      LeftAuthority13200.bound (LeftAuthority13200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274235 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13200.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13200.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13200.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound274236

namespace LeftBound274241
def owner : Owner := ⟨.program ⟨257⟩, ⟨7660⟩⟩
def transferEvent : Nat := 274241
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 274239 .coefficient) (.predecessor 1 274240 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274239 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274240 .coefficient)
      LeftBound25596.bound (LeftBound25596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound25596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound25596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound25596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound274241

namespace LeftBound274246
def owner : Owner := ⟨.program ⟨257⟩, ⟨15278⟩⟩
def transferEvent : Nat := 274246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274244 .coefficient, .predecessor 1 274245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274244 .coefficient)
      LeftBound274241.bound (LeftBound274241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274245 .coefficient)
      LeftBound274236.bound (LeftBound274236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274236.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274241.bound, LeftBound274236.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274241.bound, LeftBound274236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274241.actual selector witness, LeftBound274236.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274246

namespace LeftBound274250
def owner : Owner := ⟨.program ⟨257⟩, ⟨15279⟩⟩
def transferEvent : Nat := 274250
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 274248 .coefficient, .predecessor 1 274249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 274248 .coefficient)
      LeftBound274246.bound (LeftBound274246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1071.exact274247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound274246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound274246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 274249 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound274246.bound, LeftBound25588.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound274246.bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound274246.actual selector witness, LeftBound25588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound274250

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
