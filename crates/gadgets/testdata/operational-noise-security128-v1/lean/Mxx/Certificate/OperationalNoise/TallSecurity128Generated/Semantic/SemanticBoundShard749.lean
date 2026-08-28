import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound113994
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def transferEvent : Nat := 113994
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68380⟩⟩]⟩ [⟨.result 113986 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113986 .coefficient)
      LeftAuthority113985.bound (LeftAuthority113985.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68380⟩⟩) (rawTerms := some (Proof.Events445.exact113986RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113985.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113985.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority113985.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority113985.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound113994

namespace LeftBound113995
def owner : Owner := ⟨.program ⟨257⟩, ⟨68383⟩⟩
def transferEvent : Nat := 113995
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 113994) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 113994)
      LeftBound113994.bound (LeftBound113994.actual selector witness) := by
  exact .transfer (LeftBound113994.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound113994.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound113994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound113994.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113995

namespace LeftBound115023
def owner : Owner := ⟨.program ⟨257⟩, ⟨18886⟩⟩
def transferEvent : Nat := 115023
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115021 .coefficient, .predecessor 1 115022 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115021 .coefficient)
      LeftAuthority115019.bound (LeftAuthority115019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115019.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115022 .coefficient)
      LeftAuthority114996.bound (LeftAuthority114996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact114997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority115019.bound, LeftAuthority114996.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115019.bound, LeftAuthority114996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority115019.actual selector witness, LeftAuthority114996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115023

namespace LeftBound115027
def owner : Owner := ⟨.program ⟨257⟩, ⟨22106⟩⟩
def transferEvent : Nat := 115027
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115025 .coefficient, .predecessor 1 115026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115025 .coefficient)
      LeftBound115023.bound (LeftBound115023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115026 .coefficient)
      LeftAuthority114973.bound (LeftAuthority114973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact114974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115023.bound, LeftAuthority114973.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115023.bound, LeftAuthority114973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115023.actual selector witness, LeftAuthority114973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115027

namespace LeftBound115031
def owner : Owner := ⟨.program ⟨257⟩, ⟨32126⟩⟩
def transferEvent : Nat := 115031
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115029 .coefficient, .predecessor 1 115030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115029 .coefficient)
      LeftBound115027.bound (LeftBound115027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115030 .coefficient)
      LeftAuthority114950.bound (LeftAuthority114950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact114951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114950.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115027.bound, LeftAuthority114950.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115027.bound, LeftAuthority114950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115027.actual selector witness, LeftAuthority114950.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115031

namespace LeftBound115035
def owner : Owner := ⟨.program ⟨257⟩, ⟨51181⟩⟩
def transferEvent : Nat := 115035
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115033 .coefficient, .predecessor 1 115034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115033 .coefficient)
      LeftBound115031.bound (LeftBound115031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115034 .coefficient)
      LeftAuthority114927.bound (LeftAuthority114927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115031.bound, LeftAuthority114927.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115031.bound, LeftAuthority114927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115031.actual selector witness, LeftAuthority114927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115035

namespace LeftBound115039
def owner : Owner := ⟨.program ⟨257⟩, ⟨54161⟩⟩
def transferEvent : Nat := 115039
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115037 .coefficient, .predecessor 1 115038 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115037 .coefficient)
      LeftBound115035.bound (LeftBound115035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115038 .coefficient)
      LeftAuthority114904.bound (LeftAuthority114904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114904.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115035.bound, LeftAuthority114904.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115035.bound, LeftAuthority114904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115035.actual selector witness, LeftAuthority114904.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115039

namespace LeftBound115043
def owner : Owner := ⟨.program ⟨257⟩, ⟨57141⟩⟩
def transferEvent : Nat := 115043
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115041 .coefficient, .predecessor 1 115042 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115041 .coefficient)
      LeftBound115039.bound (LeftBound115039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115042 .coefficient)
      LeftAuthority114881.bound (LeftAuthority114881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115039.bound, LeftAuthority114881.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115039.bound, LeftAuthority114881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115039.actual selector witness, LeftAuthority114881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115043

namespace LeftBound115047
def owner : Owner := ⟨.program ⟨257⟩, ⟨60121⟩⟩
def transferEvent : Nat := 115047
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115045 .coefficient, .predecessor 1 115046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115045 .coefficient)
      LeftBound115043.bound (LeftBound115043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115046 .coefficient)
      LeftAuthority114858.bound (LeftAuthority114858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115043.bound, LeftAuthority114858.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115043.bound, LeftAuthority114858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115043.actual selector witness, LeftAuthority114858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115047

namespace LeftBound115051
def owner : Owner := ⟨.program ⟨257⟩, ⟨63101⟩⟩
def transferEvent : Nat := 115051
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115049 .coefficient, .predecessor 1 115050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115049 .coefficient)
      LeftBound115047.bound (LeftBound115047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115050 .coefficient)
      LeftAuthority114835.bound (LeftAuthority114835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115047.bound, LeftAuthority114835.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115047.bound, LeftAuthority114835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115047.actual selector witness, LeftAuthority114835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115051

namespace LeftBound115055
def owner : Owner := ⟨.program ⟨257⟩, ⟨66672⟩⟩
def transferEvent : Nat := 115055
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115053 .coefficient, .predecessor 1 115054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115053 .coefficient)
      LeftBound115051.bound (LeftBound115051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115054 .coefficient)
      LeftAuthority114812.bound (LeftAuthority114812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115051.bound, LeftAuthority114812.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115051.bound, LeftAuthority114812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115051.actual selector witness, LeftAuthority114812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115055

namespace LeftBound115059
def owner : Owner := ⟨.program ⟨257⟩, ⟨66673⟩⟩
def transferEvent : Nat := 115059
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115057 .coefficient, .predecessor 1 115058 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115057 .coefficient)
      LeftBound115055.bound (LeftBound115055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115058 .coefficient)
      LeftAuthority114789.bound (LeftAuthority114789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115055.bound, LeftAuthority114789.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115055.bound, LeftAuthority114789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115055.actual selector witness, LeftAuthority114789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115059

namespace LeftBound115063
def owner : Owner := ⟨.program ⟨257⟩, ⟨66674⟩⟩
def transferEvent : Nat := 115063
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115061 .coefficient, .predecessor 1 115062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115061 .coefficient)
      LeftBound115059.bound (LeftBound115059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115062 .coefficient)
      LeftAuthority114766.bound (LeftAuthority114766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115059.bound, LeftAuthority114766.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115059.bound, LeftAuthority114766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115059.actual selector witness, LeftAuthority114766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115063

namespace LeftBound115067
def owner : Owner := ⟨.program ⟨257⟩, ⟨66675⟩⟩
def transferEvent : Nat := 115067
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115065 .coefficient, .predecessor 1 115066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115065 .coefficient)
      LeftBound115063.bound (LeftBound115063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115066 .coefficient)
      LeftAuthority114743.bound (LeftAuthority114743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114743.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115063.bound, LeftAuthority114743.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115063.bound, LeftAuthority114743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115063.actual selector witness, LeftAuthority114743.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115067

namespace LeftBound115071
def owner : Owner := ⟨.program ⟨257⟩, ⟨66676⟩⟩
def transferEvent : Nat := 115071
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115069 .coefficient, .predecessor 1 115070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115069 .coefficient)
      LeftBound115067.bound (LeftBound115067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115070 .coefficient)
      LeftAuthority114720.bound (LeftAuthority114720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115067.bound, LeftAuthority114720.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115067.bound, LeftAuthority114720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115067.actual selector witness, LeftAuthority114720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115071

namespace LeftBound115075
def owner : Owner := ⟨.program ⟨257⟩, ⟨66677⟩⟩
def transferEvent : Nat := 115075
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115073 .coefficient, .predecessor 1 115074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115073 .coefficient)
      LeftBound115071.bound (LeftBound115071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115074 .coefficient)
      LeftAuthority114697.bound (LeftAuthority114697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events448.exact114698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority114697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority114697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115071.bound, LeftAuthority114697.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115071.bound, LeftAuthority114697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115071.actual selector witness, LeftAuthority114697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115075

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
