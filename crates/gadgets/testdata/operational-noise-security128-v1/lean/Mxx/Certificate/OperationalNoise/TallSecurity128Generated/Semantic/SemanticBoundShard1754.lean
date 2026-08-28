import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1692
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1695
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1753

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound259149
def owner : Owner := ⟨.program ⟨257⟩, ⟨18160⟩⟩
def transferEvent : Nat := 259149
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 259147 .coefficient) (.predecessor 1 259148 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259147 .coefficient)
      LeftBound259143.bound (LeftBound259143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259148 .coefficient)
      LeftAuthority12432.bound (LeftAuthority12432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound259143.bound LeftAuthority12432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259143.bound, LeftAuthority12432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound259143.actual selector witness) * (LeftAuthority12432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259149

namespace LeftBound259150
def owner : Owner := ⟨.program ⟨257⟩, ⟨18160⟩⟩
def transferEvent : Nat := 259150
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12606⟩⟩], []⟩ [⟨.result 12433 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12433 .coefficient)
      LeftAuthority12432.bound (LeftAuthority12432.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12606⟩⟩) (rawTerms := some (Proof.Events048.exact12433RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12432.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12432.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority12432.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound259150

namespace LeftBound259151
def owner : Owner := ⟨.program ⟨257⟩, ⟨18160⟩⟩
def transferEvent : Nat := 259151
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 259146 .summary) (.transfer 259150) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259146 .summary)
      LeftBound259144.bound (LeftBound259144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18159⟩⟩) (rawTerms := some (Proof.Events1012.exact259146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 259150)
      LeftBound259150.bound (LeftBound259150.actual selector witness) := by
  exact .transfer (LeftBound259150.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound259144.bound LeftBound259150.bound
def bound : CoeffClass := .finite ⟨2555904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259144.bound, LeftBound259150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound259144.actual selector witness) * (LeftBound259150.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259151

namespace LeftBound259157
def owner : Owner := ⟨.program ⟨257⟩, ⟨12607⟩⟩
def transferEvent : Nat := 259157
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 259155 .coefficient) (.predecessor 1 259156 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259155 .coefficient)
      LeftAuthority12432.bound (LeftAuthority12432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259156 .coefficient)
      LeftBound251401.bound (LeftBound251401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251401.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority12432.bound LeftBound251401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12432.bound, LeftBound251401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority12432.actual selector witness) * (LeftBound251401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound259157

namespace LeftBound259162
def owner : Owner := ⟨.program ⟨257⟩, ⟨8013⟩⟩
def transferEvent : Nat := 259162
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 259160 .coefficient) (.predecessor 1 259161 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259160 .coefficient)
      LeftBound251272.bound (LeftBound251272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259161 .coefficient)
      LeftBound25136.bound (LeftBound25136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound251272.bound LeftBound25136.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251272.bound, LeftBound25136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound251272.actual selector witness) * (LeftBound25136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259162

namespace LeftBound259167
def owner : Owner := ⟨.program ⟨257⟩, ⟨12608⟩⟩
def transferEvent : Nat := 259167
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 259165 .coefficient, .predecessor 1 259166 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259165 .coefficient)
      LeftBound259162.bound (LeftBound259162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259166 .coefficient)
      LeftBound259157.bound (LeftBound259157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259162.bound, LeftBound259157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259162.bound, LeftBound259157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259162.actual selector witness, LeftBound259157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259167

namespace LeftBound259171
def owner : Owner := ⟨.program ⟨257⟩, ⟨12609⟩⟩
def transferEvent : Nat := 259171
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 259169 .coefficient, .predecessor 1 259170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259169 .coefficient)
      LeftBound259167.bound (LeftBound259167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259170 .coefficient)
      LeftBound25128.bound (LeftBound25128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259167.bound, LeftBound25128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259167.bound, LeftBound25128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259167.actual selector witness, LeftBound25128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259171

namespace LeftBound259172
def owner : Owner := ⟨.program ⟨257⟩, ⟨12609⟩⟩
def transferEvent : Nat := 259172
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨103⟩⟩]⟩ [⟨.result 25129 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25129 .coefficient)
      LeftBound25128.bound (LeftBound25128.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events098.exact25129RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25128.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25128.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25128.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound259172

namespace LeftBound259177
def owner : Owner := ⟨.program ⟨257⟩, ⟨12610⟩⟩
def transferEvent : Nat := 259177
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 259175 .coefficient) (.predecessor 1 259176 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259175 .coefficient)
      LeftBound259171.bound (LeftBound259171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259176 .coefficient)
      LeftBound25125.bound (LeftBound25125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259171.bound LeftBound25125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259171.bound, LeftBound25125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259171.actual selector witness) * (LeftBound25125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259177

namespace LeftBound259178
def owner : Owner := ⟨.program ⟨257⟩, ⟨12610⟩⟩
def transferEvent : Nat := 259178
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩ [⟨.result 25122 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25122 .coefficient)
      LeftAuthority25121.bound (LeftAuthority25121.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9571⟩⟩) (rawTerms := some (Proof.Events098.exact25122RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25121.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25121.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority25121.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound259178

namespace LeftBound259179
def owner : Owner := ⟨.program ⟨257⟩, ⟨12610⟩⟩
def transferEvent : Nat := 259179
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 259174 .summary) (.transfer 259178) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259174 .summary)
      LeftBound259172.bound (LeftBound259172.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12609⟩⟩) (rawTerms := some (Proof.Events1012.exact259174RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 259178)
      LeftBound259178.bound (LeftBound259178.actual selector witness) := by
  exact .transfer (LeftBound259178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259172.bound LeftBound259178.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259172.bound, LeftBound259178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259172.actual selector witness) * (LeftBound259178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259179

namespace LeftBound259187
def owner : Owner := ⟨.program ⟨257⟩, ⟨18161⟩⟩
def transferEvent : Nat := 259187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 259185 .coefficient, .predecessor 1 259186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259185 .coefficient)
      LeftBound259177.bound (LeftBound259177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259186 .coefficient)
      LeftBound259149.bound (LeftBound259149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259177.bound, LeftBound259149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259177.bound, LeftBound259149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259177.actual selector witness, LeftBound259149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259187

namespace LeftBound259189
def owner : Owner := ⟨.program ⟨257⟩, ⟨18161⟩⟩
def transferEvent : Nat := 259189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 259184 .summary, .result 259154 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259184 .summary)
      LeftBound259179.bound (LeftBound259179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12610⟩⟩) (rawTerms := some (Proof.Events1012.exact259184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259154 .summary)
      LeftBound259151.bound (LeftBound259151.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18160⟩⟩) (rawTerms := some (Proof.Events1012.exact259154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound259179.bound, LeftBound259151.bound]
def bound : CoeffClass := .finite ⟨279175430144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259179.bound, LeftBound259151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound259179.actual selector witness, LeftBound259151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound259189

namespace LeftBound259193
def owner : Owner := ⟨.program ⟨257⟩, ⟨20165⟩⟩
def transferEvent : Nat := 259193
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 259191 .coefficient) (.predecessor 1 259192 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 259191 .coefficient)
      LeftBound259187.bound (LeftBound259187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 259192 .coefficient)
      LeftAuthority259125.bound (LeftAuthority259125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1012.exact259126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259187.bound LeftAuthority259125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259187.bound, LeftAuthority259125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259187.actual selector witness) * (LeftAuthority259125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259193

namespace LeftBound259194
def owner : Owner := ⟨.program ⟨257⟩, ⟨20165⟩⟩
def transferEvent : Nat := 259194
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20164⟩⟩]⟩ [⟨.result 259126 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259126 .coefficient)
      LeftAuthority259125.bound (LeftAuthority259125.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20164⟩⟩) (rawTerms := some (Proof.Events1012.exact259126RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority259125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority259125.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority259125.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority259125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority259125.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound259194

namespace LeftBound259195
def owner : Owner := ⟨.program ⟨257⟩, ⟨20165⟩⟩
def transferEvent : Nat := 259195
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 259190 .summary) (.transfer 259194) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259190 .summary)
      LeftBound259189.bound (LeftBound259189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18161⟩⟩) (rawTerms := some (Proof.Events1012.exact259190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 259194)
      LeftBound259194.bound (LeftBound259194.actual selector witness) := by
  exact .transfer (LeftBound259194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259189.bound LeftBound259194.bound
def bound : CoeffClass := .finite ⟨2997623355788031426560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259189.bound, LeftBound259194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259189.actual selector witness) * (LeftBound259194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound259195

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
