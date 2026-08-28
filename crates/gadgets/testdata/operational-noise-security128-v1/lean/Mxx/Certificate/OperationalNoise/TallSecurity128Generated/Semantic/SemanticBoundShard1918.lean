import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound283144
def owner : Owner := ⟨.program ⟨257⟩, ⟨35131⟩⟩
def transferEvent : Nat := 283144
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 283142 .coefficient) (.value (.predecessor 1 283143 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283142 .coefficient)
      LeftAuthority283140.bound (LeftAuthority283140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283143 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority283140.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283140.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority283140.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound283144

namespace LeftBound283148
def owner : Owner := ⟨.program ⟨257⟩, ⟨35132⟩⟩
def transferEvent : Nat := 283148
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 283146 .coefficient) (.predecessor 1 283147 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283146 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283147 .coefficient)
      LeftBound283144.bound (LeftBound283144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283144.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283144.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound283144.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound283144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound283144.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283148

namespace LeftBound283149
def owner : Owner := ⟨.program ⟨257⟩, ⟨35132⟩⟩
def transferEvent : Nat := 283149
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35129⟩⟩]⟩ [⟨.result 283141 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283141 .coefficient)
      LeftAuthority283140.bound (LeftAuthority283140.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35129⟩⟩) (rawTerms := some (Proof.Events1106.exact283141RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283140.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority283140.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority283140.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound283149

namespace LeftBound283150
def owner : Owner := ⟨.program ⟨257⟩, ⟨35132⟩⟩
def transferEvent : Nat := 283150
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 283149) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 283149)
      LeftBound283149.bound (LeftBound283149.actual selector witness) := by
  exact .transfer (LeftBound283149.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound283149.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound283149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound283149.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283150

namespace LeftBound283229
def owner : Owner := ⟨.program ⟨257⟩, ⟨34291⟩⟩
def transferEvent : Nat := 283229
def frameStart : Nat := 283200
def rule : BoundRule := .product (.predecessor 0 283227 .coefficient) (.predecessor 1 283228 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283227 .coefficient)
      LeftAuthority283225.bound (LeftAuthority283225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283228 .coefficient)
      LeftAuthority283222.bound (LeftAuthority283222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283222.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority283225.bound LeftAuthority283222.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283225.bound, LeftAuthority283222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority283225.actual selector witness) * (LeftAuthority283222.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283229

namespace LeftBound283233
def owner : Owner := ⟨.program ⟨257⟩, ⟨34292⟩⟩
def transferEvent : Nat := 283233
def frameStart : Nat := 283200
def rule : BoundRule := .identity (.predecessor 0 283232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283232 .coefficient)
      LeftBound283229.bound (LeftBound283229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283229.derived selector witness)

def rawBound : CoeffClass := LeftBound283229.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound283229.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound283233

namespace LeftBound283250
def owner : Owner := ⟨.program ⟨257⟩, ⟨36002⟩⟩
def transferEvent : Nat := 283250
def frameStart : Nat := 283200
def rule : BoundRule := .sum [.predecessor 0 283248 .coefficient, .predecessor 1 283249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283248 .coefficient)
      LeftBound283233.bound (LeftBound283233.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound283233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283249 .coefficient)
      LeftAuthority283246.bound (LeftAuthority283246.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority283246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283233.bound, LeftAuthority283246.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283233.bound, LeftAuthority283246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283233.actual selector witness, LeftAuthority283246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283250

namespace LeftBound283253
def owner : Owner := ⟨.program ⟨257⟩, ⟨36003⟩⟩
def transferEvent : Nat := 283253
def frameStart : Nat := 283200
def rule : BoundRule := .identity (.predecessor 0 283252 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283252 .coefficient)
      LeftBound283250.bound (LeftBound283250.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound283250.derived selector witness)

def rawBound : CoeffClass := LeftBound283250.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound283250.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound283253

namespace LeftBound283259
def owner : Owner := ⟨.program ⟨257⟩, ⟨36004⟩⟩
def transferEvent : Nat := 283259
def frameStart : Nat := 283200
def rule : BoundRule := .product (.predecessor 0 283257 .coefficient) (.predecessor 1 283258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283257 .coefficient)
      LeftAuthority283255.bound (LeftAuthority283255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283258 .coefficient)
      LeftBound283253.bound (LeftBound283253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority283255.bound LeftBound283253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283255.bound, LeftBound283253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority283255.actual selector witness) * (LeftBound283253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283259

namespace LeftBound283273
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 283273
def frameStart : Nat := 283200
def rule : BoundRule := .scale (.predecessor 0 283271 .coefficient) (.value (.predecessor 1 283272 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283271 .coefficient)
      LeftAuthority283269.bound (LeftAuthority283269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283272 .coefficient)
      LeftAuthority283203.bound (LeftAuthority283203.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority283203.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority283269.bound LeftAuthority283203.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283269.bound, LeftAuthority283203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority283269.actual selector witness) * (LeftAuthority283203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound283273

namespace LeftBound283276
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 283276
def frameStart : Nat := 283200
def rule : BoundRule := .identity (.predecessor 0 283275 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283275 .coefficient)
      LeftAuthority283263.bound (LeftAuthority283263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283263.derived selector witness)

def rawBound : CoeffClass := LeftAuthority283263.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority283263.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound283276

namespace LeftBound283280
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 283280
def frameStart : Nat := 283200
def rule : BoundRule := .product (.predecessor 0 283278 .coefficient) (.predecessor 1 283279 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283278 .coefficient)
      LeftBound283276.bound (LeftBound283276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283279 .coefficient)
      LeftBound283273.bound (LeftBound283273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283276.bound LeftBound283273.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283276.bound, LeftBound283273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283276.actual selector witness) * (LeftBound283273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283280

namespace LeftBound283285
def owner : Owner := ⟨.program ⟨257⟩, ⟨36005⟩⟩
def transferEvent : Nat := 283285
def frameStart : Nat := 283200
def rule : BoundRule := .sum [.predecessor 0 283283 .coefficient, .predecessor 1 283284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283283 .coefficient)
      LeftBound283280.bound (LeftBound283280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283284 .coefficient)
      LeftBound283259.bound (LeftBound283259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283280.bound, LeftBound283259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283280.bound, LeftBound283259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283280.actual selector witness, LeftBound283259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283285

namespace LeftBound283289
def owner : Owner := ⟨.program ⟨257⟩, ⟨36196⟩⟩
def transferEvent : Nat := 283289
def frameStart : Nat := 283200
def rule : BoundRule := .product (.predecessor 0 283287 .coefficient) (.predecessor 1 283288 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283287 .coefficient)
      LeftBound283285.bound (LeftBound283285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283288 .coefficient)
      LeftAuthority283244.bound (LeftAuthority283244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283244.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283285.bound LeftAuthority283244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283285.bound, LeftAuthority283244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283285.actual selector witness) * (LeftAuthority283244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283289

namespace LeftBound283300
def owner : Owner := ⟨.program ⟨257⟩, ⟨34702⟩⟩
def transferEvent : Nat := 283300
def frameStart : Nat := 283200
def rule : BoundRule := .product (.predecessor 0 283298 .coefficient) (.predecessor 1 283299 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283298 .coefficient)
      LeftAuthority283255.bound (LeftAuthority283255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283299 .coefficient)
      LeftAuthority283296.bound (LeftAuthority283296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283296.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283296.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority283255.bound LeftAuthority283296.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283255.bound, LeftAuthority283296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority283255.actual selector witness) * (LeftAuthority283296.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283300

namespace LeftBound283308
def owner : Owner := ⟨.program ⟨257⟩, ⟨34703⟩⟩
def transferEvent : Nat := 283308
def frameStart : Nat := 283200
def rule : BoundRule := .sum [.predecessor 0 283306 .coefficient, .predecessor 1 283307 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283306 .coefficient)
      LeftAuthority283304.bound (LeftAuthority283304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283304.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283307 .coefficient)
      LeftBound283300.bound (LeftBound283300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1106.exact283302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority283304.bound, LeftBound283300.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283304.bound, LeftBound283300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority283304.actual selector witness, LeftBound283300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283308

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
