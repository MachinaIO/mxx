import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1956
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1960
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1962

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound289110
def owner : Owner := ⟨.program ⟨257⟩, ⟨16479⟩⟩
def transferEvent : Nat := 289110
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨16476⟩⟩]⟩ [⟨.result 289102 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289102 .coefficient)
      LeftAuthority289101.bound (LeftAuthority289101.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨16476⟩⟩) (rawTerms := some (Proof.Events1129.exact289102RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289101.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority289101.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority289101.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound289110

namespace LeftBound289111
def owner : Owner := ⟨.program ⟨257⟩, ⟨16479⟩⟩
def transferEvent : Nat := 289111
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 289110) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 289110)
      LeftBound289110.bound (LeftBound289110.actual selector witness) := by
  exact .transfer (LeftBound289110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound289110.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound289110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound289110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289111

namespace LeftBound289206
def owner : Owner := ⟨.program ⟨257⟩, ⟨15741⟩⟩
def transferEvent : Nat := 289206
def frameStart : Nat := 289167
def rule : BoundRule := .identity (.predecessor 0 289205 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289205 .coefficient)
      LeftAuthority289203.bound (LeftAuthority289203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289203.derived selector witness)

def rawBound : CoeffClass := LeftAuthority289203.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority289203.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound289206

namespace LeftBound289223
def owner : Owner := ⟨.program ⟨257⟩, ⟨17182⟩⟩
def transferEvent : Nat := 289223
def frameStart : Nat := 289167
def rule : BoundRule := .sum [.predecessor 0 289221 .coefficient, .predecessor 1 289222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289221 .coefficient)
      LeftBound289206.bound (LeftBound289206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound289206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289222 .coefficient)
      LeftAuthority289219.bound (LeftAuthority289219.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority289219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289206.bound, LeftAuthority289219.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289206.bound, LeftAuthority289219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289206.actual selector witness, LeftAuthority289219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289223

namespace LeftBound289226
def owner : Owner := ⟨.program ⟨257⟩, ⟨17183⟩⟩
def transferEvent : Nat := 289226
def frameStart : Nat := 289167
def rule : BoundRule := .identity (.predecessor 0 289225 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289225 .coefficient)
      LeftBound289223.bound (LeftBound289223.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound289223.derived selector witness)

def rawBound : CoeffClass := LeftBound289223.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound289223.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound289226

namespace LeftBound289232
def owner : Owner := ⟨.program ⟨257⟩, ⟨17184⟩⟩
def transferEvent : Nat := 289232
def frameStart : Nat := 289167
def rule : BoundRule := .product (.predecessor 0 289230 .coefficient) (.predecessor 1 289231 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289230 .coefficient)
      LeftAuthority289228.bound (LeftAuthority289228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289228.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289231 .coefficient)
      LeftBound289226.bound (LeftBound289226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289226.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority289228.bound LeftBound289226.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289228.bound, LeftBound289226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority289228.actual selector witness) * (LeftBound289226.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289232

namespace LeftBound289240
def owner : Owner := ⟨.program ⟨257⟩, ⟨17185⟩⟩
def transferEvent : Nat := 289240
def frameStart : Nat := 289167
def rule : BoundRule := .sum [.predecessor 0 289238 .coefficient, .predecessor 1 289239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289238 .coefficient)
      LeftAuthority289236.bound (LeftAuthority289236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289236.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289239 .coefficient)
      LeftBound289232.bound (LeftBound289232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289232.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority289236.bound, LeftBound289232.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289236.bound, LeftBound289232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority289236.actual selector witness, LeftBound289232.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289240

namespace LeftBound289244
def owner : Owner := ⟨.program ⟨257⟩, ⟨17594⟩⟩
def transferEvent : Nat := 289244
def frameStart : Nat := 289167
def rule : BoundRule := .product (.predecessor 0 289242 .coefficient) (.predecessor 1 289243 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289242 .coefficient)
      LeftBound289240.bound (LeftBound289240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289243 .coefficient)
      LeftAuthority289217.bound (LeftAuthority289217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289240.bound LeftAuthority289217.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289240.bound, LeftAuthority289217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289240.actual selector witness) * (LeftAuthority289217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289244

namespace LeftBound289255
def owner : Owner := ⟨.program ⟨257⟩, ⟨15940⟩⟩
def transferEvent : Nat := 289255
def frameStart : Nat := 289167
def rule : BoundRule := .product (.predecessor 0 289253 .coefficient) (.predecessor 1 289254 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289253 .coefficient)
      LeftAuthority289228.bound (LeftAuthority289228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289228.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289254 .coefficient)
      LeftAuthority289251.bound (LeftAuthority289251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289251.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289251.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority289228.bound LeftAuthority289251.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289228.bound, LeftAuthority289251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority289228.actual selector witness) * (LeftAuthority289251.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289255

namespace LeftBound289263
def owner : Owner := ⟨.program ⟨257⟩, ⟨15941⟩⟩
def transferEvent : Nat := 289263
def frameStart : Nat := 289167
def rule : BoundRule := .sum [.predecessor 0 289261 .coefficient, .predecessor 1 289262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289261 .coefficient)
      LeftAuthority289259.bound (LeftAuthority289259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority289259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority289259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289262 .coefficient)
      LeftBound289255.bound (LeftBound289255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority289259.bound, LeftBound289255.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority289259.bound, LeftBound289255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority289259.actual selector witness, LeftBound289255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289263

namespace LeftBound289267
def owner : Owner := ⟨.program ⟨257⟩, ⟨17597⟩⟩
def transferEvent : Nat := 289267
def frameStart : Nat := 289167
def rule : BoundRule := .sum [.predecessor 0 289265 .coefficient, .predecessor 1 289266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289265 .coefficient)
      LeftBound289263.bound (LeftBound289263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289266 .coefficient)
      LeftBound289244.bound (LeftBound289244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289244.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289263.bound, LeftBound289244.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289263.bound, LeftBound289244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289263.actual selector witness, LeftBound289244.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289267

namespace LeftBound289280
def owner : Owner := ⟨.program ⟨257⟩, ⟨17596⟩⟩
def transferEvent : Nat := 289280
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289278 .coefficient, .predecessor 1 289279 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289278 .coefficient)
      LeftBound289109.bound (LeftBound289109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289279 .coefficient)
      LeftBound289092.bound (LeftBound289092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1129.exact289099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289109.bound, LeftBound289092.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289109.bound, LeftBound289092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289109.actual selector witness, LeftBound289092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289280

namespace LeftBound289283
def owner : Owner := ⟨.program ⟨257⟩, ⟨17596⟩⟩
def transferEvent : Nat := 289283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289277 .summary, .result 289099 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289277 .summary)
      LeftBound289111.bound (LeftBound289111.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16479⟩⟩) (rawTerms := some (Proof.Events1129.exact289277RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289099 .summary)
      LeftBound289094.bound (LeftBound289094.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17595⟩⟩) (rawTerms := some (Proof.Events1129.exact289099RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289111.bound, LeftBound289094.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289111.bound, LeftBound289094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289111.actual selector witness, LeftBound289094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289283

namespace LeftBound289287
def owner : Owner := ⟨.program ⟨257⟩, ⟨20470⟩⟩
def transferEvent : Nat := 289287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289285 .coefficient, .predecessor 1 289286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289285 .coefficient)
      LeftBound289280.bound (LeftBound289280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289286 .coefficient)
      LeftBound288800.bound (LeftBound288800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1128.exact288804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound288800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound288800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289280.bound, LeftBound288800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289280.bound, LeftBound288800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289280.actual selector witness, LeftBound288800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289287

namespace LeftBound289288
def owner : Owner := ⟨.program ⟨257⟩, ⟨20470⟩⟩
def transferEvent : Nat := 289288
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289284 .summary, .result 288804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289284 .summary)
      LeftBound289283.bound (LeftBound289283.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17596⟩⟩) (rawTerms := some (Proof.Events1130.exact289284RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 288804 .summary)
      LeftBound288803.bound (LeftBound288803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20469⟩⟩) (rawTerms := some (Proof.Events1128.exact288804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound288803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289283.bound, LeftBound288803.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289283.bound, LeftBound288803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289283.actual selector witness, LeftBound288803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289288

namespace LeftBound289292
def owner : Owner := ⟨.program ⟨257⟩, ⟨23690⟩⟩
def transferEvent : Nat := 289292
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289290 .coefficient, .predecessor 1 289291 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289290 .coefficient)
      LeftBound289287.bound (LeftBound289287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289291 .coefficient)
      LeftBound288320.bound (LeftBound288320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1126.exact288324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound288320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound288320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289287.bound, LeftBound288320.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289287.bound, LeftBound288320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289287.actual selector witness, LeftBound288320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289292

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
