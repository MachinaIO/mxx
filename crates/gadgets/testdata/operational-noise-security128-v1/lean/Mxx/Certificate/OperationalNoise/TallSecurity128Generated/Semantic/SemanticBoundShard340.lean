import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard333
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard337
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard339

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55146
def owner : Owner := ⟨.program ⟨257⟩, ⟨16759⟩⟩
def transferEvent : Nat := 55146
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨16756⟩⟩]⟩ [⟨.result 55138 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55138 .coefficient)
      LeftAuthority55137.bound (LeftAuthority55137.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨16756⟩⟩) (rawTerms := some (Proof.Events215.exact55138RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55137.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55137.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority55137.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55146

namespace LeftBound55147
def owner : Owner := ⟨.program ⟨257⟩, ⟨16759⟩⟩
def transferEvent : Nat := 55147
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 55146) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 55146)
      LeftBound55146.bound (LeftBound55146.actual selector witness) := by
  exact .transfer (LeftBound55146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound55146.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound55146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound55146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55147

namespace LeftBound55242
def owner : Owner := ⟨.program ⟨257⟩, ⟨15853⟩⟩
def transferEvent : Nat := 55242
def frameStart : Nat := 55203
def rule : BoundRule := .identity (.predecessor 0 55241 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55241 .coefficient)
      LeftAuthority55239.bound (LeftAuthority55239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55239.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55239.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority55239.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55242

namespace LeftBound55259
def owner : Owner := ⟨.program ⟨257⟩, ⟨17238⟩⟩
def transferEvent : Nat := 55259
def frameStart : Nat := 55203
def rule : BoundRule := .sum [.predecessor 0 55257 .coefficient, .predecessor 1 55258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55257 .coefficient)
      LeftBound55242.bound (LeftBound55242.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55258 .coefficient)
      LeftAuthority55255.bound (LeftAuthority55255.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55242.bound, LeftAuthority55255.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55242.bound, LeftAuthority55255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55242.actual selector witness, LeftAuthority55255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55259

namespace LeftBound55262
def owner : Owner := ⟨.program ⟨257⟩, ⟨17239⟩⟩
def transferEvent : Nat := 55262
def frameStart : Nat := 55203
def rule : BoundRule := .identity (.predecessor 0 55261 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55261 .coefficient)
      LeftBound55259.bound (LeftBound55259.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55259.derived selector witness)

def rawBound : CoeffClass := LeftBound55259.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound55259.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55262

namespace LeftBound55268
def owner : Owner := ⟨.program ⟨257⟩, ⟨17240⟩⟩
def transferEvent : Nat := 55268
def frameStart : Nat := 55203
def rule : BoundRule := .product (.predecessor 0 55266 .coefficient) (.predecessor 1 55267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55266 .coefficient)
      LeftAuthority55264.bound (LeftAuthority55264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55267 .coefficient)
      LeftBound55262.bound (LeftBound55262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55262.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority55264.bound LeftBound55262.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55264.bound, LeftBound55262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority55264.actual selector witness) * (LeftBound55262.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55268

namespace LeftBound55276
def owner : Owner := ⟨.program ⟨257⟩, ⟨17241⟩⟩
def transferEvent : Nat := 55276
def frameStart : Nat := 55203
def rule : BoundRule := .sum [.predecessor 0 55274 .coefficient, .predecessor 1 55275 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55274 .coefficient)
      LeftAuthority55272.bound (LeftAuthority55272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55275 .coefficient)
      LeftBound55268.bound (LeftBound55268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55268.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55272.bound, LeftBound55268.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55272.bound, LeftBound55268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority55272.actual selector witness, LeftBound55268.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55276

namespace LeftBound55280
def owner : Owner := ⟨.program ⟨257⟩, ⟨17986⟩⟩
def transferEvent : Nat := 55280
def frameStart : Nat := 55203
def rule : BoundRule := .product (.predecessor 0 55278 .coefficient) (.predecessor 1 55279 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55278 .coefficient)
      LeftBound55276.bound (LeftBound55276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55279 .coefficient)
      LeftAuthority55253.bound (LeftAuthority55253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55253.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55276.bound LeftAuthority55253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55276.bound, LeftAuthority55253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55276.actual selector witness) * (LeftAuthority55253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55280

namespace LeftBound55291
def owner : Owner := ⟨.program ⟨257⟩, ⟨16164⟩⟩
def transferEvent : Nat := 55291
def frameStart : Nat := 55203
def rule : BoundRule := .product (.predecessor 0 55289 .coefficient) (.predecessor 1 55290 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55289 .coefficient)
      LeftAuthority55264.bound (LeftAuthority55264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55290 .coefficient)
      LeftAuthority55287.bound (LeftAuthority55287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55287.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55264.bound LeftAuthority55287.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55264.bound, LeftAuthority55287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority55264.actual selector witness) * (LeftAuthority55287.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55291

namespace LeftBound55299
def owner : Owner := ⟨.program ⟨257⟩, ⟨16165⟩⟩
def transferEvent : Nat := 55299
def frameStart : Nat := 55203
def rule : BoundRule := .sum [.predecessor 0 55297 .coefficient, .predecessor 1 55298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55297 .coefficient)
      LeftAuthority55295.bound (LeftAuthority55295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55298 .coefficient)
      LeftBound55291.bound (LeftBound55291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55295.bound, LeftBound55291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55295.bound, LeftBound55291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority55295.actual selector witness, LeftBound55291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55299

namespace LeftBound55303
def owner : Owner := ⟨.program ⟨257⟩, ⟨17989⟩⟩
def transferEvent : Nat := 55303
def frameStart : Nat := 55203
def rule : BoundRule := .sum [.predecessor 0 55301 .coefficient, .predecessor 1 55302 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55301 .coefficient)
      LeftBound55299.bound (LeftBound55299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55302 .coefficient)
      LeftBound55280.bound (LeftBound55280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55299.bound, LeftBound55280.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55299.bound, LeftBound55280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55299.actual selector witness, LeftBound55280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55303

namespace LeftBound55316
def owner : Owner := ⟨.program ⟨257⟩, ⟨17988⟩⟩
def transferEvent : Nat := 55316
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55314 .coefficient, .predecessor 1 55315 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55314 .coefficient)
      LeftBound55145.bound (LeftBound55145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55315 .coefficient)
      LeftBound55128.bound (LeftBound55128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55145.bound, LeftBound55128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55145.bound, LeftBound55128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55145.actual selector witness, LeftBound55128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55316

namespace LeftBound55319
def owner : Owner := ⟨.program ⟨257⟩, ⟨17988⟩⟩
def transferEvent : Nat := 55319
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55313 .summary, .result 55135 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55313 .summary)
      LeftBound55147.bound (LeftBound55147.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16759⟩⟩) (rawTerms := some (Proof.Events216.exact55313RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55135 .summary)
      LeftBound55130.bound (LeftBound55130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17987⟩⟩) (rawTerms := some (Proof.Events215.exact55135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55147.bound, LeftBound55130.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55147.bound, LeftBound55130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55147.actual selector witness, LeftBound55130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55319

namespace LeftBound55323
def owner : Owner := ⟨.program ⟨257⟩, ⟨20904⟩⟩
def transferEvent : Nat := 55323
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55321 .coefficient, .predecessor 1 55322 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55321 .coefficient)
      LeftBound55316.bound (LeftBound55316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55322 .coefficient)
      LeftBound54834.bound (LeftBound54834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55316.bound, LeftBound54834.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55316.bound, LeftBound54834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55316.actual selector witness, LeftBound54834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55323

namespace LeftBound55324
def owner : Owner := ⟨.program ⟨257⟩, ⟨20904⟩⟩
def transferEvent : Nat := 55324
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55320 .summary, .result 54838 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55320 .summary)
      LeftBound55319.bound (LeftBound55319.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17988⟩⟩) (rawTerms := some (Proof.Events216.exact55320RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54838 .summary)
      LeftBound54837.bound (LeftBound54837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20903⟩⟩) (rawTerms := some (Proof.Events214.exact54838RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55319.bound, LeftBound54837.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55319.bound, LeftBound54837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55319.actual selector witness, LeftBound54837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55324

namespace LeftBound55328
def owner : Owner := ⟨.program ⟨257⟩, ⟨24124⟩⟩
def transferEvent : Nat := 55328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55326 .coefficient, .predecessor 1 55327 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55326 .coefficient)
      LeftBound55323.bound (LeftBound55323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55327 .coefficient)
      LeftBound54352.bound (LeftBound54352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55323.bound, LeftBound54352.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55323.bound, LeftBound54352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55323.actual selector witness, LeftBound54352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55328

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
