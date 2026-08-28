import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard119
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1594
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1640

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound242995
def owner : Owner := ⟨.program ⟨257⟩, ⟨55871⟩⟩
def transferEvent : Nat := 242995
def frameStart : Nat := 242918
def rule : BoundRule := .product (.predecessor 0 242993 .coefficient) (.predecessor 1 242994 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 242993 .coefficient)
      LeftBound242991.bound (LeftBound242991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact242992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 242994 .coefficient)
      LeftAuthority242968.bound (LeftAuthority242968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact242969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority242968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority242968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound242991.bound LeftAuthority242968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound242991.bound, LeftAuthority242968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound242991.actual selector witness) * (LeftAuthority242968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound242995

namespace LeftBound243006
def owner : Owner := ⟨.program ⟨257⟩, ⟨54105⟩⟩
def transferEvent : Nat := 243006
def frameStart : Nat := 242918
def rule : BoundRule := .product (.predecessor 0 243004 .coefficient) (.predecessor 1 243005 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243004 .coefficient)
      LeftAuthority242979.bound (LeftAuthority242979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact242980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority242979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority242979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243005 .coefficient)
      LeftAuthority243002.bound (LeftAuthority243002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority242979.bound LeftAuthority243002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority242979.bound, LeftAuthority243002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority242979.actual selector witness) * (LeftAuthority243002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243006

namespace LeftBound243014
def owner : Owner := ⟨.program ⟨257⟩, ⟨54106⟩⟩
def transferEvent : Nat := 243014
def frameStart : Nat := 242918
def rule : BoundRule := .sum [.predecessor 0 243012 .coefficient, .predecessor 1 243013 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243012 .coefficient)
      LeftAuthority243010.bound (LeftAuthority243010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243013 .coefficient)
      LeftBound243006.bound (LeftBound243006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority243010.bound, LeftBound243006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243010.bound, LeftBound243006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority243010.actual selector witness, LeftBound243006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243014

namespace LeftBound243018
def owner : Owner := ⟨.program ⟨257⟩, ⟨55875⟩⟩
def transferEvent : Nat := 243018
def frameStart : Nat := 242918
def rule : BoundRule := .sum [.predecessor 0 243016 .coefficient, .predecessor 1 243017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243016 .coefficient)
      LeftBound243014.bound (LeftBound243014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243017 .coefficient)
      LeftBound242995.bound (LeftBound242995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243014.bound, LeftBound242995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243014.bound, LeftBound242995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243014.actual selector witness, LeftBound242995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243018

namespace LeftBound243031
def owner : Owner := ⟨.program ⟨257⟩, ⟨55873⟩⟩
def transferEvent : Nat := 243031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 243029 .coefficient, .predecessor 1 243030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243029 .coefficient)
      LeftBound242860.bound (LeftBound242860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243030 .coefficient)
      LeftBound242843.bound (LeftBound242843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events948.exact242850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound242843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound242843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound242860.bound, LeftBound242843.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound242860.bound, LeftBound242843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound242860.actual selector witness, LeftBound242843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243031

namespace LeftBound243034
def owner : Owner := ⟨.program ⟨257⟩, ⟨55873⟩⟩
def transferEvent : Nat := 243034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 243028 .summary, .result 242850 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243028 .summary)
      LeftBound242862.bound (LeftBound242862.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54699⟩⟩) (rawTerms := some (Proof.Events949.exact243028RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound242862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 242850 .summary)
      LeftBound242845.bound (LeftBound242845.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55872⟩⟩) (rawTerms := some (Proof.Events948.exact242850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound242845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound242862.bound, LeftBound242845.bound]
def bound : CoeffClass := .finite ⟨32189789464712143775715074244608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound242862.bound, LeftBound242845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound242862.actual selector witness, LeftBound242845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243034

namespace LeftBound243058
def owner : Owner := ⟨.program ⟨257⟩, ⟨24507⟩⟩
def transferEvent : Nat := 243058
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 243056 .coefficient) (.predecessor 1 243057 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243056 .coefficient)
      LeftAuthority11612.bound (LeftAuthority11612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243057 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11612.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11612.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11612.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound243058

namespace LeftBound243063
def owner : Owner := ⟨.program ⟨257⟩, ⟨8386⟩⟩
def transferEvent : Nat := 243063
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 243061 .coefficient) (.predecessor 1 243062 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243061 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243062 .coefficient)
      LeftBound23592.bound (LeftBound23592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound23592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound23592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound23592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243063

namespace LeftBound243068
def owner : Owner := ⟨.program ⟨257⟩, ⟨24508⟩⟩
def transferEvent : Nat := 243068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 243066 .coefficient, .predecessor 1 243067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243066 .coefficient)
      LeftBound243063.bound (LeftBound243063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243067 .coefficient)
      LeftBound243058.bound (LeftBound243058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243063.bound, LeftBound243058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243063.bound, LeftBound243058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243063.actual selector witness, LeftBound243058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243068

namespace LeftBound243072
def owner : Owner := ⟨.program ⟨257⟩, ⟨24509⟩⟩
def transferEvent : Nat := 243072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 243070 .coefficient, .predecessor 1 243071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243070 .coefficient)
      LeftBound243068.bound (LeftBound243068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243071 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243068.bound, LeftBound23584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243068.bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243068.actual selector witness, LeftBound23584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243072

namespace LeftBound243073
def owner : Owner := ⟨.program ⟨257⟩, ⟨24509⟩⟩
def transferEvent : Nat := 243073
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩ [⟨.result 23585 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23585 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨134⟩⟩) (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23584.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23584.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound243073

namespace LeftBound243078
def owner : Owner := ⟨.program ⟨257⟩, ⟨50494⟩⟩
def transferEvent : Nat := 243078
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 243076 .coefficient) (.predecessor 1 243077 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243076 .coefficient)
      LeftBound243072.bound (LeftBound243072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events949.exact243075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243077 .coefficient)
      LeftAuthority11615.bound (LeftAuthority11615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11615.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound243072.bound LeftAuthority11615.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243072.bound, LeftAuthority11615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound243072.actual selector witness) * (LeftAuthority11615.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243078

namespace LeftBound243079
def owner : Owner := ⟨.program ⟨257⟩, ⟨50494⟩⟩
def transferEvent : Nat := 243079
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨50491⟩⟩], []⟩ [⟨.result 11616 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 11616 .coefficient)
      LeftAuthority11615.bound (LeftAuthority11615.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50491⟩⟩) (rawTerms := some (Proof.Events045.exact11616RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11615.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11615.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority11615.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound243079

namespace LeftBound243080
def owner : Owner := ⟨.program ⟨257⟩, ⟨50494⟩⟩
def transferEvent : Nat := 243080
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 243075 .summary) (.transfer 243079) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243075 .summary)
      LeftBound243073.bound (LeftBound243073.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24509⟩⟩) (rawTerms := some (Proof.Events949.exact243075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 243079)
      LeftBound243079.bound (LeftBound243079.actual selector witness) := by
  exact .transfer (LeftBound243079.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound243073.bound LeftBound243079.bound
def bound : CoeffClass := .finite ⟨8519680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243073.bound, LeftBound243079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound243073.actual selector witness) * (LeftBound243079.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243080

namespace LeftBound243086
def owner : Owner := ⟨.program ⟨257⟩, ⟨50495⟩⟩
def transferEvent : Nat := 243086
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 243084 .coefficient) (.predecessor 1 243085 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243084 .coefficient)
      LeftAuthority11615.bound (LeftAuthority11615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243085 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11615.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11615.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11615.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound243086

namespace LeftBound243091
def owner : Owner := ⟨.program ⟨257⟩, ⟨8366⟩⟩
def transferEvent : Nat := 243091
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 243089 .coefficient) (.predecessor 1 243090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243089 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243090 .coefficient)
      LeftBound23633.bound (LeftBound23633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound23633.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound23633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound23633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243091

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
