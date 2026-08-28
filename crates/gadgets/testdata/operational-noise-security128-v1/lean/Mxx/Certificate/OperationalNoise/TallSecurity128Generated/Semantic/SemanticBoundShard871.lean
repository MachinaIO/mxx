import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard835
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard870

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound132860
def owner : Owner := ⟨.program ⟨257⟩, ⟨51675⟩⟩
def transferEvent : Nat := 132860
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 119870 .summary) (.transfer 132859) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119870 .summary)
      LeftBound119868.bound (LeftBound119868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5527⟩⟩) (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 132859)
      LeftBound132859.bound (LeftBound132859.actual selector witness) := by
  exact .transfer (LeftBound132859.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119868.bound LeftBound132859.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119868.bound, LeftBound132859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119868.actual selector witness) * (LeftBound132859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132860

namespace LeftBound132955
def owner : Owner := ⟨.program ⟨257⟩, ⟨50857⟩⟩
def transferEvent : Nat := 132955
def frameStart : Nat := 132916
def rule : BoundRule := .identity (.predecessor 0 132954 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132954 .coefficient)
      LeftAuthority132952.bound (LeftAuthority132952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132952.derived selector witness)

def rawBound : CoeffClass := LeftAuthority132952.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority132952.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound132955

namespace LeftBound132972
def owner : Owner := ⟨.program ⟨257⟩, ⟨52350⟩⟩
def transferEvent : Nat := 132972
def frameStart : Nat := 132916
def rule : BoundRule := .sum [.predecessor 0 132970 .coefficient, .predecessor 1 132971 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132970 .coefficient)
      LeftBound132955.bound (LeftBound132955.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound132955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132971 .coefficient)
      LeftAuthority132968.bound (LeftAuthority132968.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority132968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132955.bound, LeftAuthority132968.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132955.bound, LeftAuthority132968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132955.actual selector witness, LeftAuthority132968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132972

namespace LeftBound132975
def owner : Owner := ⟨.program ⟨257⟩, ⟨52351⟩⟩
def transferEvent : Nat := 132975
def frameStart : Nat := 132916
def rule : BoundRule := .identity (.predecessor 0 132974 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132974 .coefficient)
      LeftBound132972.bound (LeftBound132972.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound132972.derived selector witness)

def rawBound : CoeffClass := LeftBound132972.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound132972.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound132975

namespace LeftBound132981
def owner : Owner := ⟨.program ⟨257⟩, ⟨52352⟩⟩
def transferEvent : Nat := 132981
def frameStart : Nat := 132916
def rule : BoundRule := .product (.predecessor 0 132979 .coefficient) (.predecessor 1 132980 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132979 .coefficient)
      LeftAuthority132977.bound (LeftAuthority132977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132980 .coefficient)
      LeftBound132975.bound (LeftBound132975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132975.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority132977.bound LeftBound132975.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132977.bound, LeftBound132975.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority132977.actual selector witness) * (LeftBound132975.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132981

namespace LeftBound132989
def owner : Owner := ⟨.program ⟨257⟩, ⟨52353⟩⟩
def transferEvent : Nat := 132989
def frameStart : Nat := 132916
def rule : BoundRule := .sum [.predecessor 0 132987 .coefficient, .predecessor 1 132988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132987 .coefficient)
      LeftAuthority132985.bound (LeftAuthority132985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132985.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132985.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132988 .coefficient)
      LeftBound132981.bound (LeftBound132981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132981.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority132985.bound, LeftBound132981.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132985.bound, LeftBound132981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority132985.actual selector witness, LeftBound132981.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132989

namespace LeftBound132993
def owner : Owner := ⟨.program ⟨257⟩, ⟨52822⟩⟩
def transferEvent : Nat := 132993
def frameStart : Nat := 132916
def rule : BoundRule := .product (.predecessor 0 132991 .coefficient) (.predecessor 1 132992 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132991 .coefficient)
      LeftBound132989.bound (LeftBound132989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132989.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132992 .coefficient)
      LeftAuthority132966.bound (LeftAuthority132966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132966.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound132989.bound LeftAuthority132966.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132989.bound, LeftAuthority132966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound132989.actual selector witness) * (LeftAuthority132966.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132993

namespace LeftBound133004
def owner : Owner := ⟨.program ⟨257⟩, ⟨51092⟩⟩
def transferEvent : Nat := 133004
def frameStart : Nat := 132916
def rule : BoundRule := .product (.predecessor 0 133002 .coefficient) (.predecessor 1 133003 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133002 .coefficient)
      LeftAuthority132977.bound (LeftAuthority132977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133003 .coefficient)
      LeftAuthority133000.bound (LeftAuthority133000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority133000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority133000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority132977.bound LeftAuthority133000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132977.bound, LeftAuthority133000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority132977.actual selector witness) * (LeftAuthority133000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound133004

namespace LeftBound133012
def owner : Owner := ⟨.program ⟨257⟩, ⟨51093⟩⟩
def transferEvent : Nat := 133012
def frameStart : Nat := 132916
def rule : BoundRule := .sum [.predecessor 0 133010 .coefficient, .predecessor 1 133011 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133010 .coefficient)
      LeftAuthority133008.bound (LeftAuthority133008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority133008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority133008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133011 .coefficient)
      LeftBound133004.bound (LeftBound133004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133004.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority133008.bound, LeftBound133004.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority133008.bound, LeftBound133004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority133008.actual selector witness, LeftBound133004.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133012

namespace LeftBound133016
def owner : Owner := ⟨.program ⟨257⟩, ⟨52827⟩⟩
def transferEvent : Nat := 133016
def frameStart : Nat := 132916
def rule : BoundRule := .sum [.predecessor 0 133014 .coefficient, .predecessor 1 133015 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133014 .coefficient)
      LeftBound133012.bound (LeftBound133012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133015 .coefficient)
      LeftBound132993.bound (LeftBound132993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact132998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133012.bound, LeftBound132993.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133012.bound, LeftBound132993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133012.actual selector witness, LeftBound132993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133016

namespace LeftBound133029
def owner : Owner := ⟨.program ⟨257⟩, ⟨52824⟩⟩
def transferEvent : Nat := 133029
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133027 .coefficient, .predecessor 1 133028 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133027 .coefficient)
      LeftBound132858.bound (LeftBound132858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133028 .coefficient)
      LeftBound132841.bound (LeftBound132841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events518.exact132848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132858.bound, LeftBound132841.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132858.bound, LeftBound132841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132858.actual selector witness, LeftBound132841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133029

namespace LeftBound133032
def owner : Owner := ⟨.program ⟨257⟩, ⟨52824⟩⟩
def transferEvent : Nat := 133032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133026 .summary, .result 132848 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133026 .summary)
      LeftBound132860.bound (LeftBound132860.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51675⟩⟩) (rawTerms := some (Proof.Events519.exact133026RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132848 .summary)
      LeftBound132843.bound (LeftBound132843.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52823⟩⟩) (rawTerms := some (Proof.Events518.exact132848RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132860.bound, LeftBound132843.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132860.bound, LeftBound132843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132860.actual selector witness, LeftBound132843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133032

namespace LeftBound133036
def owner : Owner := ⟨.program ⟨257⟩, ⟨52825⟩⟩
def transferEvent : Nat := 133036
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 133034 .coefficient) (.predecessor 1 133035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133034 .coefficient)
      LeftBound133029.bound (LeftBound133029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133035 .coefficient)
      LeftBound15801.bound (LeftBound15801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound133029.bound LeftBound15801.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133029.bound, LeftBound15801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound133029.actual selector witness) * (LeftBound15801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound133036

namespace LeftBound133037
def owner : Owner := ⟨.program ⟨257⟩, ⟨52825⟩⟩
def transferEvent : Nat := 133037
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩ [⟨.result 15798 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15798 .coefficient)
      LeftAuthority15797.bound (LeftAuthority15797.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7131⟩⟩) (rawTerms := some (Proof.Events061.exact15798RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15797.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15797.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15797.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound133037

namespace LeftBound133038
def owner : Owner := ⟨.program ⟨257⟩, ⟨52825⟩⟩
def transferEvent : Nat := 133038
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 133033 .summary) (.transfer 133037) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133033 .summary)
      LeftBound133032.bound (LeftBound133032.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52824⟩⟩) (rawTerms := some (Proof.Events519.exact133033RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 133037)
      LeftBound133037.bound (LeftBound133037.actual selector witness) := by
  exact .transfer (LeftBound133037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound133032.bound LeftBound133037.bound
def bound : CoeffClass := .finite ⟨345633123169561229153141416722874415185920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133032.bound, LeftBound133037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound133032.actual selector witness) * (LeftBound133037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound133038

namespace LeftBound133053
def owner : Owner := ⟨.program ⟨257⟩, ⟨33763⟩⟩
def transferEvent : Nat := 133053
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 133051 .coefficient) (.predecessor 1 133052 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133051 .coefficient)
      LeftBound126800.bound (LeftBound126800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133052 .coefficient)
      LeftAuthority133049.bound (LeftAuthority133049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events519.exact133050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority133049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority133049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound126800.bound LeftAuthority133049.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126800.bound, LeftAuthority133049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound126800.actual selector witness) * (LeftAuthority133049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound133053

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
