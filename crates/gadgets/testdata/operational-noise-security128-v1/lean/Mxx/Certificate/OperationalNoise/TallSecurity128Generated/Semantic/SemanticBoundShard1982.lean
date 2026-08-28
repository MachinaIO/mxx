import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1937
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1981

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound292995
def owner : Owner := ⟨.program ⟨257⟩, ⟨62974⟩⟩
def transferEvent : Nat := 292995
def frameStart : Nat := 292907
def rule : BoundRule := .product (.predecessor 0 292993 .coefficient) (.predecessor 1 292994 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292993 .coefficient)
      LeftAuthority292968.bound (LeftAuthority292968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292994 .coefficient)
      LeftAuthority292991.bound (LeftAuthority292991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292991.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292991.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority292968.bound LeftAuthority292991.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292968.bound, LeftAuthority292991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority292968.actual selector witness) * (LeftAuthority292991.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292995

namespace LeftBound293003
def owner : Owner := ⟨.program ⟨257⟩, ⟨62975⟩⟩
def transferEvent : Nat := 293003
def frameStart : Nat := 292907
def rule : BoundRule := .sum [.predecessor 0 293001 .coefficient, .predecessor 1 293002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293001 .coefficient)
      LeftAuthority292999.bound (LeftAuthority292999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292999.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293002 .coefficient)
      LeftBound292995.bound (LeftBound292995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority292999.bound, LeftBound292995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292999.bound, LeftBound292995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority292999.actual selector witness, LeftBound292995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound293003

namespace LeftBound293007
def owner : Owner := ⟨.program ⟨257⟩, ⟨64685⟩⟩
def transferEvent : Nat := 293007
def frameStart : Nat := 292907
def rule : BoundRule := .sum [.predecessor 0 293005 .coefficient, .predecessor 1 293006 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293005 .coefficient)
      LeftBound293003.bound (LeftBound293003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293006 .coefficient)
      LeftBound292984.bound (LeftBound292984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292984.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound293003.bound, LeftBound292984.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound293003.bound, LeftBound292984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound293003.actual selector witness, LeftBound292984.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound293007

namespace LeftBound293020
def owner : Owner := ⟨.program ⟨257⟩, ⟨64682⟩⟩
def transferEvent : Nat := 293020
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 293018 .coefficient, .predecessor 1 293019 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293018 .coefficient)
      LeftBound292849.bound (LeftBound292849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293019 .coefficient)
      LeftBound292832.bound (LeftBound292832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292849.bound, LeftBound292832.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292849.bound, LeftBound292832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292849.actual selector witness, LeftBound292832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound293020

namespace LeftBound293023
def owner : Owner := ⟨.program ⟨257⟩, ⟨64682⟩⟩
def transferEvent : Nat := 293023
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 293017 .summary, .result 292839 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293017 .summary)
      LeftBound292851.bound (LeftBound292851.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63555⟩⟩) (rawTerms := some (Proof.Events1144.exact293017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292839 .summary)
      LeftBound292834.bound (LeftBound292834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64681⟩⟩) (rawTerms := some (Proof.Events1143.exact292839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292851.bound, LeftBound292834.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292851.bound, LeftBound292834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292851.actual selector witness, LeftBound292834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound293023

namespace LeftBound293027
def owner : Owner := ⟨.program ⟨257⟩, ⟨64683⟩⟩
def transferEvent : Nat := 293027
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 293025 .coefficient) (.predecessor 1 293026 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293025 .coefficient)
      LeftBound293020.bound (LeftBound293020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293026 .coefficient)
      LeftBound15721.bound (LeftBound15721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15721.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound293020.bound LeftBound15721.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound293020.bound, LeftBound15721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound293020.actual selector witness) * (LeftBound15721.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293027

namespace LeftBound293028
def owner : Owner := ⟨.program ⟨257⟩, ⟨64683⟩⟩
def transferEvent : Nat := 293028
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩ [⟨.result 15718 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15718 .coefficient)
      LeftAuthority15717.bound (LeftAuthority15717.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7099⟩⟩) (rawTerms := some (Proof.Events061.exact15718RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15717.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15717.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15717.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound293028

namespace LeftBound293029
def owner : Owner := ⟨.program ⟨257⟩, ⟨64683⟩⟩
def transferEvent : Nat := 293029
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 293024 .summary) (.transfer 293028) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293024 .summary)
      LeftBound293023.bound (LeftBound293023.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64682⟩⟩) (rawTerms := some (Proof.Events1144.exact293024RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 293028)
      LeftBound293028.bound (LeftBound293028.actual selector witness) := by
  exact .transfer (LeftBound293028.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound293023.bound LeftBound293028.bound
def bound : CoeffClass := .finite ⟨345645779393153907795485959807676889169920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound293023.bound, LeftBound293028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound293023.actual selector witness) * (LeftBound293028.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293029

namespace LeftBound293044
def owner : Owner := ⟨.program ⟨257⟩, ⟨61701⟩⟩
def transferEvent : Nat := 293044
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 293042 .coefficient) (.predecessor 1 293043 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293042 .coefficient)
      LeftBound285725.bound (LeftBound285725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293043 .coefficient)
      LeftAuthority293040.bound (LeftAuthority293040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293040.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285725.bound LeftAuthority293040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285725.bound, LeftAuthority293040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285725.actual selector witness) * (LeftAuthority293040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293044

namespace LeftBound293045
def owner : Owner := ⟨.program ⟨257⟩, ⟨61701⟩⟩
def transferEvent : Nat := 293045
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61699⟩⟩]⟩ [⟨.result 293041 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293041 .coefficient)
      LeftAuthority293040.bound (LeftAuthority293040.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61699⟩⟩) (rawTerms := some (Proof.Events1144.exact293041RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293040.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293040.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority293040.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority293040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority293040.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound293045

namespace LeftBound293046
def owner : Owner := ⟨.program ⟨257⟩, ⟨61701⟩⟩
def transferEvent : Nat := 293046
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 285729 .summary) (.transfer 293045) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285729 .summary)
      LeftBound285728.bound (LeftBound285728.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61395⟩⟩) (rawTerms := some (Proof.Events1116.exact285729RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 293045)
      LeftBound293045.bound (LeftBound293045.actual selector witness) := by
  exact .transfer (LeftBound293045.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285728.bound LeftBound293045.bound
def bound : CoeffClass := .finite ⟨32190378816049003834595889643520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285728.bound, LeftBound293045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285728.actual selector witness) * (LeftBound293045.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293046

namespace LeftBound293057
def owner : Owner := ⟨.program ⟨257⟩, ⟨60574⟩⟩
def transferEvent : Nat := 293057
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 293055 .coefficient) (.value (.predecessor 1 293056 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293055 .coefficient)
      LeftAuthority293053.bound (LeftAuthority293053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293053.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293053.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293056 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority293053.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority293053.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority293053.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound293057

namespace LeftBound293061
def owner : Owner := ⟨.program ⟨257⟩, ⟨60575⟩⟩
def transferEvent : Nat := 293061
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 293059 .coefficient) (.predecessor 1 293060 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293059 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 293060 .coefficient)
      LeftBound293057.bound (LeftBound293057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound293057.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound293057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound293057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293061

namespace LeftBound293062
def owner : Owner := ⟨.program ⟨257⟩, ⟨60575⟩⟩
def transferEvent : Nat := 293062
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨60572⟩⟩]⟩ [⟨.result 293054 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293054 .coefficient)
      LeftAuthority293053.bound (LeftAuthority293053.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨60572⟩⟩) (rawTerms := some (Proof.Events1144.exact293054RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293053.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293053.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority293053.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority293053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority293053.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound293062

namespace LeftBound293063
def owner : Owner := ⟨.program ⟨257⟩, ⟨60575⟩⟩
def transferEvent : Nat := 293063
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 293062) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 293062)
      LeftBound293062.bound (LeftBound293062.actual selector witness) := by
  exact .transfer (LeftBound293062.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound293062.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound293062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound293062.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound293063

namespace LeftBound293158
def owner : Owner := ⟨.program ⟨257⟩, ⟨59781⟩⟩
def transferEvent : Nat := 293158
def frameStart : Nat := 293119
def rule : BoundRule := .identity (.predecessor 0 293157 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 293157 .coefficient)
      LeftAuthority293155.bound (LeftAuthority293155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1145.exact293156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority293155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority293155.derived selector witness)

def rawBound : CoeffClass := LeftAuthority293155.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority293155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority293155.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound293158

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
