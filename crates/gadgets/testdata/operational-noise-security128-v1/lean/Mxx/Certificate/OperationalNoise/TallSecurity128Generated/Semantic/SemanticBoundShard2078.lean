import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2028
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2077

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound305989
def owner : Owner := ⟨.program ⟨257⟩, ⟨28038⟩⟩
def transferEvent : Nat := 305989
def frameStart : Nat := 305901
def rule : BoundRule := .sum [.predecessor 0 305987 .coefficient, .predecessor 1 305988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 305987 .coefficient)
      LeftBound305985.bound (LeftBound305985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact305986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305985.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 305988 .coefficient)
      LeftBound305966.bound (LeftBound305966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact305971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound305985.bound, LeftBound305966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound305985.bound, LeftBound305966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound305985.actual selector witness, LeftBound305966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound305989

namespace LeftBound306002
def owner : Owner := ⟨.program ⟨257⟩, ⟨28036⟩⟩
def transferEvent : Nat := 306002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 306000 .coefficient, .predecessor 1 306001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306000 .coefficient)
      LeftBound305855.bound (LeftBound305855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact305999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306001 .coefficient)
      LeftBound305838.bound (LeftBound305838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1194.exact305845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound305855.bound, LeftBound305838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound305855.bound, LeftBound305838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound305855.actual selector witness, LeftBound305838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound306002

namespace LeftBound306005
def owner : Owner := ⟨.program ⟨257⟩, ⟨28036⟩⟩
def transferEvent : Nat := 306005
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 305999 .summary, .result 305845 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305999 .summary)
      LeftBound305857.bound (LeftBound305857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26955⟩⟩) (rawTerms := some (Proof.Events1195.exact305999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305845 .summary)
      LeftBound305840.bound (LeftBound305840.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28035⟩⟩) (rawTerms := some (Proof.Events1194.exact305845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound305857.bound, LeftBound305840.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound305857.bound, LeftBound305840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound305857.actual selector witness, LeftBound305840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound306005

namespace LeftBound306009
def owner : Owner := ⟨.program ⟨257⟩, ⟨28037⟩⟩
def transferEvent : Nat := 306009
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 306007 .coefficient) (.predecessor 1 306008 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306007 .coefficient)
      LeftBound306002.bound (LeftBound306002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306008 .coefficient)
      LeftBound15681.bound (LeftBound15681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound306002.bound LeftBound15681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound306002.bound, LeftBound15681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound306002.actual selector witness) * (LeftBound15681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306009

namespace LeftBound306010
def owner : Owner := ⟨.program ⟨257⟩, ⟨28037⟩⟩
def transferEvent : Nat := 306010
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩ [⟨.result 15678 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15678 .coefficient)
      LeftAuthority15677.bound (LeftAuthority15677.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7169⟩⟩) (rawTerms := some (Proof.Events061.exact15678RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15677.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15677.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15677.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound306010

namespace LeftBound306011
def owner : Owner := ⟨.program ⟨257⟩, ⟨28037⟩⟩
def transferEvent : Nat := 306011
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 306006 .summary) (.transfer 306010) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306006 .summary)
      LeftBound306005.bound (LeftBound306005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28036⟩⟩) (rawTerms := some (Proof.Events1195.exact306006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 306010)
      LeftBound306010.bound (LeftBound306010.actual selector witness) := by
  exact .transfer (LeftBound306010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound306005.bound LeftBound306010.bound
def bound : CoeffClass := .finite ⟨345654216875549026890382321864211871825920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound306005.bound, LeftBound306010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound306005.actual selector witness) * (LeftBound306010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306011

namespace LeftBound306026
def owner : Owner := ⟨.program ⟨257⟩, ⟨69374⟩⟩
def transferEvent : Nat := 306026
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 306024 .coefficient) (.predecessor 1 306025 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306024 .coefficient)
      LeftBound298825.bound (LeftBound298825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306025 .coefficient)
      LeftAuthority306022.bound (LeftAuthority306022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298825.bound LeftAuthority306022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298825.bound, LeftAuthority306022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298825.actual selector witness) * (LeftAuthority306022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306026

namespace LeftBound306027
def owner : Owner := ⟨.program ⟨257⟩, ⟨69374⟩⟩
def transferEvent : Nat := 306027
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨69372⟩⟩]⟩ [⟨.result 306023 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306023 .coefficient)
      LeftAuthority306022.bound (LeftAuthority306022.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨69372⟩⟩) (rawTerms := some (Proof.Events1195.exact306023RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306022.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority306022.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority306022.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound306027

namespace LeftBound306028
def owner : Owner := ⟨.program ⟨257⟩, ⟨69374⟩⟩
def transferEvent : Nat := 306028
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 298829 .summary) (.transfer 306027) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298829 .summary)
      LeftBound298828.bound (LeftBound298828.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69132⟩⟩) (rawTerms := some (Proof.Events1167.exact298829RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 306027)
      LeftBound306027.bound (LeftBound306027.actual selector witness) := by
  exact .transfer (LeftBound306027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298828.bound LeftBound306027.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298828.bound, LeftBound306027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298828.actual selector witness) * (LeftBound306027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306028

namespace LeftBound306039
def owner : Owner := ⟨.program ⟨257⟩, ⟨67875⟩⟩
def transferEvent : Nat := 306039
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 306037 .coefficient) (.value (.predecessor 1 306038 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306037 .coefficient)
      LeftAuthority306035.bound (LeftAuthority306035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306038 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority306035.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306035.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority306035.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound306039

namespace LeftBound306043
def owner : Owner := ⟨.program ⟨257⟩, ⟨67876⟩⟩
def transferEvent : Nat := 306043
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 306041 .coefficient) (.predecessor 1 306042 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306041 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306042 .coefficient)
      LeftBound306039.bound (LeftBound306039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound306039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound306039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound306039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306043

namespace LeftBound306044
def owner : Owner := ⟨.program ⟨257⟩, ⟨67876⟩⟩
def transferEvent : Nat := 306044
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨67873⟩⟩]⟩ [⟨.result 306036 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306036 .coefficient)
      LeftAuthority306035.bound (LeftAuthority306035.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67873⟩⟩) (rawTerms := some (Proof.Events1195.exact306036RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306035.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority306035.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority306035.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound306044

namespace LeftBound306045
def owner : Owner := ⟨.program ⟨257⟩, ⟨67876⟩⟩
def transferEvent : Nat := 306045
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295195 .summary) (.transfer 306044) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295195 .summary)
      LeftBound295193.bound (LeftBound295193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨2380⟩⟩) (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 306044)
      LeftBound306044.bound (LeftBound306044.actual selector witness) := by
  exact .transfer (LeftBound306044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295193.bound LeftBound306044.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295193.bound, LeftBound306044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295193.actual selector witness) * (LeftBound306044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound306045

namespace LeftBound306116
def owner : Owner := ⟨.program ⟨257⟩, ⟨65709⟩⟩
def transferEvent : Nat := 306116
def frameStart : Nat := 306089
def rule : BoundRule := .identity (.predecessor 0 306115 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306115 .coefficient)
      LeftAuthority306113.bound (LeftAuthority306113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority306113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority306113.derived selector witness)

def rawBound : CoeffClass := LeftAuthority306113.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority306113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority306113.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound306116

namespace LeftBound306133
def owner : Owner := ⟨.program ⟨257⟩, ⟨68967⟩⟩
def transferEvent : Nat := 306133
def frameStart : Nat := 306089
def rule : BoundRule := .sum [.predecessor 0 306131 .coefficient, .predecessor 1 306132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306131 .coefficient)
      LeftBound306116.bound (LeftBound306116.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound306116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 306132 .coefficient)
      LeftAuthority306129.bound (LeftAuthority306129.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority306129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound306116.bound, LeftAuthority306129.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound306116.bound, LeftAuthority306129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound306116.actual selector witness, LeftAuthority306129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound306133

namespace LeftBound306136
def owner : Owner := ⟨.program ⟨257⟩, ⟨68968⟩⟩
def transferEvent : Nat := 306136
def frameStart : Nat := 306089
def rule : BoundRule := .identity (.predecessor 0 306135 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 306135 .coefficient)
      LeftBound306133.bound (LeftBound306133.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound306133.derived selector witness)

def rawBound : CoeffClass := LeftBound306133.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound306133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound306133.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound306136

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
