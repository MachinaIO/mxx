import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1899
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1933
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1980

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound292815
def owner : Owner := ⟨.program ⟨257⟩, ⟨69692⟩⟩
def transferEvent : Nat := 292815
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 292813 .coefficient) (.predecessor 1 292814 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292813 .coefficient)
      LeftBound292808.bound (LeftBound292808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292814 .coefficient)
      LeftBound15701.bound (LeftBound15701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292808.bound LeftBound15701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292808.bound, LeftBound15701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292808.actual selector witness) * (LeftBound15701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292815

namespace LeftBound292816
def owner : Owner := ⟨.program ⟨257⟩, ⟨69692⟩⟩
def transferEvent : Nat := 292816
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩ [⟨.result 15698 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15698 .coefficient)
      LeftAuthority15697.bound (LeftAuthority15697.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7173⟩⟩) (rawTerms := some (Proof.Events061.exact15698RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15697.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15697.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15697.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound292816

namespace LeftBound292817
def owner : Owner := ⟨.program ⟨257⟩, ⟨69692⟩⟩
def transferEvent : Nat := 292817
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 292812 .summary) (.transfer 292816) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292812 .summary)
      LeftBound292811.bound (LeftBound292811.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69691⟩⟩) (rawTerms := some (Proof.Events1143.exact292812RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 292816)
      LeftBound292816.bound (LeftBound292816.actual selector witness) := by
  exact .transfer (LeftBound292816.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292811.bound LeftBound292816.bound
def bound : CoeffClass := .finite ⟨345652107504950247116658231350078126161920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292811.bound, LeftBound292816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292811.actual selector witness) * (LeftBound292816.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292817

namespace LeftBound292832
def owner : Owner := ⟨.program ⟨257⟩, ⟨64681⟩⟩
def transferEvent : Nat := 292832
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 292830 .coefficient) (.predecessor 1 292831 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292830 .coefficient)
      LeftBound285245.bound (LeftBound285245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1114.exact285249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292831 .coefficient)
      LeftAuthority292828.bound (LeftAuthority292828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285245.bound LeftAuthority292828.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285245.bound, LeftAuthority292828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285245.actual selector witness) * (LeftAuthority292828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292832

namespace LeftBound292833
def owner : Owner := ⟨.program ⟨257⟩, ⟨64681⟩⟩
def transferEvent : Nat := 292833
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨64679⟩⟩]⟩ [⟨.result 292829 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292829 .coefficient)
      LeftAuthority292828.bound (LeftAuthority292828.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨64679⟩⟩) (rawTerms := some (Proof.Events1143.exact292829RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292828.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority292828.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority292828.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound292833

namespace LeftBound292834
def owner : Owner := ⟨.program ⟨257⟩, ⟨64681⟩⟩
def transferEvent : Nat := 292834
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 285249 .summary) (.transfer 292833) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285249 .summary)
      LeftBound285248.bound (LeftBound285248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64375⟩⟩) (rawTerms := some (Proof.Events1114.exact285249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 292833)
      LeftBound292833.bound (LeftBound292833.actual selector witness) := by
  exact .transfer (LeftBound292833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285248.bound LeftBound292833.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285248.bound, LeftBound292833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285248.actual selector witness) * (LeftBound292833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292834

namespace LeftBound292845
def owner : Owner := ⟨.program ⟨257⟩, ⟨63554⟩⟩
def transferEvent : Nat := 292845
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 292843 .coefficient) (.value (.predecessor 1 292844 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292843 .coefficient)
      LeftAuthority292841.bound (LeftAuthority292841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292844 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority292841.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292841.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority292841.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound292845

namespace LeftBound292849
def owner : Owner := ⟨.program ⟨257⟩, ⟨63555⟩⟩
def transferEvent : Nat := 292849
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 292847 .coefficient) (.predecessor 1 292848 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292847 .coefficient)
      LeftBound280742.bound (LeftBound280742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292848 .coefficient)
      LeftBound292845.bound (LeftBound292845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292845.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280742.bound LeftBound292845.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280742.bound, LeftBound292845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280742.actual selector witness) * (LeftBound292845.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292849

namespace LeftBound292850
def owner : Owner := ⟨.program ⟨257⟩, ⟨63555⟩⟩
def transferEvent : Nat := 292850
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨63552⟩⟩]⟩ [⟨.result 292842 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292842 .coefficient)
      LeftAuthority292841.bound (LeftAuthority292841.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨63552⟩⟩) (rawTerms := some (Proof.Events1143.exact292842RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292841.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority292841.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority292841.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound292850

namespace LeftBound292851
def owner : Owner := ⟨.program ⟨257⟩, ⟨63555⟩⟩
def transferEvent : Nat := 292851
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 280745 .summary) (.transfer 292850) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280745 .summary)
      LeftBound280743.bound (LeftBound280743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5491⟩⟩) (rawTerms := some (Proof.Events1096.exact280745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 292850)
      LeftBound292850.bound (LeftBound292850.actual selector witness) := by
  exact .transfer (LeftBound292850.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound280743.bound LeftBound292850.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280743.bound, LeftBound292850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound280743.actual selector witness) * (LeftBound292850.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292851

namespace LeftBound292946
def owner : Owner := ⟨.program ⟨257⟩, ⟨62761⟩⟩
def transferEvent : Nat := 292946
def frameStart : Nat := 292907
def rule : BoundRule := .identity (.predecessor 0 292945 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292945 .coefficient)
      LeftAuthority292943.bound (LeftAuthority292943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292943.derived selector witness)

def rawBound : CoeffClass := LeftAuthority292943.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority292943.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound292946

namespace LeftBound292963
def owner : Owner := ⟨.program ⟨257⟩, ⟨64262⟩⟩
def transferEvent : Nat := 292963
def frameStart : Nat := 292907
def rule : BoundRule := .sum [.predecessor 0 292961 .coefficient, .predecessor 1 292962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292961 .coefficient)
      LeftBound292946.bound (LeftBound292946.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound292946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292962 .coefficient)
      LeftAuthority292959.bound (LeftAuthority292959.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority292959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292946.bound, LeftAuthority292959.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292946.bound, LeftAuthority292959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292946.actual selector witness, LeftAuthority292959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292963

namespace LeftBound292966
def owner : Owner := ⟨.program ⟨257⟩, ⟨64263⟩⟩
def transferEvent : Nat := 292966
def frameStart : Nat := 292907
def rule : BoundRule := .identity (.predecessor 0 292965 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292965 .coefficient)
      LeftBound292963.bound (LeftBound292963.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound292963.derived selector witness)

def rawBound : CoeffClass := LeftBound292963.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound292963.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound292966

namespace LeftBound292972
def owner : Owner := ⟨.program ⟨257⟩, ⟨64264⟩⟩
def transferEvent : Nat := 292972
def frameStart : Nat := 292907
def rule : BoundRule := .product (.predecessor 0 292970 .coefficient) (.predecessor 1 292971 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292970 .coefficient)
      LeftAuthority292968.bound (LeftAuthority292968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292971 .coefficient)
      LeftBound292966.bound (LeftBound292966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292966.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority292968.bound LeftBound292966.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292968.bound, LeftBound292966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority292968.actual selector witness) * (LeftBound292966.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292972

namespace LeftBound292980
def owner : Owner := ⟨.program ⟨257⟩, ⟨64265⟩⟩
def transferEvent : Nat := 292980
def frameStart : Nat := 292907
def rule : BoundRule := .sum [.predecessor 0 292978 .coefficient, .predecessor 1 292979 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292978 .coefficient)
      LeftAuthority292976.bound (LeftAuthority292976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292979 .coefficient)
      LeftBound292972.bound (LeftBound292972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority292976.bound, LeftBound292972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292976.bound, LeftBound292972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority292976.actual selector witness, LeftBound292972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292980

namespace LeftBound292984
def owner : Owner := ⟨.program ⟨257⟩, ⟨64680⟩⟩
def transferEvent : Nat := 292984
def frameStart : Nat := 292907
def rule : BoundRule := .product (.predecessor 0 292982 .coefficient) (.predecessor 1 292983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292982 .coefficient)
      LeftBound292980.bound (LeftBound292980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292983 .coefficient)
      LeftAuthority292957.bound (LeftAuthority292957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact292958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292957.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292980.bound LeftAuthority292957.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292980.bound, LeftAuthority292957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292980.actual selector witness) * (LeftAuthority292957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292984

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
