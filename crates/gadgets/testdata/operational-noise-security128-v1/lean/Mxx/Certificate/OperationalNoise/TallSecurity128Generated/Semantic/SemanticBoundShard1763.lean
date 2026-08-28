import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1696
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1697
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1699
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1762

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound260154
def owner : Owner := ⟨.program ⟨257⟩, ⟨69794⟩⟩
def transferEvent : Nat := 260154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 260150 .summary, .result 251876 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260150 .summary)
      LeftBound260149.bound (LeftBound260149.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69793⟩⟩) (rawTerms := some (Proof.Events1016.exact260150RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251876 .summary)
      LeftBound251875.bound (LeftBound251875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49907⟩⟩) (rawTerms := some (Proof.Events983.exact251876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound260149.bound, LeftBound251875.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260149.bound, LeftBound251875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound260149.actual selector witness, LeftBound251875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound260154

namespace LeftBound260158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def transferEvent : Nat := 260158
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 260156 .coefficient) (.predecessor 1 260157 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260156 .coefficient)
      LeftBound260153.bound (LeftBound260153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260157 .coefficient)
      LeftAuthority251377.bound (LeftAuthority251377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority251377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority251377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound260153.bound LeftAuthority251377.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260153.bound, LeftAuthority251377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound260153.actual selector witness) * (LeftAuthority251377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound260158

namespace LeftBound260159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def transferEvent : Nat := 260159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩ [⟨.result 251378 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251378 .coefficient)
      LeftAuthority251377.bound (LeftAuthority251377.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71082⟩⟩) (rawTerms := some (Proof.Events981.exact251378RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority251377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority251377.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority251377.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority251377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority251377.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound260159

namespace LeftBound260160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def transferEvent : Nat := 260160
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 260155 .summary) (.transfer 260159) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260155 .summary)
      LeftBound260154.bound (LeftBound260154.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69794⟩⟩) (rawTerms := some (Proof.Events1016.exact260155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound260154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 260159)
      LeftBound260159.bound (LeftBound260159.actual selector witness) := by
  exact .transfer (LeftBound260159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound260154.bound LeftBound260159.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound260154.bound, LeftBound260159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound260154.actual selector witness) * (LeftBound260159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound260160

namespace LeftBound260239
def owner : Owner := ⟨.program ⟨257⟩, ⟨68322⟩⟩
def transferEvent : Nat := 260239
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 260237 .coefficient) (.value (.predecessor 1 260238 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260237 .coefficient)
      LeftAuthority260235.bound (LeftAuthority260235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260238 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority260235.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority260235.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority260235.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound260239

namespace LeftBound260243
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def transferEvent : Nat := 260243
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 260241 .coefficient) (.predecessor 1 260242 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 260241 .coefficient)
      LeftBound251492.bound (LeftBound251492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 260242 .coefficient)
      LeftBound260239.bound (LeftBound260239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1016.exact260240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound260239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound260239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251492.bound LeftBound260239.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251492.bound, LeftBound260239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251492.actual selector witness) * (LeftBound260239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound260243

namespace LeftBound260244
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def transferEvent : Nat := 260244
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩ [⟨.result 260236 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 260236 .coefficient)
      LeftAuthority260235.bound (LeftAuthority260235.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68320⟩⟩) (rawTerms := some (Proof.Events1016.exact260236RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260235.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority260235.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority260235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority260235.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound260244

namespace LeftBound260245
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def transferEvent : Nat := 260245
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 251495 .summary) (.transfer 260244) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251495 .summary)
      LeftBound251493.bound (LeftBound251493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 260244)
      LeftBound260244.bound (LeftBound260244.actual selector witness) := by
  exact .transfer (LeftBound260244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251493.bound LeftBound260244.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251493.bound, LeftBound260244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251493.actual selector witness) * (LeftBound260244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound260245

namespace LeftBound261273
def owner : Owner := ⟨.program ⟨257⟩, ⟨18772⟩⟩
def transferEvent : Nat := 261273
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261271 .coefficient, .predecessor 1 261272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261271 .coefficient)
      LeftAuthority261269.bound (LeftAuthority261269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261272 .coefficient)
      LeftAuthority261246.bound (LeftAuthority261246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority261269.bound, LeftAuthority261246.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261269.bound, LeftAuthority261246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority261269.actual selector witness, LeftAuthority261246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261273

namespace LeftBound261277
def owner : Owner := ⟨.program ⟨257⟩, ⟨21992⟩⟩
def transferEvent : Nat := 261277
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261275 .coefficient, .predecessor 1 261276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261275 .coefficient)
      LeftBound261273.bound (LeftBound261273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261276 .coefficient)
      LeftAuthority261223.bound (LeftAuthority261223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261223.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261273.bound, LeftAuthority261223.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261273.bound, LeftAuthority261223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261273.actual selector witness, LeftAuthority261223.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261277

namespace LeftBound261281
def owner : Owner := ⟨.program ⟨257⟩, ⟨32012⟩⟩
def transferEvent : Nat := 261281
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261279 .coefficient, .predecessor 1 261280 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261279 .coefficient)
      LeftBound261277.bound (LeftBound261277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261280 .coefficient)
      LeftAuthority261200.bound (LeftAuthority261200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261277.bound, LeftAuthority261200.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261277.bound, LeftAuthority261200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261277.actual selector witness, LeftAuthority261200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261281

namespace LeftBound261285
def owner : Owner := ⟨.program ⟨257⟩, ⟨51067⟩⟩
def transferEvent : Nat := 261285
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261283 .coefficient, .predecessor 1 261284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261283 .coefficient)
      LeftBound261281.bound (LeftBound261281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261284 .coefficient)
      LeftAuthority261177.bound (LeftAuthority261177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261281.bound, LeftAuthority261177.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261281.bound, LeftAuthority261177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261281.actual selector witness, LeftAuthority261177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261285

namespace LeftBound261289
def owner : Owner := ⟨.program ⟨257⟩, ⟨54047⟩⟩
def transferEvent : Nat := 261289
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261287 .coefficient, .predecessor 1 261288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261287 .coefficient)
      LeftBound261285.bound (LeftBound261285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261288 .coefficient)
      LeftAuthority261154.bound (LeftAuthority261154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261154.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261285.bound, LeftAuthority261154.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261285.bound, LeftAuthority261154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261285.actual selector witness, LeftAuthority261154.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261289

namespace LeftBound261293
def owner : Owner := ⟨.program ⟨257⟩, ⟨57027⟩⟩
def transferEvent : Nat := 261293
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261291 .coefficient, .predecessor 1 261292 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261291 .coefficient)
      LeftBound261289.bound (LeftBound261289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261292 .coefficient)
      LeftAuthority261131.bound (LeftAuthority261131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261131.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261289.bound, LeftAuthority261131.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261289.bound, LeftAuthority261131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261289.actual selector witness, LeftAuthority261131.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261293

namespace LeftBound261297
def owner : Owner := ⟨.program ⟨257⟩, ⟨60007⟩⟩
def transferEvent : Nat := 261297
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261295 .coefficient, .predecessor 1 261296 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261295 .coefficient)
      LeftBound261293.bound (LeftBound261293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261296 .coefficient)
      LeftAuthority261108.bound (LeftAuthority261108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact261109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261293.bound, LeftAuthority261108.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261293.bound, LeftAuthority261108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261293.actual selector witness, LeftAuthority261108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261297

namespace LeftBound261301
def owner : Owner := ⟨.program ⟨257⟩, ⟨62987⟩⟩
def transferEvent : Nat := 261301
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261299 .coefficient, .predecessor 1 261300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261299 .coefficient)
      LeftBound261297.bound (LeftBound261297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261300 .coefficient)
      LeftAuthority261085.bound (LeftAuthority261085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact261086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261297.bound, LeftAuthority261085.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261297.bound, LeftAuthority261085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261297.actual selector witness, LeftAuthority261085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261301

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
