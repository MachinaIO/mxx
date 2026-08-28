import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard119
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1945

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound286927
def owner : Owner := ⟨.program ⟨257⟩, ⟨50386⟩⟩
def transferEvent : Nat := 286927
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286925 .coefficient) (.predecessor 1 286926 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286925 .coefficient)
      LeftBound286921.bound (LeftBound286921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286926 .coefficient)
      LeftAuthority13853.bound (LeftAuthority13853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13853.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound286921.bound LeftAuthority13853.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286921.bound, LeftAuthority13853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound286921.actual selector witness) * (LeftAuthority13853.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286927

namespace LeftBound286928
def owner : Owner := ⟨.program ⟨257⟩, ⟨50386⟩⟩
def transferEvent : Nat := 286928
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨50383⟩⟩], []⟩ [⟨.result 13854 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 13854 .coefficient)
      LeftAuthority13853.bound (LeftAuthority13853.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50383⟩⟩) (rawTerms := some (Proof.Events054.exact13854RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13853.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13853.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority13853.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound286928

namespace LeftBound286929
def owner : Owner := ⟨.program ⟨257⟩, ⟨50386⟩⟩
def transferEvent : Nat := 286929
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 286924 .summary) (.transfer 286928) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286924 .summary)
      LeftBound286922.bound (LeftBound286922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24461⟩⟩) (rawTerms := some (Proof.Events1120.exact286924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 286928)
      LeftBound286928.bound (LeftBound286928.actual selector witness) := by
  exact .transfer (LeftBound286928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound286922.bound LeftBound286928.bound
def bound : CoeffClass := .finite ⟨8519680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286922.bound, LeftBound286928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound286922.actual selector witness) * (LeftBound286928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286929

namespace LeftBound286935
def owner : Owner := ⟨.program ⟨257⟩, ⟨50387⟩⟩
def transferEvent : Nat := 286935
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 286933 .coefficient) (.predecessor 1 286934 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286933 .coefficient)
      LeftAuthority13853.bound (LeftAuthority13853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286934 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13853.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13853.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13853.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound286935

namespace LeftBound286940
def owner : Owner := ⟨.program ⟨257⟩, ⟨7910⟩⟩
def transferEvent : Nat := 286940
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286938 .coefficient) (.predecessor 1 286939 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286938 .coefficient)
      LeftBound280522.bound (LeftBound280522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286939 .coefficient)
      LeftBound23633.bound (LeftBound23633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound280522.bound LeftBound23633.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280522.bound, LeftBound23633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound280522.actual selector witness) * (LeftBound23633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286940

namespace LeftBound286945
def owner : Owner := ⟨.program ⟨257⟩, ⟨50388⟩⟩
def transferEvent : Nat := 286945
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286943 .coefficient, .predecessor 1 286944 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286943 .coefficient)
      LeftBound286940.bound (LeftBound286940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286944 .coefficient)
      LeftBound286935.bound (LeftBound286935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286940.bound, LeftBound286935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286940.bound, LeftBound286935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286940.actual selector witness, LeftBound286935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286945

namespace LeftBound286949
def owner : Owner := ⟨.program ⟨257⟩, ⟨50389⟩⟩
def transferEvent : Nat := 286949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286947 .coefficient, .predecessor 1 286948 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286947 .coefficient)
      LeftBound286945.bound (LeftBound286945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286948 .coefficient)
      LeftBound23625.bound (LeftBound23625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286945.bound, LeftBound23625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286945.bound, LeftBound23625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286945.actual selector witness, LeftBound23625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286949

namespace LeftBound286950
def owner : Owner := ⟨.program ⟨257⟩, ⟨50389⟩⟩
def transferEvent : Nat := 286950
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨114⟩⟩]⟩ [⟨.result 23626 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23626 .coefficient)
      LeftBound23625.bound (LeftBound23625.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨114⟩⟩) (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23625.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23625.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23625.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound286950

namespace LeftBound286955
def owner : Owner := ⟨.program ⟨257⟩, ⟨50390⟩⟩
def transferEvent : Nat := 286955
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286953 .coefficient) (.predecessor 1 286954 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286953 .coefficient)
      LeftBound286949.bound (LeftBound286949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286954 .coefficient)
      LeftBound23622.bound (LeftBound23622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286949.bound LeftBound23622.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286949.bound, LeftBound23622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286949.actual selector witness) * (LeftBound23622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286955

namespace LeftBound286956
def owner : Owner := ⟨.program ⟨257⟩, ⟨50390⟩⟩
def transferEvent : Nat := 286956
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩ [⟨.result 23619 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23619 .coefficient)
      LeftAuthority23618.bound (LeftAuthority23618.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9580⟩⟩) (rawTerms := some (Proof.Events092.exact23619RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23618.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23618.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority23618.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound286956

namespace LeftBound286957
def owner : Owner := ⟨.program ⟨257⟩, ⟨50390⟩⟩
def transferEvent : Nat := 286957
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 286952 .summary) (.transfer 286956) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286952 .summary)
      LeftBound286950.bound (LeftBound286950.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50389⟩⟩) (rawTerms := some (Proof.Events1120.exact286952RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 286956)
      LeftBound286956.bound (LeftBound286956.actual selector witness) := by
  exact .transfer (LeftBound286956.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286950.bound LeftBound286956.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286950.bound, LeftBound286956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286950.actual selector witness) * (LeftBound286956.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286957

namespace LeftBound286965
def owner : Owner := ⟨.program ⟨257⟩, ⟨50391⟩⟩
def transferEvent : Nat := 286965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 286963 .coefficient, .predecessor 1 286964 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286963 .coefficient)
      LeftBound286955.bound (LeftBound286955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286964 .coefficient)
      LeftBound286927.bound (LeftBound286927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286955.bound, LeftBound286927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286955.bound, LeftBound286927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286955.actual selector witness, LeftBound286927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286965

namespace LeftBound286967
def owner : Owner := ⟨.program ⟨257⟩, ⟨50391⟩⟩
def transferEvent : Nat := 286967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 286962 .summary, .result 286932 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286962 .summary)
      LeftBound286957.bound (LeftBound286957.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50390⟩⟩) (rawTerms := some (Proof.Events1120.exact286962RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286932 .summary)
      LeftBound286929.bound (LeftBound286929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50386⟩⟩) (rawTerms := some (Proof.Events1120.exact286932RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound286957.bound, LeftBound286929.bound]
def bound : CoeffClass := .finite ⟨279181393920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286957.bound, LeftBound286929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound286957.actual selector witness, LeftBound286929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound286967

namespace LeftBound286971
def owner : Owner := ⟨.program ⟨257⟩, ⟨52454⟩⟩
def transferEvent : Nat := 286971
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 286969 .coefficient) (.predecessor 1 286970 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 286969 .coefficient)
      LeftBound286965.bound (LeftBound286965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 286970 .coefficient)
      LeftAuthority286903.bound (LeftAuthority286903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286903.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286965.bound LeftAuthority286903.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286965.bound, LeftAuthority286903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286965.actual selector witness) * (LeftAuthority286903.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286971

namespace LeftBound286972
def owner : Owner := ⟨.program ⟨257⟩, ⟨52454⟩⟩
def transferEvent : Nat := 286972
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52453⟩⟩]⟩ [⟨.result 286904 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286904 .coefficient)
      LeftAuthority286903.bound (LeftAuthority286903.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52453⟩⟩) (rawTerms := some (Proof.Events1120.exact286904RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority286903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority286903.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority286903.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority286903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority286903.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound286972

namespace LeftBound286973
def owner : Owner := ⟨.program ⟨257⟩, ⟨52454⟩⟩
def transferEvent : Nat := 286973
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 286968 .summary) (.transfer 286972) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286968 .summary)
      LeftBound286967.bound (LeftBound286967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50391⟩⟩) (rawTerms := some (Proof.Events1120.exact286968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 286972)
      LeftBound286972.bound (LeftBound286972.actual selector witness) := by
  exact .transfer (LeftBound286972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound286967.bound LeftBound286972.bound
def bound : CoeffClass := .finite ⟨2997687391345233100800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound286967.bound, LeftBound286972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound286967.actual selector witness) * (LeftBound286972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound286973

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
