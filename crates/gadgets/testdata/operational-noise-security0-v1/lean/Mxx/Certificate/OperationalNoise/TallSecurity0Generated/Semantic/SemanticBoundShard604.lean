import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard096
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard603

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88136
def owner : Owner := ⟨.program ⟨214⟩, ⟨9403⟩⟩
def transferEvent : Nat := 88136
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88134 .coefficient, .predecessor 1 88135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88134 .coefficient)
      LeftBound88132.bound (LeftBound88132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88132.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88135 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88132.bound, LeftBound15021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88132.bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88132.actual selector witness, LeftBound15021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88136

namespace LeftBound88137
def owner : Owner := ⟨.program ⟨214⟩, ⟨9403⟩⟩
def transferEvent : Nat := 88137
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩ [⟨.result 15022 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15022 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨85⟩⟩) (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound15021.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound15021.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88137

namespace LeftBound88142
def owner : Owner := ⟨.program ⟨214⟩, ⟨9404⟩⟩
def transferEvent : Nat := 88142
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88140 .coefficient) (.predecessor 1 88141 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88140 .coefficient)
      LeftBound88136.bound (LeftBound88136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88141 .coefficient)
      LeftBound15018.bound (LeftBound15018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88136.bound LeftBound15018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88136.bound, LeftBound15018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88136.actual selector witness) * (LeftBound15018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88142

namespace LeftBound88143
def owner : Owner := ⟨.program ⟨214⟩, ⟨9404⟩⟩
def transferEvent : Nat := 88143
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩ [⟨.result 15015 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15015 .coefficient)
      LeftAuthority15014.bound (LeftAuthority15014.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7831⟩⟩) (rawTerms := some (Proof.Events058.exact15015RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15014.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15014.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15014.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88143

namespace LeftBound88144
def owner : Owner := ⟨.program ⟨214⟩, ⟨9404⟩⟩
def transferEvent : Nat := 88144
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88139 .summary) (.transfer 88143) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88139 .summary)
      LeftBound88137.bound (LeftBound88137.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9403⟩⟩) (rawTerms := some (Proof.Events344.exact88139RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88143)
      LeftBound88143.bound (LeftBound88143.actual selector witness) := by
  exact .transfer (LeftBound88143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88137.bound LeftBound88143.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88137.bound, LeftBound88143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88137.actual selector witness) * (LeftBound88143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88144

namespace LeftBound88152
def owner : Owner := ⟨.program ⟨214⟩, ⟨10487⟩⟩
def transferEvent : Nat := 88152
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88150 .coefficient, .predecessor 1 88151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88150 .coefficient)
      LeftBound88142.bound (LeftBound88142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88151 .coefficient)
      LeftBound88114.bound (LeftBound88114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88142.bound, LeftBound88114.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88142.bound, LeftBound88114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88142.actual selector witness, LeftBound88114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88152

namespace LeftBound88154
def owner : Owner := ⟨.program ⟨214⟩, ⟨10487⟩⟩
def transferEvent : Nat := 88154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88149 .summary, .result 88119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88149 .summary)
      LeftBound88144.bound (LeftBound88144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9404⟩⟩) (rawTerms := some (Proof.Events344.exact88149RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88119 .summary)
      LeftBound88116.bound (LeftBound88116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10486⟩⟩) (rawTerms := some (Proof.Events344.exact88119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88144.bound, LeftBound88116.bound]
def bound : CoeffClass := .finite ⟨95422080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88144.bound, LeftBound88116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88144.actual selector witness, LeftBound88116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88154

namespace LeftBound88158
def owner : Owner := ⟨.program ⟨214⟩, ⟨24912⟩⟩
def transferEvent : Nat := 88158
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88156 .coefficient) (.predecessor 1 88157 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88156 .coefficient)
      LeftBound88152.bound (LeftBound88152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88157 .coefficient)
      LeftAuthority88090.bound (LeftAuthority88090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88090.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88152.bound LeftAuthority88090.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88152.bound, LeftAuthority88090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88152.actual selector witness) * (LeftAuthority88090.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88158

namespace LeftBound88159
def owner : Owner := ⟨.program ⟨214⟩, ⟨24912⟩⟩
def transferEvent : Nat := 88159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩ [⟨.result 88091 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88091 .coefficient)
      LeftAuthority88090.bound (LeftAuthority88090.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24911⟩⟩) (rawTerms := some (Proof.Events344.exact88091RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88090.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority88090.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88090.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88159

namespace LeftBound88160
def owner : Owner := ⟨.program ⟨214⟩, ⟨24912⟩⟩
def transferEvent : Nat := 88160
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88155 .summary) (.transfer 88159) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88155 .summary)
      LeftBound88154.bound (LeftBound88154.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10487⟩⟩) (rawTerms := some (Proof.Events344.exact88155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88159)
      LeftBound88159.bound (LeftBound88159.actual selector witness) := by
  exact .transfer (LeftBound88159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88154.bound LeftBound88159.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88154.bound, LeftBound88159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88154.actual selector witness) * (LeftBound88159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88160

namespace LeftBound88171
def owner : Owner := ⟨.program ⟨214⟩, ⟨19026⟩⟩
def transferEvent : Nat := 88171
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 88169 .coefficient) (.value (.predecessor 1 88170 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88169 .coefficient)
      LeftAuthority88167.bound (LeftAuthority88167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88170 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority88167.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88167.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88167.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88171

namespace LeftBound88175
def owner : Owner := ⟨.program ⟨214⟩, ⟨19027⟩⟩
def transferEvent : Nat := 88175
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88173 .coefficient) (.predecessor 1 88174 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88173 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88174 .coefficient)
      LeftBound88171.bound (LeftBound88171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88171.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound88171.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound88171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound88171.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88175

namespace LeftBound88176
def owner : Owner := ⟨.program ⟨214⟩, ⟨19027⟩⟩
def transferEvent : Nat := 88176
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩ [⟨.result 88168 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88168 .coefficient)
      LeftAuthority88167.bound (LeftAuthority88167.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19024⟩⟩) (rawTerms := some (Proof.Events344.exact88168RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88167.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority88167.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88167.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88176

namespace LeftBound88177
def owner : Owner := ⟨.program ⟨214⟩, ⟨19027⟩⟩
def transferEvent : Nat := 88177
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 88176) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88176)
      LeftBound88176.bound (LeftBound88176.actual selector witness) := by
  exact .transfer (LeftBound88176.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound88176.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound88176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound88176.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88177

namespace LeftBound88256
def owner : Owner := ⟨.program ⟨214⟩, ⟨10481⟩⟩
def transferEvent : Nat := 88256
def frameStart : Nat := 88227
def rule : BoundRule := .product (.predecessor 0 88254 .coefficient) (.predecessor 1 88255 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88254 .coefficient)
      LeftAuthority88252.bound (LeftAuthority88252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88252.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88255 .coefficient)
      LeftAuthority88249.bound (LeftAuthority88249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority88252.bound LeftAuthority88249.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88252.bound, LeftAuthority88249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority88252.actual selector witness) * (LeftAuthority88249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88256

namespace LeftBound88260
def owner : Owner := ⟨.program ⟨214⟩, ⟨10482⟩⟩
def transferEvent : Nat := 88260
def frameStart : Nat := 88227
def rule : BoundRule := .identity (.predecessor 0 88259 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88259 .coefficient)
      LeftBound88256.bound (LeftBound88256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88256.derived selector witness)

def rawBound : CoeffClass := LeftBound88256.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound88256.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88260

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
