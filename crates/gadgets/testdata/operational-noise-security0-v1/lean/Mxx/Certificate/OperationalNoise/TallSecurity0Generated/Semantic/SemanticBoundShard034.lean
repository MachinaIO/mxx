import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound7034
def owner : Owner := ⟨.program ⟨214⟩, ⟨10264⟩⟩
def transferEvent : Nat := 7034
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7029 .summary) (.transfer 7033) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7029 .summary)
      LeftBound7027.bound (LeftBound7027.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10263⟩⟩) (rawTerms := some (Proof.Events027.exact7029RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7033)
      LeftBound7033.bound (LeftBound7033.actual selector witness) := by
  exact .transfer (LeftBound7033.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7027.bound LeftBound7033.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7027.bound, LeftBound7033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7027.actual selector witness) * (LeftBound7033.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7034

namespace LeftBound7042
def owner : Owner := ⟨.program ⟨214⟩, ⟨13193⟩⟩
def transferEvent : Nat := 7042
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7040 .coefficient, .predecessor 1 7041 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7040 .coefficient)
      LeftBound7032.bound (LeftBound7032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7041 .coefficient)
      LeftBound6991.bound (LeftBound6991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6991.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7032.bound, LeftBound6991.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7032.bound, LeftBound6991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7032.actual selector witness, LeftBound6991.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7042

namespace LeftBound7044
def owner : Owner := ⟨.program ⟨214⟩, ⟨13193⟩⟩
def transferEvent : Nat := 7044
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 7039 .summary, .result 6996 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7039 .summary)
      LeftBound7034.bound (LeftBound7034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10264⟩⟩) (rawTerms := some (Proof.Events027.exact7039RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6996 .summary)
      LeftBound6993.bound (LeftBound6993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13192⟩⟩) (rawTerms := some (Proof.Events027.exact6996RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7034.bound, LeftBound6993.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7034.bound, LeftBound6993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7034.actual selector witness, LeftBound6993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7044

namespace LeftBound7048
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def transferEvent : Nat := 7048
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7046 .coefficient) (.predecessor 1 7047 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7046 .coefficient)
      LeftBound7042.bound (LeftBound7042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7047 .coefficient)
      LeftAuthority6961.bound (LeftAuthority6961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7042.bound LeftAuthority6961.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7042.bound, LeftAuthority6961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7042.actual selector witness) * (LeftAuthority6961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7048

namespace LeftBound7049
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def transferEvent : Nat := 7049
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩ [⟨.result 6962 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6962 .coefficient)
      LeftAuthority6961.bound (LeftAuthority6961.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25701⟩⟩) (rawTerms := some (Proof.Events027.exact6962RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6961.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6961.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6961.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7049

namespace LeftBound7050
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def transferEvent : Nat := 7050
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7045 .summary) (.transfer 7049) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7045 .summary)
      LeftBound7044.bound (LeftBound7044.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13193⟩⟩) (rawTerms := some (Proof.Events027.exact7045RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7049)
      LeftBound7049.bound (LeftBound7049.actual selector witness) := by
  exact .transfer (LeftBound7049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7044.bound LeftBound7049.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7044.bound, LeftBound7049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7044.actual selector witness) * (LeftBound7049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7050

namespace LeftBound7061
def owner : Owner := ⟨.program ⟨214⟩, ⟨20194⟩⟩
def transferEvent : Nat := 7061
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 7059 .coefficient) (.value (.predecessor 1 7060 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7059 .coefficient)
      LeftAuthority7057.bound (LeftAuthority7057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7060 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7057.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7057.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7057.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7061

namespace LeftBound7065
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def transferEvent : Nat := 7065
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7063 .coefficient) (.predecessor 1 7064 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7063 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7064 .coefficient)
      LeftBound7061.bound (LeftBound7061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7061.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound7061.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound7061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound7061.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7065

namespace LeftBound7066
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def transferEvent : Nat := 7066
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩ [⟨.result 7058 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7058 .coefficient)
      LeftAuthority7057.bound (LeftAuthority7057.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20192⟩⟩) (rawTerms := some (Proof.Events027.exact7058RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7057.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7057.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7057.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7066

namespace LeftBound7067
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def transferEvent : Nat := 7067
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 7066) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7066)
      LeftBound7066.bound (LeftBound7066.actual selector witness) := by
  exact .transfer (LeftBound7066.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound7066.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound7066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound7066.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7067

namespace LeftBound7146
def owner : Owner := ⟨.program ⟨214⟩, ⟨13187⟩⟩
def transferEvent : Nat := 7146
def frameStart : Nat := 7117
def rule : BoundRule := .product (.predecessor 0 7144 .coefficient) (.predecessor 1 7145 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7144 .coefficient)
      LeftAuthority7142.bound (LeftAuthority7142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7142.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7145 .coefficient)
      LeftAuthority7139.bound (LeftAuthority7139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7139.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority7142.bound LeftAuthority7139.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7142.bound, LeftAuthority7139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority7142.actual selector witness) * (LeftAuthority7139.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7146

namespace LeftBound7150
def owner : Owner := ⟨.program ⟨214⟩, ⟨13188⟩⟩
def transferEvent : Nat := 7150
def frameStart : Nat := 7117
def rule : BoundRule := .identity (.predecessor 0 7149 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7149 .coefficient)
      LeftBound7146.bound (LeftBound7146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7146.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7146.derived selector witness)

def rawBound : CoeffClass := LeftBound7146.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound7146.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7150

namespace LeftBound7167
def owner : Owner := ⟨.program ⟨214⟩, ⟨13266⟩⟩
def transferEvent : Nat := 7167
def frameStart : Nat := 7117
def rule : BoundRule := .sum [.predecessor 0 7165 .coefficient, .predecessor 1 7166 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7165 .coefficient)
      LeftBound7150.bound (LeftBound7150.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7150.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7166 .coefficient)
      LeftAuthority7163.bound (LeftAuthority7163.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority7163.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7150.bound, LeftAuthority7163.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7150.bound, LeftAuthority7163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7150.actual selector witness, LeftAuthority7163.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7167

namespace LeftBound7170
def owner : Owner := ⟨.program ⟨214⟩, ⟨13267⟩⟩
def transferEvent : Nat := 7170
def frameStart : Nat := 7117
def rule : BoundRule := .identity (.predecessor 0 7169 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7169 .coefficient)
      LeftBound7167.bound (LeftBound7167.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7167.derived selector witness)

def rawBound : CoeffClass := LeftBound7167.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound7167.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7170

namespace LeftBound7176
def owner : Owner := ⟨.program ⟨214⟩, ⟨13268⟩⟩
def transferEvent : Nat := 7176
def frameStart : Nat := 7117
def rule : BoundRule := .product (.predecessor 0 7174 .coefficient) (.predecessor 1 7175 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7174 .coefficient)
      LeftAuthority7172.bound (LeftAuthority7172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7172.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7175 .coefficient)
      LeftBound7170.bound (LeftBound7170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7170.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority7172.bound LeftBound7170.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7172.bound, LeftBound7170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority7172.actual selector witness) * (LeftBound7170.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7176

namespace LeftBound7192
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 7192
def frameStart : Nat := 7117
def rule : BoundRule := .scale (.predecessor 0 7190 .coefficient) (.value (.predecessor 1 7191 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7190 .coefficient)
      LeftAuthority7188.bound (LeftAuthority7188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7191 .coefficient)
      LeftAuthority7179.bound (LeftAuthority7179.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority7179.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7188.bound LeftAuthority7179.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7188.bound, LeftAuthority7179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7188.actual selector witness) * (LeftAuthority7179.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7192

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
