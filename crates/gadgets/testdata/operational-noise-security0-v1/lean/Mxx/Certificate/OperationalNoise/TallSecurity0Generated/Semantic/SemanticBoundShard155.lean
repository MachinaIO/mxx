import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard154

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23894
def owner : Owner := ⟨.program ⟨214⟩, ⟨9839⟩⟩
def transferEvent : Nat := 23894
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23889 .summary) (.transfer 23893) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23889 .summary)
      LeftBound23887.bound (LeftBound23887.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9838⟩⟩) (rawTerms := some (Proof.Events093.exact23889RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23893)
      LeftBound23893.bound (LeftBound23893.actual selector witness) := by
  exact .transfer (LeftBound23893.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23887.bound LeftBound23893.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23887.bound, LeftBound23893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23887.actual selector witness) * (LeftBound23893.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23894

namespace LeftBound23902
def owner : Owner := ⟨.program ⟨214⟩, ⟨12401⟩⟩
def transferEvent : Nat := 23902
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23900 .coefficient, .predecessor 1 23901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23900 .coefficient)
      LeftBound23892.bound (LeftBound23892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23901 .coefficient)
      LeftBound23864.bound (LeftBound23864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23892.bound, LeftBound23864.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23892.bound, LeftBound23864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23892.actual selector witness, LeftBound23864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23902

namespace LeftBound23904
def owner : Owner := ⟨.program ⟨214⟩, ⟨12401⟩⟩
def transferEvent : Nat := 23904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 23899 .summary, .result 23869 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23899 .summary)
      LeftBound23894.bound (LeftBound23894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9839⟩⟩) (rawTerms := some (Proof.Events093.exact23899RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23869 .summary)
      LeftBound23866.bound (LeftBound23866.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12400⟩⟩) (rawTerms := some (Proof.Events093.exact23869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23894.bound, LeftBound23866.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23894.bound, LeftBound23866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23894.actual selector witness, LeftBound23866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23904

namespace LeftBound23908
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def transferEvent : Nat := 23908
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23906 .coefficient) (.predecessor 1 23907 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23906 .coefficient)
      LeftBound23902.bound (LeftBound23902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23907 .coefficient)
      LeftAuthority23840.bound (LeftAuthority23840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23840.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23840.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23902.bound LeftAuthority23840.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23902.bound, LeftAuthority23840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23902.actual selector witness) * (LeftAuthority23840.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23908

namespace LeftBound23909
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def transferEvent : Nat := 23909
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩ [⟨.result 23841 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23841 .coefficient)
      LeftAuthority23840.bound (LeftAuthority23840.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25388⟩⟩) (rawTerms := some (Proof.Events093.exact23841RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23840.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23840.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23840.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23840.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23909

namespace LeftBound23910
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def transferEvent : Nat := 23910
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23905 .summary) (.transfer 23909) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23905 .summary)
      LeftBound23904.bound (LeftBound23904.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12401⟩⟩) (rawTerms := some (Proof.Events093.exact23905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23909)
      LeftBound23909.bound (LeftBound23909.actual selector witness) := by
  exact .transfer (LeftBound23909.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23904.bound LeftBound23909.bound
def bound : CoeffClass := .finite ⟨350316591579136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23904.bound, LeftBound23909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23904.actual selector witness) * (LeftBound23909.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23910

namespace LeftBound23921
def owner : Owner := ⟨.program ⟨214⟩, ⟨19902⟩⟩
def transferEvent : Nat := 23921
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 23919 .coefficient) (.value (.predecessor 1 23920 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23919 .coefficient)
      LeftAuthority23917.bound (LeftAuthority23917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23920 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23917.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23917.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23917.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23921

namespace LeftBound23925
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def transferEvent : Nat := 23925
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23923 .coefficient) (.predecessor 1 23924 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23923 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23924 .coefficient)
      LeftBound23921.bound (LeftBound23921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23921.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound23921.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound23921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound23921.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23925

namespace LeftBound23926
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def transferEvent : Nat := 23926
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19900⟩⟩]⟩ [⟨.result 23918 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23918 .coefficient)
      LeftAuthority23917.bound (LeftAuthority23917.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19900⟩⟩) (rawTerms := some (Proof.Events093.exact23918RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23917.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23917.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23917.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23926

namespace LeftBound23927
def owner : Owner := ⟨.program ⟨214⟩, ⟨19903⟩⟩
def transferEvent : Nat := 23927
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 23926) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23926)
      LeftBound23926.bound (LeftBound23926.actual selector witness) := by
  exact .transfer (LeftBound23926.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound23926.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound23926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound23926.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23927

namespace LeftBound24006
def owner : Owner := ⟨.program ⟨214⟩, ⟨12395⟩⟩
def transferEvent : Nat := 24006
def frameStart : Nat := 23977
def rule : BoundRule := .product (.predecessor 0 24004 .coefficient) (.predecessor 1 24005 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24004 .coefficient)
      LeftAuthority24002.bound (LeftAuthority24002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24005 .coefficient)
      LeftAuthority23999.bound (LeftAuthority23999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23999.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24002.bound LeftAuthority23999.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24002.bound, LeftAuthority23999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24002.actual selector witness) * (LeftAuthority23999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24006

namespace LeftBound24010
def owner : Owner := ⟨.program ⟨214⟩, ⟨12396⟩⟩
def transferEvent : Nat := 24010
def frameStart : Nat := 23977
def rule : BoundRule := .identity (.predecessor 0 24009 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24009 .coefficient)
      LeftBound24006.bound (LeftBound24006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24006.derived selector witness)

def rawBound : CoeffClass := LeftBound24006.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24006.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24010

namespace LeftBound24027
def owner : Owner := ⟨.program ⟨214⟩, ⟨12478⟩⟩
def transferEvent : Nat := 24027
def frameStart : Nat := 23977
def rule : BoundRule := .sum [.predecessor 0 24025 .coefficient, .predecessor 1 24026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24025 .coefficient)
      LeftBound24010.bound (LeftBound24010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24026 .coefficient)
      LeftAuthority24023.bound (LeftAuthority24023.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24010.bound, LeftAuthority24023.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24010.bound, LeftAuthority24023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24010.actual selector witness, LeftAuthority24023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24027

namespace LeftBound24030
def owner : Owner := ⟨.program ⟨214⟩, ⟨12479⟩⟩
def transferEvent : Nat := 24030
def frameStart : Nat := 23977
def rule : BoundRule := .identity (.predecessor 0 24029 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24029 .coefficient)
      LeftBound24027.bound (LeftBound24027.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24027.derived selector witness)

def rawBound : CoeffClass := LeftBound24027.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24027.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24030

namespace LeftBound24036
def owner : Owner := ⟨.program ⟨214⟩, ⟨12480⟩⟩
def transferEvent : Nat := 24036
def frameStart : Nat := 23977
def rule : BoundRule := .product (.predecessor 0 24034 .coefficient) (.predecessor 1 24035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24034 .coefficient)
      LeftAuthority24032.bound (LeftAuthority24032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24035 .coefficient)
      LeftBound24030.bound (LeftBound24030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24030.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority24032.bound LeftBound24030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24032.bound, LeftBound24030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority24032.actual selector witness) * (LeftBound24030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24036

namespace LeftBound24052
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 24052
def frameStart : Nat := 23977
def rule : BoundRule := .scale (.predecessor 0 24050 .coefficient) (.value (.predecessor 1 24051 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24050 .coefficient)
      LeftAuthority24048.bound (LeftAuthority24048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24051 .coefficient)
      LeftAuthority24039.bound (LeftAuthority24039.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24039.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24048.bound LeftAuthority24039.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24048.bound, LeftAuthority24039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24048.actual selector witness) * (LeftAuthority24039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24052

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
