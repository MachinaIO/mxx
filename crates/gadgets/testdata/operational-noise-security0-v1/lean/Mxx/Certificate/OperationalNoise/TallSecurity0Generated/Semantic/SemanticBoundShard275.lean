import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard274

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40960
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def transferEvent : Nat := 40960
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40958 .coefficient) (.predecessor 1 40959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40958 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40959 .coefficient)
      LeftBound40956.bound (LeftBound40956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40956.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound40956.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound40956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound40956.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40960

namespace LeftBound40961
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def transferEvent : Nat := 40961
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19536⟩⟩]⟩ [⟨.result 40953 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40953 .coefficient)
      LeftAuthority40952.bound (LeftAuthority40952.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19536⟩⟩) (rawTerms := some (Proof.Events159.exact40953RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40952.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40952.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40952.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40961

namespace LeftBound40962
def owner : Owner := ⟨.program ⟨214⟩, ⟨19539⟩⟩
def transferEvent : Nat := 40962
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 40961) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40961)
      LeftBound40961.bound (LeftBound40961.actual selector witness) := by
  exact .transfer (LeftBound40961.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound40961.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound40961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound40961.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40962

namespace LeftBound41041
def owner : Owner := ⟨.program ⟨214⟩, ⟨14226⟩⟩
def transferEvent : Nat := 41041
def frameStart : Nat := 41012
def rule : BoundRule := .product (.predecessor 0 41039 .coefficient) (.predecessor 1 41040 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41039 .coefficient)
      LeftAuthority41037.bound (LeftAuthority41037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41040 .coefficient)
      LeftAuthority41034.bound (LeftAuthority41034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41034.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41034.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41037.bound LeftAuthority41034.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41037.bound, LeftAuthority41034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41037.actual selector witness) * (LeftAuthority41034.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41041

namespace LeftBound41045
def owner : Owner := ⟨.program ⟨214⟩, ⟨14227⟩⟩
def transferEvent : Nat := 41045
def frameStart : Nat := 41012
def rule : BoundRule := .identity (.predecessor 0 41044 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41044 .coefficient)
      LeftBound41041.bound (LeftBound41041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41041.derived selector witness)

def rawBound : CoeffClass := LeftBound41041.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41041.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41045

namespace LeftBound41062
def owner : Owner := ⟨.program ⟨214⟩, ⟨14322⟩⟩
def transferEvent : Nat := 41062
def frameStart : Nat := 41012
def rule : BoundRule := .sum [.predecessor 0 41060 .coefficient, .predecessor 1 41061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41060 .coefficient)
      LeftBound41045.bound (LeftBound41045.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41061 .coefficient)
      LeftAuthority41058.bound (LeftAuthority41058.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41045.bound, LeftAuthority41058.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41045.bound, LeftAuthority41058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41045.actual selector witness, LeftAuthority41058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41062

namespace LeftBound41065
def owner : Owner := ⟨.program ⟨214⟩, ⟨14323⟩⟩
def transferEvent : Nat := 41065
def frameStart : Nat := 41012
def rule : BoundRule := .identity (.predecessor 0 41064 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41064 .coefficient)
      LeftBound41062.bound (LeftBound41062.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41062.derived selector witness)

def rawBound : CoeffClass := LeftBound41062.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41062.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41065

namespace LeftBound41071
def owner : Owner := ⟨.program ⟨214⟩, ⟨14324⟩⟩
def transferEvent : Nat := 41071
def frameStart : Nat := 41012
def rule : BoundRule := .product (.predecessor 0 41069 .coefficient) (.predecessor 1 41070 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41069 .coefficient)
      LeftAuthority41067.bound (LeftAuthority41067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41070 .coefficient)
      LeftBound41065.bound (LeftBound41065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority41067.bound LeftBound41065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41067.bound, LeftBound41065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority41067.actual selector witness) * (LeftBound41065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41071

namespace LeftBound41087
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 41087
def frameStart : Nat := 41012
def rule : BoundRule := .scale (.predecessor 0 41085 .coefficient) (.value (.predecessor 1 41086 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41085 .coefficient)
      LeftAuthority41083.bound (LeftAuthority41083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41086 .coefficient)
      LeftAuthority41074.bound (LeftAuthority41074.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41074.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41083.bound LeftAuthority41074.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41083.bound, LeftAuthority41074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41083.actual selector witness) * (LeftAuthority41074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41087

namespace LeftBound41090
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 41090
def frameStart : Nat := 41012
def rule : BoundRule := .identity (.predecessor 0 41089 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41089 .coefficient)
      LeftAuthority41077.bound (LeftAuthority41077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41077.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41077.derived selector witness)

def rawBound : CoeffClass := LeftAuthority41077.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority41077.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41090

namespace LeftBound41094
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 41094
def frameStart : Nat := 41012
def rule : BoundRule := .product (.predecessor 0 41092 .coefficient) (.predecessor 1 41093 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41092 .coefficient)
      LeftBound41090.bound (LeftBound41090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41093 .coefficient)
      LeftBound41087.bound (LeftBound41087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41090.bound LeftBound41087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41090.bound, LeftBound41087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41090.actual selector witness) * (LeftBound41087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41094

namespace LeftBound41099
def owner : Owner := ⟨.program ⟨214⟩, ⟨14325⟩⟩
def transferEvent : Nat := 41099
def frameStart : Nat := 41012
def rule : BoundRule := .sum [.predecessor 0 41097 .coefficient, .predecessor 1 41098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41097 .coefficient)
      LeftBound41094.bound (LeftBound41094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41098 .coefficient)
      LeftBound41071.bound (LeftBound41071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41094.bound, LeftBound41071.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41094.bound, LeftBound41071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41094.actual selector witness, LeftBound41071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41099

namespace LeftBound41103
def owner : Owner := ⟨.program ⟨214⟩, ⟨26079⟩⟩
def transferEvent : Nat := 41103
def frameStart : Nat := 41012
def rule : BoundRule := .product (.predecessor 0 41101 .coefficient) (.predecessor 1 41102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41101 .coefficient)
      LeftBound41099.bound (LeftBound41099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41102 .coefficient)
      LeftAuthority41056.bound (LeftAuthority41056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41099.bound LeftAuthority41056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41099.bound, LeftAuthority41056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41099.actual selector witness) * (LeftAuthority41056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41103

namespace LeftBound41114
def owner : Owner := ⟨.program ⟨214⟩, ⟨15950⟩⟩
def transferEvent : Nat := 41114
def frameStart : Nat := 41012
def rule : BoundRule := .product (.predecessor 0 41112 .coefficient) (.predecessor 1 41113 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41112 .coefficient)
      LeftAuthority41067.bound (LeftAuthority41067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41113 .coefficient)
      LeftAuthority41110.bound (LeftAuthority41110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41067.bound LeftAuthority41110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41067.bound, LeftAuthority41110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41067.actual selector witness) * (LeftAuthority41110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41114

namespace LeftBound41122
def owner : Owner := ⟨.program ⟨214⟩, ⟨15951⟩⟩
def transferEvent : Nat := 41122
def frameStart : Nat := 41012
def rule : BoundRule := .sum [.predecessor 0 41120 .coefficient, .predecessor 1 41121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41120 .coefficient)
      LeftAuthority41118.bound (LeftAuthority41118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41121 .coefficient)
      LeftBound41114.bound (LeftBound41114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority41118.bound, LeftBound41114.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41118.bound, LeftBound41114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority41118.actual selector witness, LeftBound41114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41122

namespace LeftBound41126
def owner : Owner := ⟨.program ⟨214⟩, ⟨26080⟩⟩
def transferEvent : Nat := 41126
def frameStart : Nat := 41012
def rule : BoundRule := .sum [.predecessor 0 41124 .coefficient, .predecessor 1 41125 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41124 .coefficient)
      LeftBound41122.bound (LeftBound41122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41125 .coefficient)
      LeftBound41103.bound (LeftBound41103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41122.bound, LeftBound41103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41122.bound, LeftBound41103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41122.actual selector witness, LeftBound41103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41126

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
