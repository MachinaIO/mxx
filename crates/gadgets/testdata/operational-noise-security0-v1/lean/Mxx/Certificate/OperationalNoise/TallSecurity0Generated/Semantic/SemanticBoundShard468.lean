import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard467

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68943
def owner : Owner := ⟨.program ⟨214⟩, ⟨25139⟩⟩
def transferEvent : Nat := 68943
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68941 .coefficient, .predecessor 1 68942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68941 .coefficient)
      LeftBound68764.bound (LeftBound68764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68942 .coefficient)
      LeftBound68747.bound (LeftBound68747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68764.bound, LeftBound68747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68764.bound, LeftBound68747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68764.actual selector witness, LeftBound68747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68943

namespace LeftBound68946
def owner : Owner := ⟨.program ⟨214⟩, ⟨25139⟩⟩
def transferEvent : Nat := 68946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68940 .summary, .result 68754 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68940 .summary)
      LeftBound68766.bound (LeftBound68766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19743⟩⟩) (rawTerms := some (Proof.Events269.exact68940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68754 .summary)
      LeftBound68749.bound (LeftBound68749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25138⟩⟩) (rawTerms := some (Proof.Events268.exact68754RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68749.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68766.bound, LeftBound68749.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68766.bound, LeftBound68749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68766.actual selector witness, LeftBound68749.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68946

namespace LeftBound68950
def owner : Owner := ⟨.program ⟨214⟩, ⟨28506⟩⟩
def transferEvent : Nat := 68950
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68948 .coefficient) (.predecessor 1 68949 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68948 .coefficient)
      LeftBound68943.bound (LeftBound68943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68949 .coefficient)
      LeftAuthority68669.bound (LeftAuthority68669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68943.bound LeftAuthority68669.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68943.bound, LeftAuthority68669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68943.actual selector witness) * (LeftAuthority68669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68950

namespace LeftBound68951
def owner : Owner := ⟨.program ⟨214⟩, ⟨28506⟩⟩
def transferEvent : Nat := 68951
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28504⟩⟩]⟩ [⟨.result 68670 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68670 .coefficient)
      LeftAuthority68669.bound (LeftAuthority68669.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28504⟩⟩) (rawTerms := some (Proof.Events268.exact68670RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68669.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68669.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68669.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68951

namespace LeftBound68952
def owner : Owner := ⟨.program ⟨214⟩, ⟨28506⟩⟩
def transferEvent : Nat := 68952
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68947 .summary) (.transfer 68951) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68947 .summary)
      LeftBound68946.bound (LeftBound68946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25139⟩⟩) (rawTerms := some (Proof.Events269.exact68947RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68951)
      LeftBound68951.bound (LeftBound68951.actual selector witness) := by
  exact .transfer (LeftBound68951.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68946.bound LeftBound68951.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68946.bound, LeftBound68951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68946.actual selector witness) * (LeftBound68951.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68952

namespace LeftBound68963
def owner : Owner := ⟨.program ⟨214⟩, ⟨21830⟩⟩
def transferEvent : Nat := 68963
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 68961 .coefficient) (.value (.predecessor 1 68962 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68961 .coefficient)
      LeftAuthority68959.bound (LeftAuthority68959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68962 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68959.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68959.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68959.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68963

namespace LeftBound68967
def owner : Owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩
def transferEvent : Nat := 68967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68965 .coefficient) (.predecessor 1 68966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68965 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68966 .coefficient)
      LeftBound68963.bound (LeftBound68963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound68963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound68963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound68963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68967

namespace LeftBound68968
def owner : Owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩
def transferEvent : Nat := 68968
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩ [⟨.result 68960 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68960 .coefficient)
      LeftAuthority68959.bound (LeftAuthority68959.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21828⟩⟩) (rawTerms := some (Proof.Events269.exact68960RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68959.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68959.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68959.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68959.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68968

namespace LeftBound68969
def owner : Owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩
def transferEvent : Nat := 68969
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 68968) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68968)
      LeftBound68968.bound (LeftBound68968.actual selector witness) := by
  exact .transfer (LeftBound68968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound68968.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound68968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound68968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68969

namespace LeftBound69064
def owner : Owner := ⟨.program ⟨214⟩, ⟨16259⟩⟩
def transferEvent : Nat := 69064
def frameStart : Nat := 69025
def rule : BoundRule := .identity (.predecessor 0 69063 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69063 .coefficient)
      LeftAuthority69061.bound (LeftAuthority69061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69061.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69061.derived selector witness)

def rawBound : CoeffClass := LeftAuthority69061.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority69061.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69064

namespace LeftBound69081
def owner : Owner := ⟨.program ⟨214⟩, ⟨16333⟩⟩
def transferEvent : Nat := 69081
def frameStart : Nat := 69025
def rule : BoundRule := .sum [.predecessor 0 69079 .coefficient, .predecessor 1 69080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69079 .coefficient)
      LeftBound69064.bound (LeftBound69064.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69080 .coefficient)
      LeftAuthority69077.bound (LeftAuthority69077.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority69077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69064.bound, LeftAuthority69077.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69064.bound, LeftAuthority69077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69064.actual selector witness, LeftAuthority69077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69081

namespace LeftBound69084
def owner : Owner := ⟨.program ⟨214⟩, ⟨16334⟩⟩
def transferEvent : Nat := 69084
def frameStart : Nat := 69025
def rule : BoundRule := .identity (.predecessor 0 69083 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69083 .coefficient)
      LeftBound69081.bound (LeftBound69081.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69081.derived selector witness)

def rawBound : CoeffClass := LeftBound69081.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound69081.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69084

namespace LeftBound69090
def owner : Owner := ⟨.program ⟨214⟩, ⟨16335⟩⟩
def transferEvent : Nat := 69090
def frameStart : Nat := 69025
def rule : BoundRule := .product (.predecessor 0 69088 .coefficient) (.predecessor 1 69089 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69088 .coefficient)
      LeftAuthority69086.bound (LeftAuthority69086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69086.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69089 .coefficient)
      LeftBound69084.bound (LeftBound69084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69084.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority69086.bound LeftBound69084.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69086.bound, LeftBound69084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority69086.actual selector witness) * (LeftBound69084.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69090

namespace LeftBound69098
def owner : Owner := ⟨.program ⟨214⟩, ⟨16336⟩⟩
def transferEvent : Nat := 69098
def frameStart : Nat := 69025
def rule : BoundRule := .sum [.predecessor 0 69096 .coefficient, .predecessor 1 69097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69096 .coefficient)
      LeftAuthority69094.bound (LeftAuthority69094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69094.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69097 .coefficient)
      LeftBound69090.bound (LeftBound69090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69090.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69094.bound, LeftBound69090.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69094.bound, LeftBound69090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69094.actual selector witness, LeftBound69090.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69098

namespace LeftBound69102
def owner : Owner := ⟨.program ⟨214⟩, ⟨28505⟩⟩
def transferEvent : Nat := 69102
def frameStart : Nat := 69025
def rule : BoundRule := .product (.predecessor 0 69100 .coefficient) (.predecessor 1 69101 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69100 .coefficient)
      LeftBound69098.bound (LeftBound69098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69098.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69101 .coefficient)
      LeftAuthority69075.bound (LeftAuthority69075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69098.bound LeftAuthority69075.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69098.bound, LeftAuthority69075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69098.actual selector witness) * (LeftAuthority69075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69102

namespace LeftBound69113
def owner : Owner := ⟨.program ⟨214⟩, ⟨16306⟩⟩
def transferEvent : Nat := 69113
def frameStart : Nat := 69025
def rule : BoundRule := .product (.predecessor 0 69111 .coefficient) (.predecessor 1 69112 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69111 .coefficient)
      LeftAuthority69086.bound (LeftAuthority69086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69086.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69112 .coefficient)
      LeftAuthority69109.bound (LeftAuthority69109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact69110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69109.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority69086.bound LeftAuthority69109.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69086.bound, LeftAuthority69109.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority69086.actual selector witness) * (LeftAuthority69109.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69113

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
