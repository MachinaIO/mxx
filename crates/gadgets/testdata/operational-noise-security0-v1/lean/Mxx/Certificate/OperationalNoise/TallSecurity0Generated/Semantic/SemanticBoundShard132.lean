import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard017
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard131

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21050
def owner : Owner := ⟨.program ⟨214⟩, ⟨7899⟩⟩
def transferEvent : Nat := 21050
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21048 .coefficient) (.predecessor 1 21049 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21048 .coefficient)
      LeftBound21044.bound (LeftBound21044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21049 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21044.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21044.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21044.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21050

namespace LeftBound21051
def owner : Owner := ⟨.program ⟨214⟩, ⟨7899⟩⟩
def transferEvent : Nat := 21051
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21051

namespace LeftBound21052
def owner : Owner := ⟨.program ⟨214⟩, ⟨7899⟩⟩
def transferEvent : Nat := 21052
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21047 .summary) (.transfer 21051) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21047 .summary)
      LeftBound21045.bound (LeftBound21045.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7720⟩⟩) (rawTerms := some (Proof.Events082.exact21047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21051)
      LeftBound21051.bound (LeftBound21051.actual selector witness) := by
  exact .transfer (LeftBound21051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21045.bound LeftBound21051.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21045.bound, LeftBound21051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21045.actual selector witness) * (LeftBound21051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21052

namespace LeftBound21078
def owner : Owner := ⟨.program ⟨214⟩, ⟨30215⟩⟩
def transferEvent : Nat := 21078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21076 .coefficient, .predecessor 1 21077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21076 .coefficient)
      LeftBound21050.bound (LeftBound21050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21077 .coefficient)
      LeftBound21028.bound (LeftBound21028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21028.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21050.bound, LeftBound21028.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21050.bound, LeftBound21028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21050.actual selector witness, LeftBound21028.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21078

namespace LeftBound21098
def owner : Owner := ⟨.program ⟨214⟩, ⟨30215⟩⟩
def transferEvent : Nat := 21098
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21075 .summary, .result 21030 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21075 .summary)
      LeftBound21052.bound (LeftBound21052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7899⟩⟩) (rawTerms := some (Proof.Events082.exact21075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21030 .summary)
      LeftBound21029.bound (LeftBound21029.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30214⟩⟩) (rawTerms := some (Proof.Events082.exact21030RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21052.bound, LeftBound21029.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21052.bound, LeftBound21029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21052.actual selector witness, LeftBound21029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21098

namespace LeftBound21102
def owner : Owner := ⟨.program ⟨214⟩, ⟨30216⟩⟩
def transferEvent : Nat := 21102
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21100 .coefficient) (.predecessor 1 21101 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21100 .coefficient)
      LeftBound21078.bound (LeftBound21078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21101 .coefficient)
      LeftBound5486.bound (LeftBound5486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21078.bound LeftBound5486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21078.bound, LeftBound5486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21078.actual selector witness) * (LeftBound5486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21102

namespace LeftBound21103
def owner : Owner := ⟨.program ⟨214⟩, ⟨30216⟩⟩
def transferEvent : Nat := 21103
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7819⟩⟩]⟩ [⟨.result 5483 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5483 .coefficient)
      LeftAuthority5482.bound (LeftAuthority5482.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7819⟩⟩) (rawTerms := some (Proof.Events021.exact5483RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5482.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5482.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5482.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21103

namespace LeftBound21104
def owner : Owner := ⟨.program ⟨214⟩, ⟨30216⟩⟩
def transferEvent : Nat := 21104
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21099 .summary) (.transfer 21103) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21099 .summary)
      LeftBound21098.bound (LeftBound21098.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30215⟩⟩) (rawTerms := some (Proof.Events082.exact21099RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21103)
      LeftBound21103.bound (LeftBound21103.actual selector witness) := by
  exact .transfer (LeftBound21103.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21098.bound LeftBound21103.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21098.bound, LeftBound21103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21098.actual selector witness) * (LeftBound21103.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21104

namespace LeftBound21166
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def transferEvent : Nat := 21166
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21164 .coefficient, .predecessor 1 21165 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21164 .coefficient)
      LeftBound21102.bound (LeftBound21102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21165 .coefficient)
      LeftBound6332.bound (LeftBound6332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6332.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21102.bound, LeftBound6332.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21102.bound, LeftBound6332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21102.actual selector witness, LeftBound6332.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21166

namespace LeftBound21186
def owner : Owner := ⟨.program ⟨214⟩, ⟨30217⟩⟩
def transferEvent : Nat := 21186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21163 .summary, .result 6409 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21163 .summary)
      LeftBound21104.bound (LeftBound21104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30216⟩⟩) (rawTerms := some (Proof.Events082.exact21163RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6409 .summary)
      LeftBound6370.bound (LeftBound6370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18906⟩⟩) (rawTerms := some (Proof.Events025.exact6409RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21104.bound, LeftBound6370.bound]
def bound : CoeffClass := .finite ⟨1149729608724524008718218297164355856419136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21104.bound, LeftBound6370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21104.actual selector witness, LeftBound6370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21186

namespace LeftBound21190
def owner : Owner := ⟨.program ⟨214⟩, ⟨30218⟩⟩
def transferEvent : Nat := 21190
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21188 .coefficient) (.predecessor 1 21189 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21188 .coefficient)
      LeftBound21166.bound (LeftBound21166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21189 .coefficient)
      LeftBound5475.bound (LeftBound5475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21166.bound LeftBound5475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21166.bound, LeftBound5475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21166.actual selector witness) * (LeftBound5475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21190

namespace LeftBound21191
def owner : Owner := ⟨.program ⟨214⟩, ⟨30218⟩⟩
def transferEvent : Nat := 21191
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩ [⟨.result 5472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5472 .coefficient)
      LeftAuthority5471.bound (LeftAuthority5471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6645⟩⟩) (rawTerms := some (Proof.Events021.exact5472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5471.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21191

namespace LeftBound21192
def owner : Owner := ⟨.program ⟨214⟩, ⟨30218⟩⟩
def transferEvent : Nat := 21192
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21187 .summary) (.transfer 21191) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21187 .summary)
      LeftBound21186.bound (LeftBound21186.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30217⟩⟩) (rawTerms := some (Proof.Events082.exact21187RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21191)
      LeftBound21191.bound (LeftBound21191.actual selector witness) := by
  exact .transfer (LeftBound21191.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21186.bound LeftBound21191.bound
def bound : CoeffClass := .finite ⟨4219526059692742704380000642085940622751931826176, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21186.bound, LeftBound21191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21186.actual selector witness) * (LeftBound21191.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21192

namespace LeftBound21273
def owner : Owner := ⟨.program ⟨214⟩, ⟨5616⟩⟩
def transferEvent : Nat := 21273
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 21268 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21268 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21273

namespace LeftBound21277
def owner : Owner := ⟨.program ⟨214⟩, ⟨6582⟩⟩
def transferEvent : Nat := 21277
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21275 .coefficient) (.predecessor 1 21276 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21275 .coefficient)
      LeftBound21273.bound (LeftBound21273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21276 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21273.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21273.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21273.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21277

namespace LeftBound21289
def owner : Owner := ⟨.program ⟨214⟩, ⟨5557⟩⟩
def transferEvent : Nat := 21289
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 21284 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21284 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21289

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
