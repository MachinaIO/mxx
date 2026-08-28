import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard671
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard720

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105244
def owner : Owner := ⟨.program ⟨214⟩, ⟨17598⟩⟩
def transferEvent : Nat := 105244
def frameStart : Nat := 105168
def rule : BoundRule := .product (.predecessor 0 105242 .coefficient) (.predecessor 1 105243 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105242 .coefficient)
      LeftAuthority105217.bound (LeftAuthority105217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105243 .coefficient)
      LeftAuthority105240.bound (LeftAuthority105240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105217.bound LeftAuthority105240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105217.bound, LeftAuthority105240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority105217.actual selector witness) * (LeftAuthority105240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105244

namespace LeftBound105252
def owner : Owner := ⟨.program ⟨214⟩, ⟨17599⟩⟩
def transferEvent : Nat := 105252
def frameStart : Nat := 105168
def rule : BoundRule := .sum [.predecessor 0 105250 .coefficient, .predecessor 1 105251 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105250 .coefficient)
      LeftAuthority105248.bound (LeftAuthority105248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105251 .coefficient)
      LeftBound105244.bound (LeftBound105244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105244.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105248.bound, LeftBound105244.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105248.bound, LeftBound105244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105248.actual selector witness, LeftBound105244.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105252

namespace LeftBound105256
def owner : Owner := ⟨.program ⟨214⟩, ⟨28481⟩⟩
def transferEvent : Nat := 105256
def frameStart : Nat := 105168
def rule : BoundRule := .sum [.predecessor 0 105254 .coefficient, .predecessor 1 105255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105254 .coefficient)
      LeftBound105252.bound (LeftBound105252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105255 .coefficient)
      LeftBound105233.bound (LeftBound105233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105252.bound, LeftBound105233.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105252.bound, LeftBound105233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105252.actual selector witness, LeftBound105233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105256

namespace LeftBound105269
def owner : Owner := ⟨.program ⟨214⟩, ⟨28478⟩⟩
def transferEvent : Nat := 105269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105267 .coefficient, .predecessor 1 105268 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105267 .coefficient)
      LeftBound105122.bound (LeftBound105122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105268 .coefficient)
      LeftBound105105.bound (LeftBound105105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105122.bound, LeftBound105105.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105122.bound, LeftBound105105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105122.actual selector witness, LeftBound105105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105269

namespace LeftBound105272
def owner : Owner := ⟨.program ⟨214⟩, ⟨28478⟩⟩
def transferEvent : Nat := 105272
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105266 .summary, .result 105112 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105266 .summary)
      LeftBound105124.bound (LeftBound105124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21752⟩⟩) (rawTerms := some (Proof.Events411.exact105266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105112 .summary)
      LeftBound105107.bound (LeftBound105107.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28477⟩⟩) (rawTerms := some (Proof.Events410.exact105112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105124.bound, LeftBound105107.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105124.bound, LeftBound105107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105124.actual selector witness, LeftBound105107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105272

namespace LeftBound105276
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def transferEvent : Nat := 105276
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105274 .coefficient) (.predecessor 1 105275 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105274 .coefficient)
      LeftBound105269.bound (LeftBound105269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105275 .coefficient)
      LeftBound5658.bound (LeftBound5658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105269.bound LeftBound5658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105269.bound, LeftBound5658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105269.actual selector witness) * (LeftBound5658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105276

namespace LeftBound105277
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def transferEvent : Nat := 105277
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩ [⟨.result 5655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5655 .coefficient)
      LeftAuthority5654.bound (LeftAuthority5654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6677⟩⟩) (rawTerms := some (Proof.Events022.exact5655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105277

namespace LeftBound105278
def owner : Owner := ⟨.program ⟨214⟩, ⟨28479⟩⟩
def transferEvent : Nat := 105278
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105273 .summary) (.transfer 105277) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105273 .summary)
      LeftBound105272.bound (LeftBound105272.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28478⟩⟩) (rawTerms := some (Proof.Events411.exact105273RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105277)
      LeftBound105277.bound (LeftBound105277.actual selector witness) := by
  exact .transfer (LeftBound105277.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105272.bound LeftBound105277.bound
def bound : CoeffClass := .finite ⟨4742405496644812892115304448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105272.bound, LeftBound105277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105272.actual selector witness) * (LeftBound105277.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105278

namespace LeftBound105293
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def transferEvent : Nat := 105293
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105291 .coefficient) (.predecessor 1 105292 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105291 .coefficient)
      LeftBound98092.bound (LeftBound98092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105292 .coefficient)
      LeftAuthority105289.bound (LeftAuthority105289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98092.bound LeftAuthority105289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98092.bound, LeftAuthority105289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98092.actual selector witness) * (LeftAuthority105289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105293

namespace LeftBound105294
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def transferEvent : Nat := 105294
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28258⟩⟩]⟩ [⟨.result 105290 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105290 .coefficient)
      LeftAuthority105289.bound (LeftAuthority105289.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28258⟩⟩) (rawTerms := some (Proof.Events411.exact105290RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105289.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105289.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105289.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105294

namespace LeftBound105295
def owner : Owner := ⟨.program ⟨214⟩, ⟨28260⟩⟩
def transferEvent : Nat := 105295
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98096 .summary) (.transfer 105294) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98096 .summary)
      LeftBound98095.bound (LeftBound98095.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26209⟩⟩) (rawTerms := some (Proof.Events383.exact98096RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105294)
      LeftBound105294.bound (LeftBound105294.actual selector witness) := by
  exact .transfer (LeftBound105294.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98095.bound LeftBound105294.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98095.bound, LeftBound105294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98095.actual selector witness) * (LeftBound105294.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105295

namespace LeftBound105306
def owner : Owner := ⟨.program ⟨214⟩, ⟨21607⟩⟩
def transferEvent : Nat := 105306
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 105304 .coefficient) (.value (.predecessor 1 105305 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105304 .coefficient)
      LeftAuthority105302.bound (LeftAuthority105302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105305 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105302.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105302.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105302.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105306

namespace LeftBound105310
def owner : Owner := ⟨.program ⟨214⟩, ⟨21608⟩⟩
def transferEvent : Nat := 105310
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105308 .coefficient) (.predecessor 1 105309 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105308 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105309 .coefficient)
      LeftBound105306.bound (LeftBound105306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound105306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound105306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound105306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105310

namespace LeftBound105311
def owner : Owner := ⟨.program ⟨214⟩, ⟨21608⟩⟩
def transferEvent : Nat := 105311
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21605⟩⟩]⟩ [⟨.result 105303 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105303 .coefficient)
      LeftAuthority105302.bound (LeftAuthority105302.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21605⟩⟩) (rawTerms := some (Proof.Events411.exact105303RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105302.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105302.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105302.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105311

namespace LeftBound105312
def owner : Owner := ⟨.program ⟨214⟩, ⟨21608⟩⟩
def transferEvent : Nat := 105312
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 105311) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105311)
      LeftBound105311.bound (LeftBound105311.actual selector witness) := by
  exact .transfer (LeftBound105311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound105311.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound105311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound105311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105312

namespace LeftBound105383
def owner : Owner := ⟨.program ⟨214⟩, ⟨16169⟩⟩
def transferEvent : Nat := 105383
def frameStart : Nat := 105356
def rule : BoundRule := .identity (.predecessor 0 105382 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105382 .coefficient)
      LeftAuthority105380.bound (LeftAuthority105380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105380.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105380.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority105380.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105383

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
