import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard639
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard710
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard712
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard713
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard714
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard734
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard735
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard736

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107269
def owner : Owner := ⟨.program ⟨214⟩, ⟨29565⟩⟩
def transferEvent : Nat := 107269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107265 .summary, .result 104343 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107265 .summary)
      LeftBound107264.bound (LeftBound107264.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29348⟩⟩) (rawTerms := some (Proof.Events419.exact107265RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104343 .summary)
      LeftBound104338.bound (LeftBound104338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29564⟩⟩) (rawTerms := some (Proof.Events407.exact104343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107264.bound, LeftBound104338.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107264.bound, LeftBound104338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107264.actual selector witness, LeftBound104338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107269

namespace LeftBound107273
def owner : Owner := ⟨.program ⟨214⟩, ⟨29782⟩⟩
def transferEvent : Nat := 107273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107271 .coefficient, .predecessor 1 107272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107271 .coefficient)
      LeftBound107268.bound (LeftBound107268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107272 .coefficient)
      LeftBound104148.bound (LeftBound104148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107268.bound, LeftBound104148.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107268.bound, LeftBound104148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107268.actual selector witness, LeftBound104148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107273

namespace LeftBound107274
def owner : Owner := ⟨.program ⟨214⟩, ⟨29782⟩⟩
def transferEvent : Nat := 107274
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107270 .summary, .result 104155 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107270 .summary)
      LeftBound107269.bound (LeftBound107269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29565⟩⟩) (rawTerms := some (Proof.Events419.exact107270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104155 .summary)
      LeftBound104150.bound (LeftBound104150.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29781⟩⟩) (rawTerms := some (Proof.Events406.exact104155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107269.bound, LeftBound104150.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107269.bound, LeftBound104150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107269.actual selector witness, LeftBound104150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107274

namespace LeftBound107278
def owner : Owner := ⟨.program ⟨214⟩, ⟨30059⟩⟩
def transferEvent : Nat := 107278
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107276 .coefficient, .predecessor 1 107277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107276 .coefficient)
      LeftBound107273.bound (LeftBound107273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107277 .coefficient)
      LeftBound103960.bound (LeftBound103960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact103967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107273.bound, LeftBound103960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107273.bound, LeftBound103960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107273.actual selector witness, LeftBound103960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107278

namespace LeftBound107279
def owner : Owner := ⟨.program ⟨214⟩, ⟨30059⟩⟩
def transferEvent : Nat := 107279
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107275 .summary, .result 103967 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107275 .summary)
      LeftBound107274.bound (LeftBound107274.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29782⟩⟩) (rawTerms := some (Proof.Events419.exact107275RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103967 .summary)
      LeftBound103962.bound (LeftBound103962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30058⟩⟩) (rawTerms := some (Proof.Events406.exact103967RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107274.bound, LeftBound103962.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107274.bound, LeftBound103962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107274.actual selector witness, LeftBound103962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107279

namespace LeftBound107283
def owner : Owner := ⟨.program ⟨214⟩, ⟨30070⟩⟩
def transferEvent : Nat := 107283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107281 .coefficient, .predecessor 1 107282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107281 .coefficient)
      LeftBound107278.bound (LeftBound107278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107282 .coefficient)
      LeftBound103772.bound (LeftBound103772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107278.bound, LeftBound103772.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107278.bound, LeftBound103772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107278.actual selector witness, LeftBound103772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107283

namespace LeftBound107284
def owner : Owner := ⟨.program ⟨214⟩, ⟨30070⟩⟩
def transferEvent : Nat := 107284
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107280 .summary, .result 103779 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107280 .summary)
      LeftBound107279.bound (LeftBound107279.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30059⟩⟩) (rawTerms := some (Proof.Events419.exact107280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103779 .summary)
      LeftBound103774.bound (LeftBound103774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30068⟩⟩) (rawTerms := some (Proof.Events405.exact103779RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107279.bound, LeftBound103774.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107279.bound, LeftBound103774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107279.actual selector witness, LeftBound103774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107284

namespace LeftBound107288
def owner : Owner := ⟨.program ⟨214⟩, ⟨30071⟩⟩
def transferEvent : Nat := 107288
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107286 .coefficient) (.predecessor 1 107287 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107286 .coefficient)
      LeftBound107283.bound (LeftBound107283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107287 .coefficient)
      LeftBound6190.bound (LeftBound6190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound107283.bound LeftBound6190.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107283.bound, LeftBound6190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound107283.actual selector witness) * (LeftBound6190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107288

namespace LeftBound107289
def owner : Owner := ⟨.program ⟨214⟩, ⟨30071⟩⟩
def transferEvent : Nat := 107289
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6639⟩⟩]⟩ [⟨.result 6187 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6187 .coefficient)
      LeftAuthority6186.bound (LeftAuthority6186.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6639⟩⟩) (rawTerms := some (Proof.Events024.exact6187RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6186.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6186.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6186.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107289

namespace LeftBound107290
def owner : Owner := ⟨.program ⟨214⟩, ⟨30071⟩⟩
def transferEvent : Nat := 107290
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 107285 .summary) (.transfer 107289) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107285 .summary)
      LeftBound107284.bound (LeftBound107284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30070⟩⟩) (rawTerms := some (Proof.Events419.exact107285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 107289)
      LeftBound107289.bound (LeftBound107289.actual selector witness) := by
  exact .transfer (LeftBound107289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound107284.bound LeftBound107289.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178603181359497216, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107284.bound, LeftBound107289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound107284.actual selector witness) * (LeftBound107289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107290

namespace LeftBound107370
def owner : Owner := ⟨.program ⟨214⟩, ⟨30072⟩⟩
def transferEvent : Nat := 107370
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107368 .coefficient, .predecessor 1 107369 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107368 .coefficient)
      LeftBound107186.bound (LeftBound107186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107369 .coefficient)
      LeftBound107288.bound (LeftBound107288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107288.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107186.bound, LeftBound107288.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107186.bound, LeftBound107288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107186.actual selector witness, LeftBound107288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107370

namespace LeftBound107371
def owner : Owner := ⟨.program ⟨214⟩, ⟨30072⟩⟩
def transferEvent : Nat := 107371
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107190 .summary, .result 107367 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107190 .summary)
      LeftBound107189.bound (LeftBound107189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7805⟩⟩) (rawTerms := some (Proof.Events418.exact107190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107367 .summary)
      LeftBound107290.bound (LeftBound107290.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30071⟩⟩) (rawTerms := some (Proof.Events419.exact107367RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107189.bound, LeftBound107290.bound]
def bound : CoeffClass := .finite ⟨1149729608724517268372876178603181359497268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107189.bound, LeftBound107290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107189.actual selector witness, LeftBound107290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107371

namespace LeftBound107375
def owner : Owner := ⟨.program ⟨214⟩, ⟨30130⟩⟩
def transferEvent : Nat := 107375
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107373 .coefficient, .predecessor 1 107374 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107373 .coefficient)
      LeftBound107370.bound (LeftBound107370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107374 .coefficient)
      LeftBound94279.bound (LeftBound94279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94279.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107370.bound, LeftBound94279.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107370.bound, LeftBound94279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107370.actual selector witness, LeftBound94279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107375

namespace LeftBound107376
def owner : Owner := ⟨.program ⟨214⟩, ⟨30130⟩⟩
def transferEvent : Nat := 107376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107372 .summary, .result 94340 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107372 .summary)
      LeftBound107371.bound (LeftBound107371.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30072⟩⟩) (rawTerms := some (Proof.Events419.exact107372RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94340 .summary)
      LeftBound94281.bound (LeftBound94281.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30129⟩⟩) (rawTerms := some (Proof.Events368.exact94340RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107371.bound, LeftBound94281.bound]
def bound : CoeffClass := .finite ⟨4219527209422351428897269014962119225933291323444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107371.bound, LeftBound94281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107371.actual selector witness, LeftBound94281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107376

namespace LeftBound107380
def owner : Owner := ⟨.program ⟨214⟩, ⟨30131⟩⟩
def transferEvent : Nat := 107380
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107378 .coefficient, .predecessor 1 107379 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107378 .coefficient)
      LeftBound107375.bound (LeftBound107375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107379 .coefficient)
      LeftBound79690.bound (LeftBound79690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107375.bound, LeftBound79690.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107375.bound, LeftBound79690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107375.actual selector witness, LeftBound79690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107380

namespace LeftBound107381
def owner : Owner := ⟨.program ⟨214⟩, ⟨30131⟩⟩
def transferEvent : Nat := 107381
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107377 .summary, .result 79751 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107377 .summary)
      LeftBound107376.bound (LeftBound107376.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30130⟩⟩) (rawTerms := some (Proof.Events419.exact107377RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79751 .summary)
      LeftBound79692.bound (LeftBound79692.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30108⟩⟩) (rawTerms := some (Proof.Events311.exact79751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107376.bound, LeftBound79692.bound]
def bound : CoeffClass := .finite ⟨8439053269115094133277269657048059848685223149620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107376.bound, LeftBound79692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107376.actual selector witness, LeftBound79692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107381

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
