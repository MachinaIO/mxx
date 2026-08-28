import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard623

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92075
def owner : Owner := ⟨.program ⟨214⟩, ⟨28296⟩⟩
def transferEvent : Nat := 92075
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92073 .coefficient, .predecessor 1 92074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92073 .coefficient)
      LeftBound91904.bound (LeftBound91904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92074 .coefficient)
      LeftBound91887.bound (LeftBound91887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91904.bound, LeftBound91887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91904.bound, LeftBound91887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91904.actual selector witness, LeftBound91887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92075

namespace LeftBound92078
def owner : Owner := ⟨.program ⟨214⟩, ⟨28296⟩⟩
def transferEvent : Nat := 92078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92072 .summary, .result 91894 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92072 .summary)
      LeftBound91906.bound (LeftBound91906.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21619⟩⟩) (rawTerms := some (Proof.Events359.exact92072RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91894 .summary)
      LeftBound91889.bound (LeftBound91889.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28295⟩⟩) (rawTerms := some (Proof.Events358.exact91894RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91906.bound, LeftBound91889.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91906.bound, LeftBound91889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91906.actual selector witness, LeftBound91889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92078

namespace LeftBound92082
def owner : Owner := ⟨.program ⟨214⟩, ⟨28297⟩⟩
def transferEvent : Nat := 92082
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92080 .coefficient) (.predecessor 1 92081 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92080 .coefficient)
      LeftBound92075.bound (LeftBound92075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92081 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92075.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92075.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92075.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92082

namespace LeftBound92083
def owner : Owner := ⟨.program ⟨214⟩, ⟨28297⟩⟩
def transferEvent : Nat := 92083
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92083

namespace LeftBound92084
def owner : Owner := ⟨.program ⟨214⟩, ⟨28297⟩⟩
def transferEvent : Nat := 92084
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92079 .summary) (.transfer 92083) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92079 .summary)
      LeftBound92078.bound (LeftBound92078.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28296⟩⟩) (rawTerms := some (Proof.Events359.exact92079RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92083)
      LeftBound92083.bound (LeftBound92083.actual selector witness) := by
  exact .transfer (LeftBound92083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92078.bound LeftBound92083.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92078.bound, LeftBound92083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92078.actual selector witness) * (LeftBound92083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92084

namespace LeftBound92099
def owner : Owner := ⟨.program ⟨214⟩, ⟨28078⟩⟩
def transferEvent : Nat := 92099
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92097 .coefficient) (.predecessor 1 92098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92097 .coefficient)
      LeftBound84512.bound (LeftBound84512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92098 .coefficient)
      LeftAuthority92095.bound (LeftAuthority92095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84512.bound LeftAuthority92095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84512.bound, LeftAuthority92095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84512.actual selector witness) * (LeftAuthority92095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92099

namespace LeftBound92100
def owner : Owner := ⟨.program ⟨214⟩, ⟨28078⟩⟩
def transferEvent : Nat := 92100
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28076⟩⟩]⟩ [⟨.result 92096 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92096 .coefficient)
      LeftAuthority92095.bound (LeftAuthority92095.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28076⟩⟩) (rawTerms := some (Proof.Events359.exact92096RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92095.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92095.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92095.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92100

namespace LeftBound92101
def owner : Owner := ⟨.program ⟨214⟩, ⟨28078⟩⟩
def transferEvent : Nat := 92101
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84516 .summary) (.transfer 92100) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84516 .summary)
      LeftBound84515.bound (LeftBound84515.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26145⟩⟩) (rawTerms := some (Proof.Events330.exact84516RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92100)
      LeftBound92100.bound (LeftBound92100.actual selector witness) := by
  exact .transfer (LeftBound92100.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84515.bound LeftBound92100.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84515.bound, LeftBound92100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84515.actual selector witness) * (LeftBound92100.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92101

namespace LeftBound92112
def owner : Owner := ⟨.program ⟨214⟩, ⟨21474⟩⟩
def transferEvent : Nat := 92112
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92110 .coefficient) (.value (.predecessor 1 92111 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92110 .coefficient)
      LeftAuthority92108.bound (LeftAuthority92108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92111 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92108.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92108.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92108.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92112

namespace LeftBound92116
def owner : Owner := ⟨.program ⟨214⟩, ⟨21475⟩⟩
def transferEvent : Nat := 92116
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92114 .coefficient) (.predecessor 1 92115 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92114 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92115 .coefficient)
      LeftBound92112.bound (LeftBound92112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92112.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound92112.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound92112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound92112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92116

namespace LeftBound92117
def owner : Owner := ⟨.program ⟨214⟩, ⟨21475⟩⟩
def transferEvent : Nat := 92117
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21472⟩⟩]⟩ [⟨.result 92109 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92109 .coefficient)
      LeftAuthority92108.bound (LeftAuthority92108.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21472⟩⟩) (rawTerms := some (Proof.Events359.exact92109RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92108.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92108.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92108.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92117

namespace LeftBound92118
def owner : Owner := ⟨.program ⟨214⟩, ⟨21475⟩⟩
def transferEvent : Nat := 92118
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 92117) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92117)
      LeftBound92117.bound (LeftBound92117.actual selector witness) := by
  exact .transfer (LeftBound92117.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound92117.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound92117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound92117.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92118

namespace LeftBound92213
def owner : Owner := ⟨.program ⟨214⟩, ⟨16060⟩⟩
def transferEvent : Nat := 92213
def frameStart : Nat := 92174
def rule : BoundRule := .identity (.predecessor 0 92212 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92212 .coefficient)
      LeftAuthority92210.bound (LeftAuthority92210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92210.derived selector witness)

def rawBound : CoeffClass := LeftAuthority92210.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority92210.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92213

namespace LeftBound92230
def owner : Owner := ⟨.program ⟨214⟩, ⟨16134⟩⟩
def transferEvent : Nat := 92230
def frameStart : Nat := 92174
def rule : BoundRule := .sum [.predecessor 0 92228 .coefficient, .predecessor 1 92229 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92228 .coefficient)
      LeftBound92213.bound (LeftBound92213.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92229 .coefficient)
      LeftAuthority92226.bound (LeftAuthority92226.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92213.bound, LeftAuthority92226.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92213.bound, LeftAuthority92226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92213.actual selector witness, LeftAuthority92226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92230

namespace LeftBound92233
def owner : Owner := ⟨.program ⟨214⟩, ⟨16135⟩⟩
def transferEvent : Nat := 92233
def frameStart : Nat := 92174
def rule : BoundRule := .identity (.predecessor 0 92232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92232 .coefficient)
      LeftBound92230.bound (LeftBound92230.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92230.derived selector witness)

def rawBound : CoeffClass := LeftBound92230.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound92230.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92233

namespace LeftBound92239
def owner : Owner := ⟨.program ⟨214⟩, ⟨16136⟩⟩
def transferEvent : Nat := 92239
def frameStart : Nat := 92174
def rule : BoundRule := .product (.predecessor 0 92237 .coefficient) (.predecessor 1 92238 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92237 .coefficient)
      LeftAuthority92235.bound (LeftAuthority92235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92238 .coefficient)
      LeftBound92233.bound (LeftBound92233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority92235.bound LeftBound92233.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92235.bound, LeftBound92233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority92235.actual selector witness) * (LeftBound92233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92239

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
