import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard550

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81112
def owner : Owner := ⟨.program ⟨214⟩, ⟨13057⟩⟩
def transferEvent : Nat := 81112
def frameStart : Nat := 81027
def rule : BoundRule := .sum [.predecessor 0 81110 .coefficient, .predecessor 1 81111 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81110 .coefficient)
      LeftBound81107.bound (LeftBound81107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81111 .coefficient)
      LeftBound81086.bound (LeftBound81086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81107.bound, LeftBound81086.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81107.bound, LeftBound81086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81107.actual selector witness, LeftBound81086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81112

namespace LeftBound81116
def owner : Owner := ⟨.program ⟨214⟩, ⟨25607⟩⟩
def transferEvent : Nat := 81116
def frameStart : Nat := 81027
def rule : BoundRule := .product (.predecessor 0 81114 .coefficient) (.predecessor 1 81115 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81114 .coefficient)
      LeftBound81112.bound (LeftBound81112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81115 .coefficient)
      LeftAuthority81071.bound (LeftAuthority81071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81071.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81112.bound LeftAuthority81071.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81112.bound, LeftAuthority81071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81112.actual selector witness) * (LeftAuthority81071.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81116

namespace LeftBound81127
def owner : Owner := ⟨.program ⟨214⟩, ⟨16754⟩⟩
def transferEvent : Nat := 81127
def frameStart : Nat := 81027
def rule : BoundRule := .product (.predecessor 0 81125 .coefficient) (.predecessor 1 81126 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81125 .coefficient)
      LeftAuthority81082.bound (LeftAuthority81082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81126 .coefficient)
      LeftAuthority81123.bound (LeftAuthority81123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81123.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81082.bound LeftAuthority81123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81082.bound, LeftAuthority81123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81082.actual selector witness) * (LeftAuthority81123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81127

namespace LeftBound81135
def owner : Owner := ⟨.program ⟨214⟩, ⟨16755⟩⟩
def transferEvent : Nat := 81135
def frameStart : Nat := 81027
def rule : BoundRule := .sum [.predecessor 0 81133 .coefficient, .predecessor 1 81134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81133 .coefficient)
      LeftAuthority81131.bound (LeftAuthority81131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81134 .coefficient)
      LeftBound81127.bound (LeftBound81127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81127.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81131.bound, LeftBound81127.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81131.bound, LeftBound81127.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81131.actual selector witness, LeftBound81127.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81135

namespace LeftBound81139
def owner : Owner := ⟨.program ⟨214⟩, ⟨25608⟩⟩
def transferEvent : Nat := 81139
def frameStart : Nat := 81027
def rule : BoundRule := .sum [.predecessor 0 81137 .coefficient, .predecessor 1 81138 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81137 .coefficient)
      LeftBound81135.bound (LeftBound81135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81138 .coefficient)
      LeftBound81116.bound (LeftBound81116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81135.bound, LeftBound81116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81135.bound, LeftBound81116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81135.actual selector witness, LeftBound81116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81139

namespace LeftBound81152
def owner : Owner := ⟨.program ⟨214⟩, ⟨25606⟩⟩
def transferEvent : Nat := 81152
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81150 .coefficient, .predecessor 1 81151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81150 .coefficient)
      LeftBound80975.bound (LeftBound80975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81151 .coefficient)
      LeftBound80958.bound (LeftBound80958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact80965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80958.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80975.bound, LeftBound80958.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80975.bound, LeftBound80958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80975.actual selector witness, LeftBound80958.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81152

namespace LeftBound81155
def owner : Owner := ⟨.program ⟨214⟩, ⟨25606⟩⟩
def transferEvent : Nat := 81155
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81149 .summary, .result 80965 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81149 .summary)
      LeftBound80977.bound (LeftBound80977.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20107⟩⟩) (rawTerms := some (Proof.Events316.exact81149RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80965 .summary)
      LeftBound80960.bound (LeftBound80960.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25605⟩⟩) (rawTerms := some (Proof.Events316.exact80965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80977.bound, LeftBound80960.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80977.bound, LeftBound80960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80977.actual selector witness, LeftBound80960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81155

namespace LeftBound81159
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def transferEvent : Nat := 81159
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81157 .coefficient) (.predecessor 1 81158 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81157 .coefficient)
      LeftBound81152.bound (LeftBound81152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81158 .coefficient)
      LeftAuthority80880.bound (LeftAuthority80880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81152.bound LeftAuthority80880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81152.bound, LeftAuthority80880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81152.actual selector witness) * (LeftAuthority80880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81159

namespace LeftBound81160
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def transferEvent : Nat := 81160
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29602⟩⟩]⟩ [⟨.result 80881 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80881 .coefficient)
      LeftAuthority80880.bound (LeftAuthority80880.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29602⟩⟩) (rawTerms := some (Proof.Events315.exact80881RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80880.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80880.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80880.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81160

namespace LeftBound81161
def owner : Owner := ⟨.program ⟨214⟩, ⟨29604⟩⟩
def transferEvent : Nat := 81161
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81156 .summary) (.transfer 81160) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81156 .summary)
      LeftBound81155.bound (LeftBound81155.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25606⟩⟩) (rawTerms := some (Proof.Events317.exact81156RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81160)
      LeftBound81160.bound (LeftBound81160.actual selector witness) := by
  exact .transfer (LeftBound81160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81155.bound LeftBound81160.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81155.bound, LeftBound81160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81155.actual selector witness) * (LeftBound81160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81161

namespace LeftBound81172
def owner : Owner := ⟨.program ⟨214⟩, ⟨22554⟩⟩
def transferEvent : Nat := 81172
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 81170 .coefficient) (.value (.predecessor 1 81171 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81170 .coefficient)
      LeftAuthority81168.bound (LeftAuthority81168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81171 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81168.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81168.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81168.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81172

namespace LeftBound81176
def owner : Owner := ⟨.program ⟨214⟩, ⟨22555⟩⟩
def transferEvent : Nat := 81176
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81174 .coefficient) (.predecessor 1 81175 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81174 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81175 .coefficient)
      LeftBound81172.bound (LeftBound81172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound81172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound81172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound81172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81176

namespace LeftBound81177
def owner : Owner := ⟨.program ⟨214⟩, ⟨22555⟩⟩
def transferEvent : Nat := 81177
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22552⟩⟩]⟩ [⟨.result 81169 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81169 .coefficient)
      LeftAuthority81168.bound (LeftAuthority81168.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22552⟩⟩) (rawTerms := some (Proof.Events317.exact81169RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81168.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81168.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81168.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81177

namespace LeftBound81178
def owner : Owner := ⟨.program ⟨214⟩, ⟨22555⟩⟩
def transferEvent : Nat := 81178
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 81177) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81177)
      LeftBound81177.bound (LeftBound81177.actual selector witness) := by
  exact .transfer (LeftBound81177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound81177.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound81177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound81177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81178

namespace LeftBound81273
def owner : Owner := ⟨.program ⟨214⟩, ⟨16753⟩⟩
def transferEvent : Nat := 81273
def frameStart : Nat := 81234
def rule : BoundRule := .identity (.predecessor 0 81272 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81272 .coefficient)
      LeftAuthority81270.bound (LeftAuthority81270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81270.derived selector witness)

def rawBound : CoeffClass := LeftAuthority81270.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority81270.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81273

namespace LeftBound81290
def owner : Owner := ⟨.program ⟨214⟩, ⟨16827⟩⟩
def transferEvent : Nat := 81290
def frameStart : Nat := 81234
def rule : BoundRule := .sum [.predecessor 0 81288 .coefficient, .predecessor 1 81289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81288 .coefficient)
      LeftBound81273.bound (LeftBound81273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81289 .coefficient)
      LeftAuthority81286.bound (LeftAuthority81286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81273.bound, LeftAuthority81286.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81273.bound, LeftAuthority81286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81273.actual selector witness, LeftAuthority81286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81290

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
