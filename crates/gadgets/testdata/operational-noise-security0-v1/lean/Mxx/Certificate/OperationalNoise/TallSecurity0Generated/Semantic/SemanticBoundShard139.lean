import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard138

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21835
def owner : Owner := ⟨.program ⟨214⟩, ⟨17064⟩⟩
def transferEvent : Nat := 21835
def frameStart : Nat := 21776
def rule : BoundRule := .identity (.predecessor 0 21834 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21834 .coefficient)
      LeftBound21832.bound (LeftBound21832.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21832.derived selector witness)

def rawBound : CoeffClass := LeftBound21832.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound21832.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21835

namespace LeftBound21841
def owner : Owner := ⟨.program ⟨214⟩, ⟨17065⟩⟩
def transferEvent : Nat := 21841
def frameStart : Nat := 21776
def rule : BoundRule := .product (.predecessor 0 21839 .coefficient) (.predecessor 1 21840 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21839 .coefficient)
      LeftAuthority21837.bound (LeftAuthority21837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21840 .coefficient)
      LeftBound21835.bound (LeftBound21835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21835.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority21837.bound LeftBound21835.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21837.bound, LeftBound21835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority21837.actual selector witness) * (LeftBound21835.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21841

namespace LeftBound21849
def owner : Owner := ⟨.program ⟨214⟩, ⟨17066⟩⟩
def transferEvent : Nat := 21849
def frameStart : Nat := 21776
def rule : BoundRule := .sum [.predecessor 0 21847 .coefficient, .predecessor 1 21848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21847 .coefficient)
      LeftAuthority21845.bound (LeftAuthority21845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21848 .coefficient)
      LeftBound21841.bound (LeftBound21841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21845.bound, LeftBound21841.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21845.bound, LeftBound21841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority21845.actual selector witness, LeftBound21841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21849

namespace LeftBound21853
def owner : Owner := ⟨.program ⟨214⟩, ⟨30184⟩⟩
def transferEvent : Nat := 21853
def frameStart : Nat := 21776
def rule : BoundRule := .product (.predecessor 0 21851 .coefficient) (.predecessor 1 21852 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21851 .coefficient)
      LeftBound21849.bound (LeftBound21849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21852 .coefficient)
      LeftAuthority21826.bound (LeftAuthority21826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21826.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21849.bound LeftAuthority21826.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21849.bound, LeftAuthority21826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21849.actual selector witness) * (LeftAuthority21826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21853

namespace LeftBound21864
def owner : Owner := ⟨.program ⟨214⟩, ⟨18180⟩⟩
def transferEvent : Nat := 21864
def frameStart : Nat := 21776
def rule : BoundRule := .product (.predecessor 0 21862 .coefficient) (.predecessor 1 21863 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21862 .coefficient)
      LeftAuthority21837.bound (LeftAuthority21837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21863 .coefficient)
      LeftAuthority21860.bound (LeftAuthority21860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority21837.bound LeftAuthority21860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21837.bound, LeftAuthority21860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority21837.actual selector witness) * (LeftAuthority21860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21864

namespace LeftBound21872
def owner : Owner := ⟨.program ⟨214⟩, ⟨18181⟩⟩
def transferEvent : Nat := 21872
def frameStart : Nat := 21776
def rule : BoundRule := .sum [.predecessor 0 21870 .coefficient, .predecessor 1 21871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21870 .coefficient)
      LeftAuthority21868.bound (LeftAuthority21868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21871 .coefficient)
      LeftBound21864.bound (LeftBound21864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21864.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21868.bound, LeftBound21864.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21868.bound, LeftBound21864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority21868.actual selector witness, LeftBound21864.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21872

namespace LeftBound21876
def owner : Owner := ⟨.program ⟨214⟩, ⟨30191⟩⟩
def transferEvent : Nat := 21876
def frameStart : Nat := 21776
def rule : BoundRule := .sum [.predecessor 0 21874 .coefficient, .predecessor 1 21875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21874 .coefficient)
      LeftBound21872.bound (LeftBound21872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21875 .coefficient)
      LeftBound21853.bound (LeftBound21853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21872.bound, LeftBound21853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21872.bound, LeftBound21853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21872.actual selector witness, LeftBound21853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21876

namespace LeftBound21889
def owner : Owner := ⟨.program ⟨214⟩, ⟨30186⟩⟩
def transferEvent : Nat := 21889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21887 .coefficient, .predecessor 1 21888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21887 .coefficient)
      LeftBound21718.bound (LeftBound21718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21888 .coefficient)
      LeftBound21701.bound (LeftBound21701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21718.bound, LeftBound21701.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21718.bound, LeftBound21701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21718.actual selector witness, LeftBound21701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21889

namespace LeftBound21892
def owner : Owner := ⟨.program ⟨214⟩, ⟨30186⟩⟩
def transferEvent : Nat := 21892
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21886 .summary, .result 21708 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21886 .summary)
      LeftBound21720.bound (LeftBound21720.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22855⟩⟩) (rawTerms := some (Proof.Events085.exact21886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21708 .summary)
      LeftBound21703.bound (LeftBound21703.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30185⟩⟩) (rawTerms := some (Proof.Events084.exact21708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21720.bound, LeftBound21703.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21720.bound, LeftBound21703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21720.actual selector witness, LeftBound21703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21892

namespace LeftBound21916
def owner : Owner := ⟨.program ⟨214⟩, ⟨13181⟩⟩
def transferEvent : Nat := 21916
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 21914 .coefficient) (.predecessor 1 21915 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21914 .coefficient)
      LeftAuthority864.bound (LeftAuthority864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21915 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority864.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority864.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority864.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21916

namespace LeftBound21921
def owner : Owner := ⟨.program ⟨214⟩, ⟨7359⟩⟩
def transferEvent : Nat := 21921
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21919 .coefficient) (.predecessor 1 21920 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21919 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21920 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21921

namespace LeftBound21926
def owner : Owner := ⟨.program ⟨214⟩, ⟨13182⟩⟩
def transferEvent : Nat := 21926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21924 .coefficient, .predecessor 1 21925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21924 .coefficient)
      LeftBound21921.bound (LeftBound21921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21925 .coefficient)
      LeftBound21916.bound (LeftBound21916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21921.bound, LeftBound21916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21921.bound, LeftBound21916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21921.actual selector witness, LeftBound21916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21926

namespace LeftBound21930
def owner : Owner := ⟨.program ⟨214⟩, ⟨13183⟩⟩
def transferEvent : Nat := 21930
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21928 .coefficient, .predecessor 1 21929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21928 .coefficient)
      LeftBound21926.bound (LeftBound21926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21929 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21926.bound, LeftBound6964.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21926.bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21926.actual selector witness, LeftBound6964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21930

namespace LeftBound21931
def owner : Owner := ⟨.program ⟨214⟩, ⟨13183⟩⟩
def transferEvent : Nat := 21931
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩ [⟨.result 6965 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6965 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6964.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6964.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21931

namespace LeftBound21936
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def transferEvent : Nat := 21936
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21934 .coefficient) (.predecessor 1 21935 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21934 .coefficient)
      LeftBound21930.bound (LeftBound21930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21935 .coefficient)
      LeftAuthority867.bound (LeftAuthority867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority867.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound21930.bound LeftAuthority867.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21930.bound, LeftAuthority867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound21930.actual selector witness) * (LeftAuthority867.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21936

namespace LeftBound21937
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def transferEvent : Nat := 21937
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10255⟩⟩], []⟩ [⟨.result 868 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 868 .coefficient)
      LeftAuthority867.bound (LeftAuthority867.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10255⟩⟩) (rawTerms := some (Proof.Events003.exact868RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority867.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority867.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority867.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21937

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
