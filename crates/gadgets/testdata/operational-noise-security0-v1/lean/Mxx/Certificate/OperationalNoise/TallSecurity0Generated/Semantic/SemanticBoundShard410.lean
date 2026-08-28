import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard344
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard409

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60996
def owner : Owner := ⟨.program ⟨214⟩, ⟨22775⟩⟩
def transferEvent : Nat := 60996
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 60995) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 60995)
      LeftBound60995.bound (LeftBound60995.actual selector witness) := by
  exact .transfer (LeftBound60995.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound60995.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound60995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound60995.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60996

namespace LeftBound61091
def owner : Owner := ⟨.program ⟨214⟩, ⟨17016⟩⟩
def transferEvent : Nat := 61091
def frameStart : Nat := 61052
def rule : BoundRule := .identity (.predecessor 0 61090 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61090 .coefficient)
      LeftAuthority61088.bound (LeftAuthority61088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61088.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61088.derived selector witness)

def rawBound : CoeffClass := LeftAuthority61088.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority61088.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61091

namespace LeftBound61108
def owner : Owner := ⟨.program ⟨214⟩, ⟨17055⟩⟩
def transferEvent : Nat := 61108
def frameStart : Nat := 61052
def rule : BoundRule := .sum [.predecessor 0 61106 .coefficient, .predecessor 1 61107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61106 .coefficient)
      LeftBound61091.bound (LeftBound61091.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61107 .coefficient)
      LeftAuthority61104.bound (LeftAuthority61104.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority61104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61091.bound, LeftAuthority61104.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61091.bound, LeftAuthority61104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61091.actual selector witness, LeftAuthority61104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61108

namespace LeftBound61111
def owner : Owner := ⟨.program ⟨214⟩, ⟨17056⟩⟩
def transferEvent : Nat := 61111
def frameStart : Nat := 61052
def rule : BoundRule := .identity (.predecessor 0 61110 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61110 .coefficient)
      LeftBound61108.bound (LeftBound61108.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61108.derived selector witness)

def rawBound : CoeffClass := LeftBound61108.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound61108.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61111

namespace LeftBound61117
def owner : Owner := ⟨.program ⟨214⟩, ⟨17057⟩⟩
def transferEvent : Nat := 61117
def frameStart : Nat := 61052
def rule : BoundRule := .product (.predecessor 0 61115 .coefficient) (.predecessor 1 61116 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61115 .coefficient)
      LeftAuthority61113.bound (LeftAuthority61113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61116 .coefficient)
      LeftBound61111.bound (LeftBound61111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority61113.bound LeftBound61111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61113.bound, LeftBound61111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority61113.actual selector witness) * (LeftBound61111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61117

namespace LeftBound61125
def owner : Owner := ⟨.program ⟨214⟩, ⟨17058⟩⟩
def transferEvent : Nat := 61125
def frameStart : Nat := 61052
def rule : BoundRule := .sum [.predecessor 0 61123 .coefficient, .predecessor 1 61124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61123 .coefficient)
      LeftAuthority61121.bound (LeftAuthority61121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61124 .coefficient)
      LeftBound61117.bound (LeftBound61117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61117.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61121.bound, LeftBound61117.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61121.bound, LeftBound61117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61121.actual selector witness, LeftBound61117.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61125

namespace LeftBound61129
def owner : Owner := ⟨.program ⟨214⟩, ⟨30133⟩⟩
def transferEvent : Nat := 61129
def frameStart : Nat := 61052
def rule : BoundRule := .product (.predecessor 0 61127 .coefficient) (.predecessor 1 61128 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61127 .coefficient)
      LeftBound61125.bound (LeftBound61125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61128 .coefficient)
      LeftAuthority61102.bound (LeftAuthority61102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61102.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61125.bound LeftAuthority61102.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61125.bound, LeftAuthority61102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61125.actual selector witness) * (LeftAuthority61102.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61129

namespace LeftBound61140
def owner : Owner := ⟨.program ⟨214⟩, ⟨18130⟩⟩
def transferEvent : Nat := 61140
def frameStart : Nat := 61052
def rule : BoundRule := .product (.predecessor 0 61138 .coefficient) (.predecessor 1 61139 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61138 .coefficient)
      LeftAuthority61113.bound (LeftAuthority61113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61139 .coefficient)
      LeftAuthority61136.bound (LeftAuthority61136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61136.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority61113.bound LeftAuthority61136.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61113.bound, LeftAuthority61136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority61113.actual selector witness) * (LeftAuthority61136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61140

namespace LeftBound61148
def owner : Owner := ⟨.program ⟨214⟩, ⟨18131⟩⟩
def transferEvent : Nat := 61148
def frameStart : Nat := 61052
def rule : BoundRule := .sum [.predecessor 0 61146 .coefficient, .predecessor 1 61147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61146 .coefficient)
      LeftAuthority61144.bound (LeftAuthority61144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61147 .coefficient)
      LeftBound61140.bound (LeftBound61140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61144.bound, LeftBound61140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61144.bound, LeftBound61140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61144.actual selector witness, LeftBound61140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61148

namespace LeftBound61152
def owner : Owner := ⟨.program ⟨214⟩, ⟨30138⟩⟩
def transferEvent : Nat := 61152
def frameStart : Nat := 61052
def rule : BoundRule := .sum [.predecessor 0 61150 .coefficient, .predecessor 1 61151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61150 .coefficient)
      LeftBound61148.bound (LeftBound61148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61151 .coefficient)
      LeftBound61129.bound (LeftBound61129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61148.bound, LeftBound61129.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61148.bound, LeftBound61129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61148.actual selector witness, LeftBound61129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61152

namespace LeftBound61165
def owner : Owner := ⟨.program ⟨214⟩, ⟨30135⟩⟩
def transferEvent : Nat := 61165
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 61163 .coefficient, .predecessor 1 61164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61163 .coefficient)
      LeftBound60994.bound (LeftBound60994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61164 .coefficient)
      LeftBound60977.bound (LeftBound60977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact60984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60994.bound, LeftBound60977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60994.bound, LeftBound60977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60994.actual selector witness, LeftBound60977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61165

namespace LeftBound61168
def owner : Owner := ⟨.program ⟨214⟩, ⟨30135⟩⟩
def transferEvent : Nat := 61168
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 61162 .summary, .result 60984 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61162 .summary)
      LeftBound60996.bound (LeftBound60996.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22775⟩⟩) (rawTerms := some (Proof.Events238.exact61162RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60984 .summary)
      LeftBound60979.bound (LeftBound60979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30134⟩⟩) (rawTerms := some (Proof.Events238.exact60984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60996.bound, LeftBound60979.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60996.bound, LeftBound60979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60996.actual selector witness, LeftBound60979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61168

namespace LeftBound61172
def owner : Owner := ⟨.program ⟨214⟩, ⟨30136⟩⟩
def transferEvent : Nat := 61172
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61170 .coefficient) (.predecessor 1 61171 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61170 .coefficient)
      LeftBound61165.bound (LeftBound61165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61171 .coefficient)
      LeftBound5518.bound (LeftBound5518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61165.bound LeftBound5518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61165.bound, LeftBound5518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61165.actual selector witness) * (LeftBound5518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61172

namespace LeftBound61173
def owner : Owner := ⟨.program ⟨214⟩, ⟨30136⟩⟩
def transferEvent : Nat := 61173
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩ [⟨.result 5515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5515 .coefficient)
      LeftAuthority5514.bound (LeftAuthority5514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6657⟩⟩) (rawTerms := some (Proof.Events021.exact5515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5514.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61173

namespace LeftBound61174
def owner : Owner := ⟨.program ⟨214⟩, ⟨30136⟩⟩
def transferEvent : Nat := 61174
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61169 .summary) (.transfer 61173) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61169 .summary)
      LeftBound61168.bound (LeftBound61168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30135⟩⟩) (rawTerms := some (Proof.Events238.exact61169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61173)
      LeftBound61173.bound (LeftBound61173.actual selector witness) := by
  exact .transfer (LeftBound61173.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61168.bound LeftBound61173.bound
def bound : CoeffClass := .finite ⟨4743639307122182955475140608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61168.bound, LeftBound61173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61168.actual selector witness) * (LeftBound61173.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61174

namespace LeftBound61189
def owner : Owner := ⟨.program ⟨214⟩, ⟨29827⟩⟩
def transferEvent : Nat := 61189
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61187 .coefficient) (.predecessor 1 61188 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61187 .coefficient)
      LeftBound51426.bound (LeftBound51426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61188 .coefficient)
      LeftAuthority61185.bound (LeftAuthority61185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61185.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51426.bound LeftAuthority61185.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51426.bound, LeftAuthority61185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51426.actual selector witness) * (LeftAuthority61185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61189

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
