import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard305

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound45967
def owner : Owner := ⟨.program ⟨214⟩, ⟨18372⟩⟩
def transferEvent : Nat := 45967
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45965 .coefficient, .predecessor 1 45966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45965 .coefficient)
      LeftBound45963.bound (LeftBound45963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45966 .coefficient)
      LeftAuthority45589.bound (LeftAuthority45589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45589.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45963.bound, LeftAuthority45589.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45963.bound, LeftAuthority45589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45963.actual selector witness, LeftAuthority45589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45967

namespace LeftBound45971
def owner : Owner := ⟨.program ⟨214⟩, ⟨18373⟩⟩
def transferEvent : Nat := 45971
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45969 .coefficient, .predecessor 1 45970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45969 .coefficient)
      LeftBound45967.bound (LeftBound45967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45970 .coefficient)
      LeftAuthority45566.bound (LeftAuthority45566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events177.exact45567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45967.bound, LeftAuthority45566.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45967.bound, LeftAuthority45566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45967.actual selector witness, LeftAuthority45566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45971

namespace LeftBound45975
def owner : Owner := ⟨.program ⟨214⟩, ⟨18374⟩⟩
def transferEvent : Nat := 45975
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45973 .coefficient, .predecessor 1 45974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45973 .coefficient)
      LeftBound45971.bound (LeftBound45971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45974 .coefficient)
      LeftAuthority45543.bound (LeftAuthority45543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events177.exact45544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45971.bound, LeftAuthority45543.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45971.bound, LeftAuthority45543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45971.actual selector witness, LeftAuthority45543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45975

namespace LeftBound45979
def owner : Owner := ⟨.program ⟨214⟩, ⟨18375⟩⟩
def transferEvent : Nat := 45979
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45977 .coefficient, .predecessor 1 45978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45977 .coefficient)
      LeftBound45975.bound (LeftBound45975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45978 .coefficient)
      LeftAuthority45520.bound (LeftAuthority45520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events177.exact45521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45975.bound, LeftAuthority45520.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45975.bound, LeftAuthority45520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45975.actual selector witness, LeftAuthority45520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45979

namespace LeftBound45982
def owner : Owner := ⟨.program ⟨214⟩, ⟨18376⟩⟩
def transferEvent : Nat := 45982
def frameStart : Nat := 45478
def rule : BoundRule := .identity (.predecessor 0 45981 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45981 .coefficient)
      LeftBound45979.bound (LeftBound45979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact45980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45979.derived selector witness)

def rawBound : CoeffClass := LeftBound45979.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound45979.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound45982

namespace LeftBound45999
def owner : Owner := ⟨.program ⟨214⟩, ⟨18655⟩⟩
def transferEvent : Nat := 45999
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 45997 .coefficient, .predecessor 1 45998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 45997 .coefficient)
      LeftBound45982.bound (LeftBound45982.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound45982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 45998 .coefficient)
      LeftAuthority45995.bound (LeftAuthority45995.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority45995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45982.bound, LeftAuthority45995.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45982.bound, LeftAuthority45995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound45982.actual selector witness, LeftAuthority45995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45999

namespace LeftBound46002
def owner : Owner := ⟨.program ⟨214⟩, ⟨18656⟩⟩
def transferEvent : Nat := 46002
def frameStart : Nat := 45478
def rule : BoundRule := .identity (.predecessor 0 46001 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46001 .coefficient)
      LeftBound45999.bound (LeftBound45999.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound45999.derived selector witness)

def rawBound : CoeffClass := LeftBound45999.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound45999.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46002

namespace LeftBound46008
def owner : Owner := ⟨.program ⟨214⟩, ⟨18657⟩⟩
def transferEvent : Nat := 46008
def frameStart : Nat := 45478
def rule : BoundRule := .product (.predecessor 0 46006 .coefficient) (.predecessor 1 46007 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46006 .coefficient)
      LeftAuthority46004.bound (LeftAuthority46004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46007 .coefficient)
      LeftBound46002.bound (LeftBound46002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority46004.bound LeftBound46002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46004.bound, LeftBound46002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority46004.actual selector witness) * (LeftBound46002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46008

namespace LeftBound46084
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 46084
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46082 .coefficient, .predecessor 1 46083 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46082 .coefficient)
      LeftAuthority46080.bound (LeftAuthority46080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46083 .coefficient)
      LeftAuthority46077.bound (LeftAuthority46077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46077.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46080.bound, LeftAuthority46077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46080.bound, LeftAuthority46077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46080.actual selector witness, LeftAuthority46077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46084

namespace LeftBound46088
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 46088
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46086 .coefficient, .predecessor 1 46087 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46086 .coefficient)
      LeftBound46084.bound (LeftBound46084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46087 .coefficient)
      LeftAuthority46074.bound (LeftAuthority46074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46084.bound, LeftAuthority46074.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46084.bound, LeftAuthority46074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46084.actual selector witness, LeftAuthority46074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46088

namespace LeftBound46092
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 46092
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46090 .coefficient, .predecessor 1 46091 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46090 .coefficient)
      LeftBound46088.bound (LeftBound46088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46091 .coefficient)
      LeftAuthority46071.bound (LeftAuthority46071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46088.bound, LeftAuthority46071.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46088.bound, LeftAuthority46071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46088.actual selector witness, LeftAuthority46071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46092

namespace LeftBound46096
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 46096
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46094 .coefficient, .predecessor 1 46095 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46094 .coefficient)
      LeftBound46092.bound (LeftBound46092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46095 .coefficient)
      LeftAuthority46068.bound (LeftAuthority46068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46092.bound, LeftAuthority46068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46092.bound, LeftAuthority46068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46092.actual selector witness, LeftAuthority46068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46096

namespace LeftBound46100
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 46100
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46098 .coefficient, .predecessor 1 46099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46098 .coefficient)
      LeftBound46096.bound (LeftBound46096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46099 .coefficient)
      LeftAuthority46065.bound (LeftAuthority46065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46065.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46096.bound, LeftAuthority46065.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46096.bound, LeftAuthority46065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46096.actual selector witness, LeftAuthority46065.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46100

namespace LeftBound46104
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 46104
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46102 .coefficient, .predecessor 1 46103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46102 .coefficient)
      LeftBound46100.bound (LeftBound46100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46103 .coefficient)
      LeftAuthority46062.bound (LeftAuthority46062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46062.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46100.bound, LeftAuthority46062.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46100.bound, LeftAuthority46062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46100.actual selector witness, LeftAuthority46062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46104

namespace LeftBound46108
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 46108
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46106 .coefficient, .predecessor 1 46107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46106 .coefficient)
      LeftBound46104.bound (LeftBound46104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46107 .coefficient)
      LeftAuthority46059.bound (LeftAuthority46059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46104.bound, LeftAuthority46059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46104.bound, LeftAuthority46059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46104.actual selector witness, LeftAuthority46059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46108

namespace LeftBound46112
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 46112
def frameStart : Nat := 45478
def rule : BoundRule := .sum [.predecessor 0 46110 .coefficient, .predecessor 1 46111 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46110 .coefficient)
      LeftBound46108.bound (LeftBound46108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46111 .coefficient)
      LeftAuthority46056.bound (LeftAuthority46056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events179.exact46057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46108.bound, LeftAuthority46056.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46108.bound, LeftAuthority46056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46108.actual selector witness, LeftAuthority46056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46112

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
