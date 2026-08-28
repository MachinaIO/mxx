import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard140

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21998
def owner : Owner := ⟨.program ⟨214⟩, ⟨20191⟩⟩
def transferEvent : Nat := 21998
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20188⟩⟩]⟩ [⟨.result 21990 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21990 .coefficient)
      LeftAuthority21989.bound (LeftAuthority21989.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20188⟩⟩) (rawTerms := some (Proof.Events085.exact21990RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21989.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21989.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21989.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21998

namespace LeftBound21999
def owner : Owner := ⟨.program ⟨214⟩, ⟨20191⟩⟩
def transferEvent : Nat := 21999
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 21998) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21998)
      LeftBound21998.bound (LeftBound21998.actual selector witness) := by
  exact .transfer (LeftBound21998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound21998.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound21998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound21998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21999

namespace LeftBound22078
def owner : Owner := ⟨.program ⟨214⟩, ⟨13179⟩⟩
def transferEvent : Nat := 22078
def frameStart : Nat := 22049
def rule : BoundRule := .product (.predecessor 0 22076 .coefficient) (.predecessor 1 22077 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22076 .coefficient)
      LeftAuthority22074.bound (LeftAuthority22074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22077 .coefficient)
      LeftAuthority22071.bound (LeftAuthority22071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22071.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22074.bound LeftAuthority22071.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22074.bound, LeftAuthority22071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22074.actual selector witness) * (LeftAuthority22071.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22078

namespace LeftBound22082
def owner : Owner := ⟨.program ⟨214⟩, ⟨13180⟩⟩
def transferEvent : Nat := 22082
def frameStart : Nat := 22049
def rule : BoundRule := .identity (.predecessor 0 22081 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22081 .coefficient)
      LeftBound22078.bound (LeftBound22078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22078.derived selector witness)

def rawBound : CoeffClass := LeftBound22078.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22078.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22082

namespace LeftBound22099
def owner : Owner := ⟨.program ⟨214⟩, ⟨13262⟩⟩
def transferEvent : Nat := 22099
def frameStart : Nat := 22049
def rule : BoundRule := .sum [.predecessor 0 22097 .coefficient, .predecessor 1 22098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22097 .coefficient)
      LeftBound22082.bound (LeftBound22082.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22098 .coefficient)
      LeftAuthority22095.bound (LeftAuthority22095.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22082.bound, LeftAuthority22095.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22082.bound, LeftAuthority22095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22082.actual selector witness, LeftAuthority22095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22099

namespace LeftBound22102
def owner : Owner := ⟨.program ⟨214⟩, ⟨13263⟩⟩
def transferEvent : Nat := 22102
def frameStart : Nat := 22049
def rule : BoundRule := .identity (.predecessor 0 22101 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22101 .coefficient)
      LeftBound22099.bound (LeftBound22099.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22099.derived selector witness)

def rawBound : CoeffClass := LeftBound22099.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22099.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22102

namespace LeftBound22108
def owner : Owner := ⟨.program ⟨214⟩, ⟨13264⟩⟩
def transferEvent : Nat := 22108
def frameStart : Nat := 22049
def rule : BoundRule := .product (.predecessor 0 22106 .coefficient) (.predecessor 1 22107 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22106 .coefficient)
      LeftAuthority22104.bound (LeftAuthority22104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22104.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22107 .coefficient)
      LeftBound22102.bound (LeftBound22102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22102.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority22104.bound LeftBound22102.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22104.bound, LeftBound22102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority22104.actual selector witness) * (LeftBound22102.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22108

namespace LeftBound22124
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 22124
def frameStart : Nat := 22049
def rule : BoundRule := .scale (.predecessor 0 22122 .coefficient) (.value (.predecessor 1 22123 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22122 .coefficient)
      LeftAuthority22120.bound (LeftAuthority22120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22123 .coefficient)
      LeftAuthority22111.bound (LeftAuthority22111.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22111.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22120.bound LeftAuthority22111.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22120.bound, LeftAuthority22111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22120.actual selector witness) * (LeftAuthority22111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22124

namespace LeftBound22127
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 22127
def frameStart : Nat := 22049
def rule : BoundRule := .identity (.predecessor 0 22126 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22126 .coefficient)
      LeftAuthority22114.bound (LeftAuthority22114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22114.derived selector witness)

def rawBound : CoeffClass := LeftAuthority22114.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority22114.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22127

namespace LeftBound22131
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 22131
def frameStart : Nat := 22049
def rule : BoundRule := .product (.predecessor 0 22129 .coefficient) (.predecessor 1 22130 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22129 .coefficient)
      LeftBound22127.bound (LeftBound22127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22130 .coefficient)
      LeftBound22124.bound (LeftBound22124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22127.bound LeftBound22124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22127.bound, LeftBound22124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22127.actual selector witness) * (LeftBound22124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22131

namespace LeftBound22136
def owner : Owner := ⟨.program ⟨214⟩, ⟨13265⟩⟩
def transferEvent : Nat := 22136
def frameStart : Nat := 22049
def rule : BoundRule := .sum [.predecessor 0 22134 .coefficient, .predecessor 1 22135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22134 .coefficient)
      LeftBound22131.bound (LeftBound22131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22135 .coefficient)
      LeftBound22108.bound (LeftBound22108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22131.bound, LeftBound22108.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22131.bound, LeftBound22108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22131.actual selector witness, LeftBound22108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22136

namespace LeftBound22140
def owner : Owner := ⟨.program ⟨214⟩, ⟨25699⟩⟩
def transferEvent : Nat := 22140
def frameStart : Nat := 22049
def rule : BoundRule := .product (.predecessor 0 22138 .coefficient) (.predecessor 1 22139 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22138 .coefficient)
      LeftBound22136.bound (LeftBound22136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22139 .coefficient)
      LeftAuthority22093.bound (LeftAuthority22093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22136.bound LeftAuthority22093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22136.bound, LeftAuthority22093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22136.actual selector witness) * (LeftAuthority22093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22140

namespace LeftBound22151
def owner : Owner := ⟨.program ⟨214⟩, ⟨16885⟩⟩
def transferEvent : Nat := 22151
def frameStart : Nat := 22049
def rule : BoundRule := .product (.predecessor 0 22149 .coefficient) (.predecessor 1 22150 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22149 .coefficient)
      LeftAuthority22104.bound (LeftAuthority22104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22104.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22150 .coefficient)
      LeftAuthority22147.bound (LeftAuthority22147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22104.bound LeftAuthority22147.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22104.bound, LeftAuthority22147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22104.actual selector witness) * (LeftAuthority22147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22151

namespace LeftBound22159
def owner : Owner := ⟨.program ⟨214⟩, ⟨16886⟩⟩
def transferEvent : Nat := 22159
def frameStart : Nat := 22049
def rule : BoundRule := .sum [.predecessor 0 22157 .coefficient, .predecessor 1 22158 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22157 .coefficient)
      LeftAuthority22155.bound (LeftAuthority22155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22158 .coefficient)
      LeftBound22151.bound (LeftBound22151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22155.bound, LeftBound22151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22155.bound, LeftBound22151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22155.actual selector witness, LeftBound22151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22159

namespace LeftBound22163
def owner : Owner := ⟨.program ⟨214⟩, ⟨25700⟩⟩
def transferEvent : Nat := 22163
def frameStart : Nat := 22049
def rule : BoundRule := .sum [.predecessor 0 22161 .coefficient, .predecessor 1 22162 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22161 .coefficient)
      LeftBound22159.bound (LeftBound22159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22162 .coefficient)
      LeftBound22140.bound (LeftBound22140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22159.bound, LeftBound22140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22159.bound, LeftBound22140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22159.actual selector witness, LeftBound22140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22163

namespace LeftBound22176
def owner : Owner := ⟨.program ⟨214⟩, ⟨25698⟩⟩
def transferEvent : Nat := 22176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22174 .coefficient, .predecessor 1 22175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22174 .coefficient)
      LeftBound21997.bound (LeftBound21997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22175 .coefficient)
      LeftBound21980.bound (LeftBound21980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21997.bound, LeftBound21980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21997.bound, LeftBound21980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21997.actual selector witness, LeftBound21980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22176

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
