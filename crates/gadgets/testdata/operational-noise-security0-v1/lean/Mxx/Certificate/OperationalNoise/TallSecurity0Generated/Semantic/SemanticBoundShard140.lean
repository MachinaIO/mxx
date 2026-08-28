import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard139

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21938
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def transferEvent : Nat := 21938
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21933 .summary) (.transfer 21937) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21933 .summary)
      LeftBound21931.bound (LeftBound21931.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13183⟩⟩) (rawTerms := some (Proof.Events085.exact21933RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21937)
      LeftBound21937.bound (LeftBound21937.actual selector witness) := by
  exact .transfer (LeftBound21937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21931.bound LeftBound21937.bound
def bound : CoeffClass := .finite ⟨48256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21931.bound, LeftBound21937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21931.actual selector witness) * (LeftBound21937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21938

namespace LeftBound21944
def owner : Owner := ⟨.program ⟨214⟩, ⟨10256⟩⟩
def transferEvent : Nat := 21944
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 21942 .coefficient) (.predecessor 1 21943 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21942 .coefficient)
      LeftAuthority867.bound (LeftAuthority867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21943 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority867.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority867.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority867.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21944

namespace LeftBound21949
def owner : Owner := ⟨.program ⟨214⟩, ⟨7339⟩⟩
def transferEvent : Nat := 21949
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21947 .coefficient) (.predecessor 1 21948 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21947 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21948 .coefficient)
      LeftBound7013.bound (LeftBound7013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound7013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound7013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound7013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21949

namespace LeftBound21954
def owner : Owner := ⟨.program ⟨214⟩, ⟨10257⟩⟩
def transferEvent : Nat := 21954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21952 .coefficient, .predecessor 1 21953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21952 .coefficient)
      LeftBound21949.bound (LeftBound21949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21953 .coefficient)
      LeftBound21944.bound (LeftBound21944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21949.bound, LeftBound21944.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21949.bound, LeftBound21944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21949.actual selector witness, LeftBound21944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21954

namespace LeftBound21958
def owner : Owner := ⟨.program ⟨214⟩, ⟨10258⟩⟩
def transferEvent : Nat := 21958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21956 .coefficient, .predecessor 1 21957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21956 .coefficient)
      LeftBound21954.bound (LeftBound21954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21957 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21954.bound, LeftBound7005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21954.bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21954.actual selector witness, LeftBound7005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21958

namespace LeftBound21959
def owner : Owner := ⟨.program ⟨214⟩, ⟨10258⟩⟩
def transferEvent : Nat := 21959
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩ [⟨.result 7006 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7006 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨83⟩⟩) (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7005.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21959

namespace LeftBound21964
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def transferEvent : Nat := 21964
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21962 .coefficient) (.predecessor 1 21963 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21962 .coefficient)
      LeftBound21958.bound (LeftBound21958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21963 .coefficient)
      LeftBound7002.bound (LeftBound7002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21958.bound LeftBound7002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21958.bound, LeftBound7002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21958.actual selector witness) * (LeftBound7002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21964

namespace LeftBound21965
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def transferEvent : Nat := 21965
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩ [⟨.result 6999 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6999 .coefficient)
      LeftAuthority6998.bound (LeftAuthority6998.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7879⟩⟩) (rawTerms := some (Proof.Events027.exact6999RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6998.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6998.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6998.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21965

namespace LeftBound21966
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def transferEvent : Nat := 21966
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21961 .summary) (.transfer 21965) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21961 .summary)
      LeftBound21959.bound (LeftBound21959.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10258⟩⟩) (rawTerms := some (Proof.Events085.exact21961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21965)
      LeftBound21965.bound (LeftBound21965.actual selector witness) := by
  exact .transfer (LeftBound21965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21959.bound LeftBound21965.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21959.bound, LeftBound21965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21959.actual selector witness) * (LeftBound21965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21966

namespace LeftBound21974
def owner : Owner := ⟨.program ⟨214⟩, ⟨13185⟩⟩
def transferEvent : Nat := 21974
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21972 .coefficient, .predecessor 1 21973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21972 .coefficient)
      LeftBound21964.bound (LeftBound21964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21973 .coefficient)
      LeftBound21936.bound (LeftBound21936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21964.bound, LeftBound21936.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21964.bound, LeftBound21936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21964.actual selector witness, LeftBound21936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21974

namespace LeftBound21976
def owner : Owner := ⟨.program ⟨214⟩, ⟨13185⟩⟩
def transferEvent : Nat := 21976
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21971 .summary, .result 21941 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21971 .summary)
      LeftBound21966.bound (LeftBound21966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10259⟩⟩) (rawTerms := some (Proof.Events085.exact21971RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21941 .summary)
      LeftBound21938.bound (LeftBound21938.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13184⟩⟩) (rawTerms := some (Proof.Events085.exact21941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21966.bound, LeftBound21938.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21966.bound, LeftBound21938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21966.actual selector witness, LeftBound21938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21976

namespace LeftBound21980
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def transferEvent : Nat := 21980
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21978 .coefficient) (.predecessor 1 21979 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21978 .coefficient)
      LeftBound21974.bound (LeftBound21974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21979 .coefficient)
      LeftAuthority21912.bound (LeftAuthority21912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21912.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21974.bound LeftAuthority21912.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21974.bound, LeftAuthority21912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21974.actual selector witness) * (LeftAuthority21912.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21980

namespace LeftBound21981
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def transferEvent : Nat := 21981
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩ [⟨.result 21913 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21913 .coefficient)
      LeftAuthority21912.bound (LeftAuthority21912.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25696⟩⟩) (rawTerms := some (Proof.Events085.exact21913RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21912.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21912.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21912.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21981

namespace LeftBound21982
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def transferEvent : Nat := 21982
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21977 .summary) (.transfer 21981) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21977 .summary)
      LeftBound21976.bound (LeftBound21976.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13185⟩⟩) (rawTerms := some (Proof.Events085.exact21977RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21981)
      LeftBound21981.bound (LeftBound21981.actual selector witness) := by
  exact .transfer (LeftBound21981.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21976.bound LeftBound21981.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21976.bound, LeftBound21981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21976.actual selector witness) * (LeftBound21981.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21982

namespace LeftBound21993
def owner : Owner := ⟨.program ⟨214⟩, ⟨20190⟩⟩
def transferEvent : Nat := 21993
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 21991 .coefficient) (.value (.predecessor 1 21992 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21991 .coefficient)
      LeftAuthority21989.bound (LeftAuthority21989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21992 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority21989.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21989.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21989.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21993

namespace LeftBound21997
def owner : Owner := ⟨.program ⟨214⟩, ⟨20191⟩⟩
def transferEvent : Nat := 21997
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21995 .coefficient) (.predecessor 1 21996 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21995 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21996 .coefficient)
      LeftBound21993.bound (LeftBound21993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21993.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound21993.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound21993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound21993.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21997

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
