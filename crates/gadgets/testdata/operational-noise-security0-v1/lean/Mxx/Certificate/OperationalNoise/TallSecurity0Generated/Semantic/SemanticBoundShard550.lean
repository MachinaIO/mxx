import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard549

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80954
def owner : Owner := ⟨.program ⟨214⟩, ⟨12965⟩⟩
def transferEvent : Nat := 80954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80949 .summary, .result 80919 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80949 .summary)
      LeftBound80944.bound (LeftBound80944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10139⟩⟩) (rawTerms := some (Proof.Events316.exact80949RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80919 .summary)
      LeftBound80916.bound (LeftBound80916.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12964⟩⟩) (rawTerms := some (Proof.Events316.exact80919RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80944.bound, LeftBound80916.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80944.bound, LeftBound80916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80944.actual selector witness, LeftBound80916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80954

namespace LeftBound80958
def owner : Owner := ⟨.program ⟨214⟩, ⟨25605⟩⟩
def transferEvent : Nat := 80958
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80956 .coefficient) (.predecessor 1 80957 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80956 .coefficient)
      LeftBound80952.bound (LeftBound80952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact80955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80957 .coefficient)
      LeftAuthority80890.bound (LeftAuthority80890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80952.bound LeftAuthority80890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80952.bound, LeftAuthority80890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80952.actual selector witness) * (LeftAuthority80890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80958

namespace LeftBound80959
def owner : Owner := ⟨.program ⟨214⟩, ⟨25605⟩⟩
def transferEvent : Nat := 80959
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25604⟩⟩]⟩ [⟨.result 80891 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80891 .coefficient)
      LeftAuthority80890.bound (LeftAuthority80890.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25604⟩⟩) (rawTerms := some (Proof.Events315.exact80891RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80890.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80890.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80890.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80959

namespace LeftBound80960
def owner : Owner := ⟨.program ⟨214⟩, ⟨25605⟩⟩
def transferEvent : Nat := 80960
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80955 .summary) (.transfer 80959) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80955 .summary)
      LeftBound80954.bound (LeftBound80954.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12965⟩⟩) (rawTerms := some (Proof.Events316.exact80955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80959)
      LeftBound80959.bound (LeftBound80959.actual selector witness) := by
  exact .transfer (LeftBound80959.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80954.bound LeftBound80959.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80954.bound, LeftBound80959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80954.actual selector witness) * (LeftBound80959.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80960

namespace LeftBound80971
def owner : Owner := ⟨.program ⟨214⟩, ⟨20106⟩⟩
def transferEvent : Nat := 80971
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 80969 .coefficient) (.value (.predecessor 1 80970 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80969 .coefficient)
      LeftAuthority80967.bound (LeftAuthority80967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact80968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80970 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80967.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80967.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80967.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80971

namespace LeftBound80975
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def transferEvent : Nat := 80975
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80973 .coefficient) (.predecessor 1 80974 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80973 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80974 .coefficient)
      LeftBound80971.bound (LeftBound80971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact80972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80971.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound80971.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound80971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound80971.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80975

namespace LeftBound80976
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def transferEvent : Nat := 80976
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20104⟩⟩]⟩ [⟨.result 80968 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80968 .coefficient)
      LeftAuthority80967.bound (LeftAuthority80967.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20104⟩⟩) (rawTerms := some (Proof.Events316.exact80968RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80967.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80967.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80967.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80976

namespace LeftBound80977
def owner : Owner := ⟨.program ⟨214⟩, ⟨20107⟩⟩
def transferEvent : Nat := 80977
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 80976) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80976)
      LeftBound80976.bound (LeftBound80976.actual selector witness) := by
  exact .transfer (LeftBound80976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound80976.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound80976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound80976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80977

namespace LeftBound81056
def owner : Owner := ⟨.program ⟨214⟩, ⟨12959⟩⟩
def transferEvent : Nat := 81056
def frameStart : Nat := 81027
def rule : BoundRule := .product (.predecessor 0 81054 .coefficient) (.predecessor 1 81055 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81054 .coefficient)
      LeftAuthority81052.bound (LeftAuthority81052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81055 .coefficient)
      LeftAuthority81049.bound (LeftAuthority81049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81049.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81052.bound LeftAuthority81049.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81052.bound, LeftAuthority81049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81052.actual selector witness) * (LeftAuthority81049.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81056

namespace LeftBound81060
def owner : Owner := ⟨.program ⟨214⟩, ⟨12960⟩⟩
def transferEvent : Nat := 81060
def frameStart : Nat := 81027
def rule : BoundRule := .identity (.predecessor 0 81059 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81059 .coefficient)
      LeftBound81056.bound (LeftBound81056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81056.derived selector witness)

def rawBound : CoeffClass := LeftBound81056.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81056.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81060

namespace LeftBound81077
def owner : Owner := ⟨.program ⟨214⟩, ⟨13054⟩⟩
def transferEvent : Nat := 81077
def frameStart : Nat := 81027
def rule : BoundRule := .sum [.predecessor 0 81075 .coefficient, .predecessor 1 81076 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81075 .coefficient)
      LeftBound81060.bound (LeftBound81060.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81060.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81076 .coefficient)
      LeftAuthority81073.bound (LeftAuthority81073.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81073.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81060.bound, LeftAuthority81073.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81060.bound, LeftAuthority81073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81060.actual selector witness, LeftAuthority81073.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81077

namespace LeftBound81080
def owner : Owner := ⟨.program ⟨214⟩, ⟨13055⟩⟩
def transferEvent : Nat := 81080
def frameStart : Nat := 81027
def rule : BoundRule := .identity (.predecessor 0 81079 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81079 .coefficient)
      LeftBound81077.bound (LeftBound81077.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81077.derived selector witness)

def rawBound : CoeffClass := LeftBound81077.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81077.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81080

namespace LeftBound81086
def owner : Owner := ⟨.program ⟨214⟩, ⟨13056⟩⟩
def transferEvent : Nat := 81086
def frameStart : Nat := 81027
def rule : BoundRule := .product (.predecessor 0 81084 .coefficient) (.predecessor 1 81085 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81084 .coefficient)
      LeftAuthority81082.bound (LeftAuthority81082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81085 .coefficient)
      LeftBound81080.bound (LeftBound81080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority81082.bound LeftBound81080.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81082.bound, LeftBound81080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority81082.actual selector witness) * (LeftBound81080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81086

namespace LeftBound81100
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 81100
def frameStart : Nat := 81027
def rule : BoundRule := .scale (.predecessor 0 81098 .coefficient) (.value (.predecessor 1 81099 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81098 .coefficient)
      LeftAuthority81096.bound (LeftAuthority81096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81096.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81099 .coefficient)
      LeftAuthority81030.bound (LeftAuthority81030.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81030.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81096.bound LeftAuthority81030.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81096.bound, LeftAuthority81030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81096.actual selector witness) * (LeftAuthority81030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81100

namespace LeftBound81103
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 81103
def frameStart : Nat := 81027
def rule : BoundRule := .identity (.predecessor 0 81102 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81102 .coefficient)
      LeftAuthority81090.bound (LeftAuthority81090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81090.derived selector witness)

def rawBound : CoeffClass := LeftAuthority81090.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority81090.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81103

namespace LeftBound81107
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 81107
def frameStart : Nat := 81027
def rule : BoundRule := .product (.predecessor 0 81105 .coefficient) (.predecessor 1 81106 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81105 .coefficient)
      LeftBound81103.bound (LeftBound81103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81106 .coefficient)
      LeftBound81100.bound (LeftBound81100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events316.exact81101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81100.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81103.bound LeftBound81100.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81103.bound, LeftBound81100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81103.actual selector witness) * (LeftBound81100.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81107

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
