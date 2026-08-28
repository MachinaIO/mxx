import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard661

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96993
def owner : Owner := ⟨.program ⟨214⟩, ⟨7121⟩⟩
def transferEvent : Nat := 96993
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96991 .coefficient) (.predecessor 1 96992 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96991 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96992 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96993

namespace LeftBound96998
def owner : Owner := ⟨.program ⟨214⟩, ⟨11937⟩⟩
def transferEvent : Nat := 96998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96996 .coefficient, .predecessor 1 96997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96996 .coefficient)
      LeftBound96993.bound (LeftBound96993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96997 .coefficient)
      LeftBound96988.bound (LeftBound96988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96988.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96993.bound, LeftBound96988.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96993.bound, LeftBound96988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96993.actual selector witness, LeftBound96988.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96998

namespace LeftBound97002
def owner : Owner := ⟨.program ⟨214⟩, ⟨11938⟩⟩
def transferEvent : Nat := 97002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97000 .coefficient, .predecessor 1 97001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97000 .coefficient)
      LeftBound96998.bound (LeftBound96998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97001 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96998.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96998.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96998.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97002

namespace LeftBound97003
def owner : Owner := ⟨.program ⟨214⟩, ⟨11938⟩⟩
def transferEvent : Nat := 97003
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97003

namespace LeftBound97008
def owner : Owner := ⟨.program ⟨214⟩, ⟨11939⟩⟩
def transferEvent : Nat := 97008
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97006 .coefficient) (.predecessor 1 97007 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97006 .coefficient)
      LeftBound97002.bound (LeftBound97002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact97005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97007 .coefficient)
      LeftAuthority4706.bound (LeftAuthority4706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound97002.bound LeftAuthority4706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97002.bound, LeftAuthority4706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound97002.actual selector witness) * (LeftAuthority4706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97008

namespace LeftBound97009
def owner : Owner := ⟨.program ⟨214⟩, ⟨11939⟩⟩
def transferEvent : Nat := 97009
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9700⟩⟩], []⟩ [⟨.result 4707 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4707 .coefficient)
      LeftAuthority4706.bound (LeftAuthority4706.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9700⟩⟩) (rawTerms := some (Proof.Events018.exact4707RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4706.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4706.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4706.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97009

namespace LeftBound97010
def owner : Owner := ⟨.program ⟨214⟩, ⟨11939⟩⟩
def transferEvent : Nat := 97010
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97005 .summary) (.transfer 97009) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97005 .summary)
      LeftBound97003.bound (LeftBound97003.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11938⟩⟩) (rawTerms := some (Proof.Events378.exact97005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97009)
      LeftBound97009.bound (LeftBound97009.actual selector witness) := by
  exact .transfer (LeftBound97009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound97003.bound LeftBound97009.bound
def bound : CoeffClass := .finite ⟨29952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97003.bound, LeftBound97009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound97003.actual selector witness) * (LeftBound97009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97010

namespace LeftBound97016
def owner : Owner := ⟨.program ⟨214⟩, ⟨9701⟩⟩
def transferEvent : Nat := 97016
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 97014 .coefficient) (.predecessor 1 97015 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97014 .coefficient)
      LeftAuthority4706.bound (LeftAuthority4706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97015 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4706.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4706.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4706.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97016

namespace LeftBound97021
def owner : Owner := ⟨.program ⟨214⟩, ⟨7101⟩⟩
def transferEvent : Nat := 97021
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97019 .coefficient) (.predecessor 1 97020 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97019 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97020 .coefficient)
      LeftBound9518.bound (LeftBound9518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound9518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound9518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound9518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97021

namespace LeftBound97026
def owner : Owner := ⟨.program ⟨214⟩, ⟨9702⟩⟩
def transferEvent : Nat := 97026
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97024 .coefficient, .predecessor 1 97025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97024 .coefficient)
      LeftBound97021.bound (LeftBound97021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact97023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97025 .coefficient)
      LeftBound97016.bound (LeftBound97016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact97018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97021.bound, LeftBound97016.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97021.bound, LeftBound97016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97021.actual selector witness, LeftBound97016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97026

namespace LeftBound97030
def owner : Owner := ⟨.program ⟨214⟩, ⟨9703⟩⟩
def transferEvent : Nat := 97030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97028 .coefficient, .predecessor 1 97029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97028 .coefficient)
      LeftBound97026.bound (LeftBound97026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97029 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97026.bound, LeftBound9510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97026.bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97026.actual selector witness, LeftBound9510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97030

namespace LeftBound97031
def owner : Owner := ⟨.program ⟨214⟩, ⟨9703⟩⟩
def transferEvent : Nat := 97031
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩ [⟨.result 9511 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9511 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨78⟩⟩) (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9510.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9510.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97031

namespace LeftBound97036
def owner : Owner := ⟨.program ⟨214⟩, ⟨9704⟩⟩
def transferEvent : Nat := 97036
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97034 .coefficient) (.predecessor 1 97035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97034 .coefficient)
      LeftBound97030.bound (LeftBound97030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97035 .coefficient)
      LeftBound9507.bound (LeftBound9507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97030.bound LeftBound9507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97030.bound, LeftBound9507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97030.actual selector witness) * (LeftBound9507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97036

namespace LeftBound97037
def owner : Owner := ⟨.program ⟨214⟩, ⟨9704⟩⟩
def transferEvent : Nat := 97037
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩ [⟨.result 9504 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9504 .coefficient)
      LeftAuthority9503.bound (LeftAuthority9503.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7864⟩⟩) (rawTerms := some (Proof.Events037.exact9504RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9503.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9503.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9503.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97037

namespace LeftBound97038
def owner : Owner := ⟨.program ⟨214⟩, ⟨9704⟩⟩
def transferEvent : Nat := 97038
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97033 .summary) (.transfer 97037) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97033 .summary)
      LeftBound97031.bound (LeftBound97031.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9703⟩⟩) (rawTerms := some (Proof.Events379.exact97033RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97037)
      LeftBound97037.bound (LeftBound97037.actual selector witness) := by
  exact .transfer (LeftBound97037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97031.bound LeftBound97037.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97031.bound, LeftBound97037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97031.actual selector witness) * (LeftBound97037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97038

namespace LeftBound97046
def owner : Owner := ⟨.program ⟨214⟩, ⟨11940⟩⟩
def transferEvent : Nat := 97046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97044 .coefficient, .predecessor 1 97045 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97044 .coefficient)
      LeftBound97036.bound (LeftBound97036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97045 .coefficient)
      LeftBound97008.bound (LeftBound97008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact97013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97008.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97036.bound, LeftBound97008.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97036.bound, LeftBound97008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97036.actual selector witness, LeftBound97008.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97046

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
