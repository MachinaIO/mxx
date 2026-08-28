import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard678

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98968
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def transferEvent : Nat := 98968
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27831⟩⟩]⟩ [⟨.result 98711 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98711 .coefficient)
      LeftAuthority98710.bound (LeftAuthority98710.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27831⟩⟩) (rawTerms := some (Proof.Events385.exact98711RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98710.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98710.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98710.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98968

namespace LeftBound98969
def owner : Owner := ⟨.program ⟨214⟩, ⟨27833⟩⟩
def transferEvent : Nat := 98969
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98964 .summary) (.transfer 98968) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98964 .summary)
      LeftBound98963.bound (LeftBound98963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26055⟩⟩) (rawTerms := some (Proof.Events386.exact98964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98968)
      LeftBound98968.bound (LeftBound98968.actual selector witness) := by
  exact .transfer (LeftBound98968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98963.bound LeftBound98968.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98963.bound, LeftBound98968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98963.actual selector witness) * (LeftBound98968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98969

namespace LeftBound98980
def owner : Owner := ⟨.program ⟨214⟩, ⟨21391⟩⟩
def transferEvent : Nat := 98980
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98978 .coefficient) (.value (.predecessor 1 98979 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98978 .coefficient)
      LeftAuthority98976.bound (LeftAuthority98976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98979 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98976.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98976.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98976.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98980

namespace LeftBound98984
def owner : Owner := ⟨.program ⟨214⟩, ⟨21392⟩⟩
def transferEvent : Nat := 98984
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98982 .coefficient) (.predecessor 1 98983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98982 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98983 .coefficient)
      LeftBound98980.bound (LeftBound98980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound98980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound98980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound98980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98984

namespace LeftBound98985
def owner : Owner := ⟨.program ⟨214⟩, ⟨21392⟩⟩
def transferEvent : Nat := 98985
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21389⟩⟩]⟩ [⟨.result 98977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98977 .coefficient)
      LeftAuthority98976.bound (LeftAuthority98976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21389⟩⟩) (rawTerms := some (Proof.Events386.exact98977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98976.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98985

namespace LeftBound98986
def owner : Owner := ⟨.program ⟨214⟩, ⟨21392⟩⟩
def transferEvent : Nat := 98986
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 98985) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98985)
      LeftBound98985.bound (LeftBound98985.actual selector witness) := by
  exact .transfer (LeftBound98985.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound98985.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound98985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound98985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98986

namespace LeftBound99057
def owner : Owner := ⟨.program ⟨214⟩, ⟨15931⟩⟩
def transferEvent : Nat := 99057
def frameStart : Nat := 99030
def rule : BoundRule := .identity (.predecessor 0 99056 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99056 .coefficient)
      LeftAuthority99054.bound (LeftAuthority99054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact99055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99054.derived selector witness)

def rawBound : CoeffClass := LeftAuthority99054.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority99054.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99057

namespace LeftBound99074
def owner : Owner := ⟨.program ⟨214⟩, ⟨16007⟩⟩
def transferEvent : Nat := 99074
def frameStart : Nat := 99030
def rule : BoundRule := .sum [.predecessor 0 99072 .coefficient, .predecessor 1 99073 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99072 .coefficient)
      LeftBound99057.bound (LeftBound99057.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99073 .coefficient)
      LeftAuthority99070.bound (LeftAuthority99070.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99057.bound, LeftAuthority99070.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99057.bound, LeftAuthority99070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99057.actual selector witness, LeftAuthority99070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99074

namespace LeftBound99077
def owner : Owner := ⟨.program ⟨214⟩, ⟨16008⟩⟩
def transferEvent : Nat := 99077
def frameStart : Nat := 99030
def rule : BoundRule := .identity (.predecessor 0 99076 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99076 .coefficient)
      LeftBound99074.bound (LeftBound99074.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99074.derived selector witness)

def rawBound : CoeffClass := LeftBound99074.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99074.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99077

namespace LeftBound99083
def owner : Owner := ⟨.program ⟨214⟩, ⟨16009⟩⟩
def transferEvent : Nat := 99083
def frameStart : Nat := 99030
def rule : BoundRule := .product (.predecessor 0 99081 .coefficient) (.predecessor 1 99082 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99081 .coefficient)
      LeftAuthority99079.bound (LeftAuthority99079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99079.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99082 .coefficient)
      LeftBound99077.bound (LeftBound99077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99077.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority99079.bound LeftBound99077.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99079.bound, LeftBound99077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority99079.actual selector witness) * (LeftBound99077.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99083

namespace LeftBound99091
def owner : Owner := ⟨.program ⟨214⟩, ⟨16010⟩⟩
def transferEvent : Nat := 99091
def frameStart : Nat := 99030
def rule : BoundRule := .sum [.predecessor 0 99089 .coefficient, .predecessor 1 99090 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99089 .coefficient)
      LeftAuthority99087.bound (LeftAuthority99087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99087.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99090 .coefficient)
      LeftBound99083.bound (LeftBound99083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99087.bound, LeftBound99083.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99087.bound, LeftBound99083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99087.actual selector witness, LeftBound99083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99091

namespace LeftBound99095
def owner : Owner := ⟨.program ⟨214⟩, ⟨27832⟩⟩
def transferEvent : Nat := 99095
def frameStart : Nat := 99030
def rule : BoundRule := .product (.predecessor 0 99093 .coefficient) (.predecessor 1 99094 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99093 .coefficient)
      LeftBound99091.bound (LeftBound99091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99094 .coefficient)
      LeftAuthority99068.bound (LeftAuthority99068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact99069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99068.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99091.bound LeftAuthority99068.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99091.bound, LeftAuthority99068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99091.actual selector witness) * (LeftAuthority99068.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99095

namespace LeftBound99106
def owner : Owner := ⟨.program ⟨214⟩, ⟨15980⟩⟩
def transferEvent : Nat := 99106
def frameStart : Nat := 99030
def rule : BoundRule := .product (.predecessor 0 99104 .coefficient) (.predecessor 1 99105 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99104 .coefficient)
      LeftAuthority99079.bound (LeftAuthority99079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99079.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99105 .coefficient)
      LeftAuthority99102.bound (LeftAuthority99102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99102.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99079.bound LeftAuthority99102.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99079.bound, LeftAuthority99102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99079.actual selector witness) * (LeftAuthority99102.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99106

namespace LeftBound99114
def owner : Owner := ⟨.program ⟨214⟩, ⟨15981⟩⟩
def transferEvent : Nat := 99114
def frameStart : Nat := 99030
def rule : BoundRule := .sum [.predecessor 0 99112 .coefficient, .predecessor 1 99113 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99112 .coefficient)
      LeftAuthority99110.bound (LeftAuthority99110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99113 .coefficient)
      LeftBound99106.bound (LeftBound99106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99106.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99110.bound, LeftBound99106.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99110.bound, LeftBound99106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99110.actual selector witness, LeftBound99106.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99114

namespace LeftBound99118
def owner : Owner := ⟨.program ⟨214⟩, ⟨27836⟩⟩
def transferEvent : Nat := 99118
def frameStart : Nat := 99030
def rule : BoundRule := .sum [.predecessor 0 99116 .coefficient, .predecessor 1 99117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99116 .coefficient)
      LeftBound99114.bound (LeftBound99114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99117 .coefficient)
      LeftBound99095.bound (LeftBound99095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99114.bound, LeftBound99095.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99114.bound, LeftBound99095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99114.actual selector witness, LeftBound99095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99118

namespace LeftBound99131
def owner : Owner := ⟨.program ⟨214⟩, ⟨27834⟩⟩
def transferEvent : Nat := 99131
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99129 .coefficient, .predecessor 1 99130 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99129 .coefficient)
      LeftBound98984.bound (LeftBound98984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99130 .coefficient)
      LeftBound98967.bound (LeftBound98967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events386.exact98974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98984.bound, LeftBound98967.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98984.bound, LeftBound98967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98984.actual selector witness, LeftBound98967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99131

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
