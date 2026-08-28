import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard578

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84794
def owner : Owner := ⟨.program ⟨214⟩, ⟨14215⟩⟩
def transferEvent : Nat := 84794
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84789 .summary, .result 84759 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84789 .summary)
      LeftBound84784.bound (LeftBound84784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14214⟩⟩) (rawTerms := some (Proof.Events331.exact84789RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84759 .summary)
      LeftBound84756.bound (LeftBound84756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14210⟩⟩) (rawTerms := some (Proof.Events331.exact84759RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84784.bound, LeftBound84756.bound]
def bound : CoeffClass := .finite ⟨95435392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84784.bound, LeftBound84756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84784.actual selector witness, LeftBound84756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84794

namespace LeftBound84798
def owner : Owner := ⟨.program ⟨214⟩, ⟨26067⟩⟩
def transferEvent : Nat := 84798
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84796 .coefficient) (.predecessor 1 84797 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84796 .coefficient)
      LeftBound84792.bound (LeftBound84792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84797 .coefficient)
      LeftAuthority84730.bound (LeftAuthority84730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84730.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84792.bound LeftAuthority84730.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84792.bound, LeftAuthority84730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84792.actual selector witness) * (LeftAuthority84730.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84798

namespace LeftBound84799
def owner : Owner := ⟨.program ⟨214⟩, ⟨26067⟩⟩
def transferEvent : Nat := 84799
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26066⟩⟩]⟩ [⟨.result 84731 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84731 .coefficient)
      LeftAuthority84730.bound (LeftAuthority84730.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26066⟩⟩) (rawTerms := some (Proof.Events330.exact84731RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84730.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84730.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84730.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84799

namespace LeftBound84800
def owner : Owner := ⟨.program ⟨214⟩, ⟨26067⟩⟩
def transferEvent : Nat := 84800
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84795 .summary) (.transfer 84799) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84795 .summary)
      LeftBound84794.bound (LeftBound84794.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14215⟩⟩) (rawTerms := some (Proof.Events331.exact84795RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84799)
      LeftBound84799.bound (LeftBound84799.actual selector witness) := by
  exact .transfer (LeftBound84799.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84794.bound LeftBound84799.bound
def bound : CoeffClass := .finite ⟨350249415606272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84794.bound, LeftBound84799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84794.actual selector witness) * (LeftBound84799.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84800

namespace LeftBound84811
def owner : Owner := ⟨.program ⟨214⟩, ⟨19530⟩⟩
def transferEvent : Nat := 84811
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 84809 .coefficient) (.value (.predecessor 1 84810 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84809 .coefficient)
      LeftAuthority84807.bound (LeftAuthority84807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84810 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84807.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84807.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84807.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84811

namespace LeftBound84815
def owner : Owner := ⟨.program ⟨214⟩, ⟨19531⟩⟩
def transferEvent : Nat := 84815
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84813 .coefficient) (.predecessor 1 84814 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84813 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84814 .coefficient)
      LeftBound84811.bound (LeftBound84811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84811.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound84811.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound84811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound84811.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84815

namespace LeftBound84816
def owner : Owner := ⟨.program ⟨214⟩, ⟨19531⟩⟩
def transferEvent : Nat := 84816
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19528⟩⟩]⟩ [⟨.result 84808 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84808 .coefficient)
      LeftAuthority84807.bound (LeftAuthority84807.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19528⟩⟩) (rawTerms := some (Proof.Events331.exact84808RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84807.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84807.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84807.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84816

namespace LeftBound84817
def owner : Owner := ⟨.program ⟨214⟩, ⟨19531⟩⟩
def transferEvent : Nat := 84817
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 84816) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84816)
      LeftBound84816.bound (LeftBound84816.actual selector witness) := by
  exact .transfer (LeftBound84816.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound84816.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound84816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound84816.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84817

namespace LeftBound84896
def owner : Owner := ⟨.program ⟨214⟩, ⟨14208⟩⟩
def transferEvent : Nat := 84896
def frameStart : Nat := 84867
def rule : BoundRule := .product (.predecessor 0 84894 .coefficient) (.predecessor 1 84895 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84894 .coefficient)
      LeftAuthority84892.bound (LeftAuthority84892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84895 .coefficient)
      LeftAuthority84889.bound (LeftAuthority84889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84889.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84892.bound LeftAuthority84889.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84892.bound, LeftAuthority84889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84892.actual selector witness) * (LeftAuthority84889.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84896

namespace LeftBound84900
def owner : Owner := ⟨.program ⟨214⟩, ⟨14209⟩⟩
def transferEvent : Nat := 84900
def frameStart : Nat := 84867
def rule : BoundRule := .identity (.predecessor 0 84899 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84899 .coefficient)
      LeftBound84896.bound (LeftBound84896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84896.derived selector witness)

def rawBound : CoeffClass := LeftBound84896.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84896.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84900

namespace LeftBound84917
def owner : Owner := ⟨.program ⟨214⟩, ⟨14314⟩⟩
def transferEvent : Nat := 84917
def frameStart : Nat := 84867
def rule : BoundRule := .sum [.predecessor 0 84915 .coefficient, .predecessor 1 84916 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84915 .coefficient)
      LeftBound84900.bound (LeftBound84900.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84916 .coefficient)
      LeftAuthority84913.bound (LeftAuthority84913.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84900.bound, LeftAuthority84913.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84900.bound, LeftAuthority84913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84900.actual selector witness, LeftAuthority84913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84917

namespace LeftBound84920
def owner : Owner := ⟨.program ⟨214⟩, ⟨14315⟩⟩
def transferEvent : Nat := 84920
def frameStart : Nat := 84867
def rule : BoundRule := .identity (.predecessor 0 84919 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84919 .coefficient)
      LeftBound84917.bound (LeftBound84917.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84917.derived selector witness)

def rawBound : CoeffClass := LeftBound84917.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84917.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84920

namespace LeftBound84926
def owner : Owner := ⟨.program ⟨214⟩, ⟨14316⟩⟩
def transferEvent : Nat := 84926
def frameStart : Nat := 84867
def rule : BoundRule := .product (.predecessor 0 84924 .coefficient) (.predecessor 1 84925 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84924 .coefficient)
      LeftAuthority84922.bound (LeftAuthority84922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84925 .coefficient)
      LeftBound84920.bound (LeftBound84920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority84922.bound LeftBound84920.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84922.bound, LeftBound84920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority84922.actual selector witness) * (LeftBound84920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84926

namespace LeftBound84940
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 84940
def frameStart : Nat := 84867
def rule : BoundRule := .scale (.predecessor 0 84938 .coefficient) (.value (.predecessor 1 84939 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84938 .coefficient)
      LeftAuthority84936.bound (LeftAuthority84936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84939 .coefficient)
      LeftAuthority84870.bound (LeftAuthority84870.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84870.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84936.bound LeftAuthority84870.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84936.bound, LeftAuthority84870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84936.actual selector witness) * (LeftAuthority84870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84940

namespace LeftBound84943
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 84943
def frameStart : Nat := 84867
def rule : BoundRule := .identity (.predecessor 0 84942 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84942 .coefficient)
      LeftAuthority84930.bound (LeftAuthority84930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84930.derived selector witness)

def rawBound : CoeffClass := LeftAuthority84930.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority84930.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84943

namespace LeftBound84947
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 84947
def frameStart : Nat := 84867
def rule : BoundRule := .product (.predecessor 0 84945 .coefficient) (.predecessor 1 84946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84945 .coefficient)
      LeftBound84943.bound (LeftBound84943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84946 .coefficient)
      LeftBound84940.bound (LeftBound84940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84940.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84943.bound LeftBound84940.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84943.bound, LeftBound84940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84943.actual selector witness) * (LeftBound84940.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84947

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
