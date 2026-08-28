import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard545
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard608

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88635
def owner : Owner := ⟨.program ⟨214⟩, ⟨30120⟩⟩
def transferEvent : Nat := 88635
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88631 .summary, .result 80391 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88631 .summary)
      LeftBound88630.bound (LeftBound88630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29823⟩⟩) (rawTerms := some (Proof.Events346.exact88631RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80391 .summary)
      LeftBound80390.bound (LeftBound80390.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30119⟩⟩) (rawTerms := some (Proof.Events314.exact80391RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88630.bound, LeftBound80390.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88630.bound, LeftBound80390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88630.actual selector witness, LeftBound80390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88635

namespace LeftBound88639
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def transferEvent : Nat := 88639
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88637 .coefficient) (.predecessor 1 88638 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88637 .coefficient)
      LeftBound88634.bound (LeftBound88634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88638 .coefficient)
      LeftAuthority79894.bound (LeftAuthority79894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88634.bound LeftAuthority79894.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88634.bound, LeftAuthority79894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88634.actual selector witness) * (LeftAuthority79894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88639

namespace LeftBound88640
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def transferEvent : Nat := 88640
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩ [⟨.result 79895 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79895 .coefficient)
      LeftAuthority79894.bound (LeftAuthority79894.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18681⟩⟩) (rawTerms := some (Proof.Events312.exact79895RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79894.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79894.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79894.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88640

namespace LeftBound88641
def owner : Owner := ⟨.program ⟨214⟩, ⟨30121⟩⟩
def transferEvent : Nat := 88641
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88636 .summary) (.transfer 88640) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88636 .summary)
      LeftBound88635.bound (LeftBound88635.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30120⟩⟩) (rawTerms := some (Proof.Events346.exact88636RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88640)
      LeftBound88640.bound (LeftBound88640.actual selector witness) := by
  exact .transfer (LeftBound88640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88635.bound LeftBound88640.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88635.bound, LeftBound88640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88635.actual selector witness) * (LeftBound88640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88641

namespace LeftBound88720
def owner : Owner := ⟨.program ⟨214⟩, ⟨18561⟩⟩
def transferEvent : Nat := 88720
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 88718 .coefficient) (.value (.predecessor 1 88719 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88718 .coefficient)
      LeftAuthority88716.bound (LeftAuthority88716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88719 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority88716.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88716.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88716.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88720

namespace LeftBound88724
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def transferEvent : Nat := 88724
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88722 .coefficient) (.predecessor 1 88723 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88722 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88723 .coefficient)
      LeftBound88720.bound (LeftBound88720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound88720.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound88720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound88720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88724

namespace LeftBound88725
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def transferEvent : Nat := 88725
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩ [⟨.result 88717 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88717 .coefficient)
      LeftAuthority88716.bound (LeftAuthority88716.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18559⟩⟩) (rawTerms := some (Proof.Events346.exact88717RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88716.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority88716.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88716.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88725

namespace LeftBound88726
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def transferEvent : Nat := 88726
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 88725) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88725)
      LeftBound88725.bound (LeftBound88725.actual selector witness) := by
  exact .transfer (LeftBound88725.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound88725.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound88725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound88725.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88726

namespace LeftBound89754
def owner : Owner := ⟨.program ⟨214⟩, ⟨15311⟩⟩
def transferEvent : Nat := 89754
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89752 .coefficient, .predecessor 1 89753 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89752 .coefficient)
      LeftAuthority89750.bound (LeftAuthority89750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89753 .coefficient)
      LeftAuthority89727.bound (LeftAuthority89727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority89750.bound, LeftAuthority89727.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority89750.bound, LeftAuthority89727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority89750.actual selector witness, LeftAuthority89727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89754

namespace LeftBound89758
def owner : Owner := ⟨.program ⟨214⟩, ⟨15367⟩⟩
def transferEvent : Nat := 89758
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89756 .coefficient, .predecessor 1 89757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89756 .coefficient)
      LeftBound89754.bound (LeftBound89754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89757 .coefficient)
      LeftAuthority89704.bound (LeftAuthority89704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89754.bound, LeftAuthority89704.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89754.bound, LeftAuthority89704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89754.actual selector witness, LeftAuthority89704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89758

namespace LeftBound89762
def owner : Owner := ⟨.program ⟨214⟩, ⟨17328⟩⟩
def transferEvent : Nat := 89762
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89760 .coefficient, .predecessor 1 89761 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89760 .coefficient)
      LeftBound89758.bound (LeftBound89758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89761 .coefficient)
      LeftAuthority89681.bound (LeftAuthority89681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89758.bound, LeftAuthority89681.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89758.bound, LeftAuthority89681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89758.actual selector witness, LeftAuthority89681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89762

namespace LeftBound89766
def owner : Owner := ⟨.program ⟨214⟩, ⟨17329⟩⟩
def transferEvent : Nat := 89766
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89764 .coefficient, .predecessor 1 89765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89764 .coefficient)
      LeftBound89762.bound (LeftBound89762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89765 .coefficient)
      LeftAuthority89658.bound (LeftAuthority89658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89762.bound, LeftAuthority89658.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89762.bound, LeftAuthority89658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89762.actual selector witness, LeftAuthority89658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89766

namespace LeftBound89770
def owner : Owner := ⟨.program ⟨214⟩, ⟨17330⟩⟩
def transferEvent : Nat := 89770
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89768 .coefficient, .predecessor 1 89769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89768 .coefficient)
      LeftBound89766.bound (LeftBound89766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89769 .coefficient)
      LeftAuthority89635.bound (LeftAuthority89635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89766.bound, LeftAuthority89635.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89766.bound, LeftAuthority89635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89766.actual selector witness, LeftAuthority89635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89770

namespace LeftBound89774
def owner : Owner := ⟨.program ⟨214⟩, ⟨17331⟩⟩
def transferEvent : Nat := 89774
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89772 .coefficient, .predecessor 1 89773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89772 .coefficient)
      LeftBound89770.bound (LeftBound89770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89773 .coefficient)
      LeftAuthority89612.bound (LeftAuthority89612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89770.bound, LeftAuthority89612.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89770.bound, LeftAuthority89612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89770.actual selector witness, LeftAuthority89612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89774

namespace LeftBound89778
def owner : Owner := ⟨.program ⟨214⟩, ⟨17332⟩⟩
def transferEvent : Nat := 89778
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89776 .coefficient, .predecessor 1 89777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89776 .coefficient)
      LeftBound89774.bound (LeftBound89774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89777 .coefficient)
      LeftAuthority89589.bound (LeftAuthority89589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89589.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89774.bound, LeftAuthority89589.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89774.bound, LeftAuthority89589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89774.actual selector witness, LeftAuthority89589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89778

namespace LeftBound89782
def owner : Owner := ⟨.program ⟨214⟩, ⟨17333⟩⟩
def transferEvent : Nat := 89782
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89780 .coefficient, .predecessor 1 89781 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89780 .coefficient)
      LeftBound89778.bound (LeftBound89778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89781 .coefficient)
      LeftAuthority89566.bound (LeftAuthority89566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89778.bound, LeftAuthority89566.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89778.bound, LeftAuthority89566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89778.actual selector witness, LeftAuthority89566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89782

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
