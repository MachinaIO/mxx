import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard147
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard148

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23115
def owner : Owner := ⟨.program ⟨214⟩, ⟨16647⟩⟩
def transferEvent : Nat := 23115
def frameStart : Nat := 23013
def rule : BoundRule := .product (.predecessor 0 23113 .coefficient) (.predecessor 1 23114 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23113 .coefficient)
      LeftAuthority23068.bound (LeftAuthority23068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23114 .coefficient)
      LeftAuthority23111.bound (LeftAuthority23111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23068.bound LeftAuthority23111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23068.bound, LeftAuthority23111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23068.actual selector witness) * (LeftAuthority23111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23115

namespace LeftBound23123
def owner : Owner := ⟨.program ⟨214⟩, ⟨16648⟩⟩
def transferEvent : Nat := 23123
def frameStart : Nat := 23013
def rule : BoundRule := .sum [.predecessor 0 23121 .coefficient, .predecessor 1 23122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23121 .coefficient)
      LeftAuthority23119.bound (LeftAuthority23119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23122 .coefficient)
      LeftBound23115.bound (LeftBound23115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23119.bound, LeftBound23115.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23119.bound, LeftBound23115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23119.actual selector witness, LeftBound23115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23123

namespace LeftBound23127
def owner : Owner := ⟨.program ⟨214⟩, ⟨25546⟩⟩
def transferEvent : Nat := 23127
def frameStart : Nat := 23013
def rule : BoundRule := .sum [.predecessor 0 23125 .coefficient, .predecessor 1 23126 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23125 .coefficient)
      LeftBound23123.bound (LeftBound23123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23126 .coefficient)
      LeftBound23104.bound (LeftBound23104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23123.bound, LeftBound23104.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23123.bound, LeftBound23104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23123.actual selector witness, LeftBound23104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23127

namespace LeftBound23140
def owner : Owner := ⟨.program ⟨214⟩, ⟨25544⟩⟩
def transferEvent : Nat := 23140
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23138 .coefficient, .predecessor 1 23139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23138 .coefficient)
      LeftBound22961.bound (LeftBound22961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23139 .coefficient)
      LeftBound22944.bound (LeftBound22944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22961.bound, LeftBound22944.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22961.bound, LeftBound22944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22961.actual selector witness, LeftBound22944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23140

namespace LeftBound23143
def owner : Owner := ⟨.program ⟨214⟩, ⟨25544⟩⟩
def transferEvent : Nat := 23143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 23137 .summary, .result 22951 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23137 .summary)
      LeftBound22963.bound (LeftBound22963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20047⟩⟩) (rawTerms := some (Proof.Events090.exact23137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22951 .summary)
      LeftBound22946.bound (LeftBound22946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25543⟩⟩) (rawTerms := some (Proof.Events089.exact22951RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22963.bound, LeftBound22946.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22963.bound, LeftBound22946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22963.actual selector witness, LeftBound22946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23143

namespace LeftBound23147
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def transferEvent : Nat := 23147
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23145 .coefficient) (.predecessor 1 23146 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23145 .coefficient)
      LeftBound23140.bound (LeftBound23140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23146 .coefficient)
      LeftAuthority22866.bound (LeftAuthority22866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23140.bound LeftAuthority22866.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23140.bound, LeftAuthority22866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23140.actual selector witness) * (LeftAuthority22866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23147

namespace LeftBound23148
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def transferEvent : Nat := 23148
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29424⟩⟩]⟩ [⟨.result 22867 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22867 .coefficient)
      LeftAuthority22866.bound (LeftAuthority22866.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29424⟩⟩) (rawTerms := some (Proof.Events089.exact22867RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22866.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22866.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22866.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23148

namespace LeftBound23149
def owner : Owner := ⟨.program ⟨214⟩, ⟨29426⟩⟩
def transferEvent : Nat := 23149
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23144 .summary) (.transfer 23148) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23144 .summary)
      LeftBound23143.bound (LeftBound23143.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25544⟩⟩) (rawTerms := some (Proof.Events090.exact23144RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23148)
      LeftBound23148.bound (LeftBound23148.actual selector witness) := by
  exact .transfer (LeftBound23148.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23143.bound LeftBound23148.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23143.bound, LeftBound23148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23143.actual selector witness) * (LeftBound23148.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23149

namespace LeftBound23160
def owner : Owner := ⟨.program ⟨214⟩, ⟨22422⟩⟩
def transferEvent : Nat := 23160
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 23158 .coefficient) (.value (.predecessor 1 23159 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23158 .coefficient)
      LeftAuthority23156.bound (LeftAuthority23156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23159 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23156.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23156.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23156.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23160

namespace LeftBound23164
def owner : Owner := ⟨.program ⟨214⟩, ⟨22423⟩⟩
def transferEvent : Nat := 23164
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23162 .coefficient) (.predecessor 1 23163 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23162 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23163 .coefficient)
      LeftBound23160.bound (LeftBound23160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound23160.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound23160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound23160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23164

namespace LeftBound23165
def owner : Owner := ⟨.program ⟨214⟩, ⟨22423⟩⟩
def transferEvent : Nat := 23165
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22420⟩⟩]⟩ [⟨.result 23157 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23157 .coefficient)
      LeftAuthority23156.bound (LeftAuthority23156.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22420⟩⟩) (rawTerms := some (Proof.Events090.exact23157RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23156.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23156.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23156.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23165

namespace LeftBound23166
def owner : Owner := ⟨.program ⟨214⟩, ⟨22423⟩⟩
def transferEvent : Nat := 23166
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 23165) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23165)
      LeftBound23165.bound (LeftBound23165.actual selector witness) := by
  exact .transfer (LeftBound23165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound23165.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound23165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound23165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23166

namespace LeftBound23261
def owner : Owner := ⟨.program ⟨214⟩, ⟨16646⟩⟩
def transferEvent : Nat := 23261
def frameStart : Nat := 23222
def rule : BoundRule := .identity (.predecessor 0 23260 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23260 .coefficient)
      LeftAuthority23258.bound (LeftAuthority23258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23258.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23258.derived selector witness)

def rawBound : CoeffClass := LeftAuthority23258.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority23258.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23261

namespace LeftBound23278
def owner : Owner := ⟨.program ⟨214⟩, ⟨16720⟩⟩
def transferEvent : Nat := 23278
def frameStart : Nat := 23222
def rule : BoundRule := .sum [.predecessor 0 23276 .coefficient, .predecessor 1 23277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23276 .coefficient)
      LeftBound23261.bound (LeftBound23261.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23277 .coefficient)
      LeftAuthority23274.bound (LeftAuthority23274.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23261.bound, LeftAuthority23274.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23261.bound, LeftAuthority23274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23261.actual selector witness, LeftAuthority23274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23278

namespace LeftBound23281
def owner : Owner := ⟨.program ⟨214⟩, ⟨16721⟩⟩
def transferEvent : Nat := 23281
def frameStart : Nat := 23222
def rule : BoundRule := .identity (.predecessor 0 23280 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23280 .coefficient)
      LeftBound23278.bound (LeftBound23278.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23278.derived selector witness)

def rawBound : CoeffClass := LeftBound23278.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23278.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23281

namespace LeftBound23287
def owner : Owner := ⟨.program ⟨214⟩, ⟨16722⟩⟩
def transferEvent : Nat := 23287
def frameStart : Nat := 23222
def rule : BoundRule := .product (.predecessor 0 23285 .coefficient) (.predecessor 1 23286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23285 .coefficient)
      LeftAuthority23283.bound (LeftAuthority23283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23286 .coefficient)
      LeftBound23281.bound (LeftBound23281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23281.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority23283.bound LeftBound23281.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23283.bound, LeftBound23281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority23283.actual selector witness) * (LeftBound23281.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23287

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
