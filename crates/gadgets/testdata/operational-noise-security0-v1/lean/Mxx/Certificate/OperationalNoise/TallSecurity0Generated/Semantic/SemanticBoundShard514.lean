import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard513

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76044
def owner : Owner := ⟨.program ⟨214⟩, ⟨22479⟩⟩
def transferEvent : Nat := 76044
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22476⟩⟩]⟩ [⟨.result 76036 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76036 .coefficient)
      LeftAuthority76035.bound (LeftAuthority76035.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22476⟩⟩) (rawTerms := some (Proof.Events297.exact76036RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76035.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76035.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76035.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76044

namespace LeftBound76045
def owner : Owner := ⟨.program ⟨214⟩, ⟨22479⟩⟩
def transferEvent : Nat := 76045
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 76044) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76044)
      LeftBound76044.bound (LeftBound76044.actual selector witness) := by
  exact .transfer (LeftBound76044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound76044.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound76044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound76044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76045

namespace LeftBound76140
def owner : Owner := ⟨.program ⟨214⟩, ⟨16749⟩⟩
def transferEvent : Nat := 76140
def frameStart : Nat := 76101
def rule : BoundRule := .identity (.predecessor 0 76139 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76139 .coefficient)
      LeftAuthority76137.bound (LeftAuthority76137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76137.derived selector witness)

def rawBound : CoeffClass := LeftAuthority76137.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority76137.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76140

namespace LeftBound76157
def owner : Owner := ⟨.program ⟨214⟩, ⟨16823⟩⟩
def transferEvent : Nat := 76157
def frameStart : Nat := 76101
def rule : BoundRule := .sum [.predecessor 0 76155 .coefficient, .predecessor 1 76156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76155 .coefficient)
      LeftBound76140.bound (LeftBound76140.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76156 .coefficient)
      LeftAuthority76153.bound (LeftAuthority76153.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76153.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76140.bound, LeftAuthority76153.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76140.bound, LeftAuthority76153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76140.actual selector witness, LeftAuthority76153.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76157

namespace LeftBound76160
def owner : Owner := ⟨.program ⟨214⟩, ⟨16824⟩⟩
def transferEvent : Nat := 76160
def frameStart : Nat := 76101
def rule : BoundRule := .identity (.predecessor 0 76159 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76159 .coefficient)
      LeftBound76157.bound (LeftBound76157.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76157.derived selector witness)

def rawBound : CoeffClass := LeftBound76157.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound76157.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76160

namespace LeftBound76166
def owner : Owner := ⟨.program ⟨214⟩, ⟨16825⟩⟩
def transferEvent : Nat := 76166
def frameStart : Nat := 76101
def rule : BoundRule := .product (.predecessor 0 76164 .coefficient) (.predecessor 1 76165 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76164 .coefficient)
      LeftAuthority76162.bound (LeftAuthority76162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76165 .coefficient)
      LeftBound76160.bound (LeftBound76160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority76162.bound LeftBound76160.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76162.bound, LeftBound76160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority76162.actual selector witness) * (LeftBound76160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76166

namespace LeftBound76174
def owner : Owner := ⟨.program ⟨214⟩, ⟨16826⟩⟩
def transferEvent : Nat := 76174
def frameStart : Nat := 76101
def rule : BoundRule := .sum [.predecessor 0 76172 .coefficient, .predecessor 1 76173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76172 .coefficient)
      LeftAuthority76170.bound (LeftAuthority76170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76173 .coefficient)
      LeftBound76166.bound (LeftBound76166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76170.bound, LeftBound76166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76170.bound, LeftBound76166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76170.actual selector witness, LeftBound76166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76174

namespace LeftBound76178
def owner : Owner := ⟨.program ⟨214⟩, ⟨29583⟩⟩
def transferEvent : Nat := 76178
def frameStart : Nat := 76101
def rule : BoundRule := .product (.predecessor 0 76176 .coefficient) (.predecessor 1 76177 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76176 .coefficient)
      LeftBound76174.bound (LeftBound76174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76177 .coefficient)
      LeftAuthority76151.bound (LeftAuthority76151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76151.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76174.bound LeftAuthority76151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76174.bound, LeftAuthority76151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76174.actual selector witness) * (LeftAuthority76151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76178

namespace LeftBound76189
def owner : Owner := ⟨.program ⟨214⟩, ⟨17492⟩⟩
def transferEvent : Nat := 76189
def frameStart : Nat := 76101
def rule : BoundRule := .product (.predecessor 0 76187 .coefficient) (.predecessor 1 76188 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76187 .coefficient)
      LeftAuthority76162.bound (LeftAuthority76162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76188 .coefficient)
      LeftAuthority76185.bound (LeftAuthority76185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76185.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76162.bound LeftAuthority76185.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76162.bound, LeftAuthority76185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority76162.actual selector witness) * (LeftAuthority76185.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76189

namespace LeftBound76197
def owner : Owner := ⟨.program ⟨214⟩, ⟨17493⟩⟩
def transferEvent : Nat := 76197
def frameStart : Nat := 76101
def rule : BoundRule := .sum [.predecessor 0 76195 .coefficient, .predecessor 1 76196 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76195 .coefficient)
      LeftAuthority76193.bound (LeftAuthority76193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76196 .coefficient)
      LeftBound76189.bound (LeftBound76189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76193.bound, LeftBound76189.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76193.bound, LeftBound76189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76193.actual selector witness, LeftBound76189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76197

namespace LeftBound76201
def owner : Owner := ⟨.program ⟨214⟩, ⟨29588⟩⟩
def transferEvent : Nat := 76201
def frameStart : Nat := 76101
def rule : BoundRule := .sum [.predecessor 0 76199 .coefficient, .predecessor 1 76200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76199 .coefficient)
      LeftBound76197.bound (LeftBound76197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76200 .coefficient)
      LeftBound76178.bound (LeftBound76178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76197.bound, LeftBound76178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76197.bound, LeftBound76178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76197.actual selector witness, LeftBound76178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76201

namespace LeftBound76214
def owner : Owner := ⟨.program ⟨214⟩, ⟨29585⟩⟩
def transferEvent : Nat := 76214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76212 .coefficient, .predecessor 1 76213 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76212 .coefficient)
      LeftBound76043.bound (LeftBound76043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76213 .coefficient)
      LeftBound76026.bound (LeftBound76026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76043.bound, LeftBound76026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76043.bound, LeftBound76026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76043.actual selector witness, LeftBound76026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76214

namespace LeftBound76217
def owner : Owner := ⟨.program ⟨214⟩, ⟨29585⟩⟩
def transferEvent : Nat := 76217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 76211 .summary, .result 76033 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76211 .summary)
      LeftBound76045.bound (LeftBound76045.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22479⟩⟩) (rawTerms := some (Proof.Events297.exact76211RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76033 .summary)
      LeftBound76028.bound (LeftBound76028.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29584⟩⟩) (rawTerms := some (Proof.Events297.exact76033RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76028.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76045.bound, LeftBound76028.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76045.bound, LeftBound76028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76045.actual selector witness, LeftBound76028.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76217

namespace LeftBound76221
def owner : Owner := ⟨.program ⟨214⟩, ⟨29586⟩⟩
def transferEvent : Nat := 76221
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76219 .coefficient) (.predecessor 1 76220 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76219 .coefficient)
      LeftBound76214.bound (LeftBound76214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76220 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76214.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76214.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76214.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76221

namespace LeftBound76222
def owner : Owner := ⟨.program ⟨214⟩, ⟨29586⟩⟩
def transferEvent : Nat := 76222
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76222

namespace LeftBound76223
def owner : Owner := ⟨.program ⟨214⟩, ⟨29586⟩⟩
def transferEvent : Nat := 76223
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76218 .summary) (.transfer 76222) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76218 .summary)
      LeftBound76217.bound (LeftBound76217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29585⟩⟩) (rawTerms := some (Proof.Events297.exact76218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76222)
      LeftBound76222.bound (LeftBound76222.actual selector witness) := by
  exact .transfer (LeftBound76222.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76217.bound LeftBound76222.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76217.bound, LeftBound76222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76217.actual selector witness) * (LeftBound76222.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76223

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
