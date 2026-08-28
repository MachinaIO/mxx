import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard169

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25854
def owner : Owner := ⟨.program ⟨214⟩, ⟨19615⟩⟩
def transferEvent : Nat := 25854
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19612⟩⟩]⟩ [⟨.result 25846 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25846 .coefficient)
      LeftAuthority25845.bound (LeftAuthority25845.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19612⟩⟩) (rawTerms := some (Proof.Events100.exact25846RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25845.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25845.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25845.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25854

namespace LeftBound25855
def owner : Owner := ⟨.program ⟨214⟩, ⟨19615⟩⟩
def transferEvent : Nat := 25855
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 25854) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25854)
      LeftBound25854.bound (LeftBound25854.actual selector witness) := by
  exact .transfer (LeftBound25854.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound25854.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound25854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound25854.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25855

namespace LeftBound25934
def owner : Owner := ⟨.program ⟨214⟩, ⟨14452⟩⟩
def transferEvent : Nat := 25934
def frameStart : Nat := 25905
def rule : BoundRule := .product (.predecessor 0 25932 .coefficient) (.predecessor 1 25933 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25932 .coefficient)
      LeftAuthority25930.bound (LeftAuthority25930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25930.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25933 .coefficient)
      LeftAuthority25927.bound (LeftAuthority25927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25927.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25930.bound LeftAuthority25927.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25930.bound, LeftAuthority25927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25930.actual selector witness) * (LeftAuthority25927.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25934

namespace LeftBound25938
def owner : Owner := ⟨.program ⟨214⟩, ⟨14453⟩⟩
def transferEvent : Nat := 25938
def frameStart : Nat := 25905
def rule : BoundRule := .identity (.predecessor 0 25937 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25937 .coefficient)
      LeftBound25934.bound (LeftBound25934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25934.derived selector witness)

def rawBound : CoeffClass := LeftBound25934.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25934.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25938

namespace LeftBound25955
def owner : Owner := ⟨.program ⟨214⟩, ⟨14543⟩⟩
def transferEvent : Nat := 25955
def frameStart : Nat := 25905
def rule : BoundRule := .sum [.predecessor 0 25953 .coefficient, .predecessor 1 25954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25953 .coefficient)
      LeftBound25938.bound (LeftBound25938.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25954 .coefficient)
      LeftAuthority25951.bound (LeftAuthority25951.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25938.bound, LeftAuthority25951.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25938.bound, LeftAuthority25951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25938.actual selector witness, LeftAuthority25951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25955

namespace LeftBound25958
def owner : Owner := ⟨.program ⟨214⟩, ⟨14544⟩⟩
def transferEvent : Nat := 25958
def frameStart : Nat := 25905
def rule : BoundRule := .identity (.predecessor 0 25957 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25957 .coefficient)
      LeftBound25955.bound (LeftBound25955.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25955.derived selector witness)

def rawBound : CoeffClass := LeftBound25955.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25955.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25958

namespace LeftBound25964
def owner : Owner := ⟨.program ⟨214⟩, ⟨14545⟩⟩
def transferEvent : Nat := 25964
def frameStart : Nat := 25905
def rule : BoundRule := .product (.predecessor 0 25962 .coefficient) (.predecessor 1 25963 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25962 .coefficient)
      LeftAuthority25960.bound (LeftAuthority25960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25963 .coefficient)
      LeftBound25958.bound (LeftBound25958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25958.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority25960.bound LeftBound25958.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25960.bound, LeftBound25958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority25960.actual selector witness) * (LeftBound25958.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25964

namespace LeftBound25980
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 25980
def frameStart : Nat := 25905
def rule : BoundRule := .scale (.predecessor 0 25978 .coefficient) (.value (.predecessor 1 25979 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25978 .coefficient)
      LeftAuthority25976.bound (LeftAuthority25976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25979 .coefficient)
      LeftAuthority25967.bound (LeftAuthority25967.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25967.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25976.bound LeftAuthority25967.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25976.bound, LeftAuthority25967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25976.actual selector witness) * (LeftAuthority25967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25980

namespace LeftBound25983
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 25983
def frameStart : Nat := 25905
def rule : BoundRule := .identity (.predecessor 0 25982 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25982 .coefficient)
      LeftAuthority25970.bound (LeftAuthority25970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25970.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25970.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority25970.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25983

namespace LeftBound25987
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 25987
def frameStart : Nat := 25905
def rule : BoundRule := .product (.predecessor 0 25985 .coefficient) (.predecessor 1 25986 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25985 .coefficient)
      LeftBound25983.bound (LeftBound25983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25986 .coefficient)
      LeftBound25980.bound (LeftBound25980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25983.bound LeftBound25980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25983.bound, LeftBound25980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25983.actual selector witness) * (LeftBound25980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25987

namespace LeftBound25992
def owner : Owner := ⟨.program ⟨214⟩, ⟨14546⟩⟩
def transferEvent : Nat := 25992
def frameStart : Nat := 25905
def rule : BoundRule := .sum [.predecessor 0 25990 .coefficient, .predecessor 1 25991 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25990 .coefficient)
      LeftBound25987.bound (LeftBound25987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25991 .coefficient)
      LeftBound25964.bound (LeftBound25964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25987.bound, LeftBound25964.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25987.bound, LeftBound25964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25987.actual selector witness, LeftBound25964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25992

namespace LeftBound25996
def owner : Owner := ⟨.program ⟨214⟩, ⟨26161⟩⟩
def transferEvent : Nat := 25996
def frameStart : Nat := 25905
def rule : BoundRule := .product (.predecessor 0 25994 .coefficient) (.predecessor 1 25995 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25994 .coefficient)
      LeftBound25992.bound (LeftBound25992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25995 .coefficient)
      LeftAuthority25949.bound (LeftAuthority25949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25949.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25949.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25992.bound LeftAuthority25949.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25992.bound, LeftAuthority25949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25992.actual selector witness) * (LeftAuthority25949.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25996

namespace LeftBound26007
def owner : Owner := ⟨.program ⟨214⟩, ⟨16073⟩⟩
def transferEvent : Nat := 26007
def frameStart : Nat := 25905
def rule : BoundRule := .product (.predecessor 0 26005 .coefficient) (.predecessor 1 26006 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26005 .coefficient)
      LeftAuthority25960.bound (LeftAuthority25960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact25961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26006 .coefficient)
      LeftAuthority26003.bound (LeftAuthority26003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25960.bound LeftAuthority26003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25960.bound, LeftAuthority26003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25960.actual selector witness) * (LeftAuthority26003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26007

namespace LeftBound26015
def owner : Owner := ⟨.program ⟨214⟩, ⟨16074⟩⟩
def transferEvent : Nat := 26015
def frameStart : Nat := 25905
def rule : BoundRule := .sum [.predecessor 0 26013 .coefficient, .predecessor 1 26014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26013 .coefficient)
      LeftAuthority26011.bound (LeftAuthority26011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26014 .coefficient)
      LeftBound26007.bound (LeftBound26007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26011.bound, LeftBound26007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26011.bound, LeftBound26007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26011.actual selector witness, LeftBound26007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26015

namespace LeftBound26019
def owner : Owner := ⟨.program ⟨214⟩, ⟨26162⟩⟩
def transferEvent : Nat := 26019
def frameStart : Nat := 25905
def rule : BoundRule := .sum [.predecessor 0 26017 .coefficient, .predecessor 1 26018 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26017 .coefficient)
      LeftBound26015.bound (LeftBound26015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26018 .coefficient)
      LeftBound25996.bound (LeftBound25996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26015.bound, LeftBound25996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26015.bound, LeftBound25996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26015.actual selector witness, LeftBound25996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26019

namespace LeftBound26032
def owner : Owner := ⟨.program ⟨214⟩, ⟨26160⟩⟩
def transferEvent : Nat := 26032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26030 .coefficient, .predecessor 1 26031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26030 .coefficient)
      LeftBound25853.bound (LeftBound25853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26031 .coefficient)
      LeftBound25836.bound (LeftBound25836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25853.bound, LeftBound25836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25853.bound, LeftBound25836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25853.actual selector witness, LeftBound25836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26032

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
