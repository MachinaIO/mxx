import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard198
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard199

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29891
def owner : Owner := ⟨.program ⟨214⟩, ⟨24928⟩⟩
def transferEvent : Nat := 29891
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29885 .summary, .result 29699 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29885 .summary)
      LeftBound29711.bound (LeftBound29711.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19039⟩⟩) (rawTerms := some (Proof.Events116.exact29885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29699 .summary)
      LeftBound29694.bound (LeftBound29694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24927⟩⟩) (rawTerms := some (Proof.Events116.exact29699RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29711.bound, LeftBound29694.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29711.bound, LeftBound29694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29711.actual selector witness, LeftBound29694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29891

namespace LeftBound29895
def owner : Owner := ⟨.program ⟨214⟩, ⟨26396⟩⟩
def transferEvent : Nat := 29895
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29893 .coefficient) (.predecessor 1 29894 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29893 .coefficient)
      LeftBound29888.bound (LeftBound29888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29894 .coefficient)
      LeftAuthority29614.bound (LeftAuthority29614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29888.bound LeftAuthority29614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29888.bound, LeftAuthority29614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29888.actual selector witness) * (LeftAuthority29614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29895

namespace LeftBound29896
def owner : Owner := ⟨.program ⟨214⟩, ⟨26396⟩⟩
def transferEvent : Nat := 29896
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26394⟩⟩]⟩ [⟨.result 29615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29615 .coefficient)
      LeftAuthority29614.bound (LeftAuthority29614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26394⟩⟩) (rawTerms := some (Proof.Events115.exact29615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29896

namespace LeftBound29897
def owner : Owner := ⟨.program ⟨214⟩, ⟨26396⟩⟩
def transferEvent : Nat := 29897
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29892 .summary) (.transfer 29896) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29892 .summary)
      LeftBound29891.bound (LeftBound29891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24928⟩⟩) (rawTerms := some (Proof.Events116.exact29892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29896)
      LeftBound29896.bound (LeftBound29896.actual selector witness) := by
  exact .transfer (LeftBound29896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29891.bound LeftBound29896.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29891.bound, LeftBound29896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29891.actual selector witness) * (LeftBound29896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29897

namespace LeftBound29908
def owner : Owner := ⟨.program ⟨214⟩, ⟨20406⟩⟩
def transferEvent : Nat := 29908
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 29906 .coefficient) (.value (.predecessor 1 29907 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29906 .coefficient)
      LeftAuthority29904.bound (LeftAuthority29904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29907 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29904.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29904.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29904.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29908

namespace LeftBound29912
def owner : Owner := ⟨.program ⟨214⟩, ⟨20407⟩⟩
def transferEvent : Nat := 29912
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29910 .coefficient) (.predecessor 1 29911 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29910 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29911 .coefficient)
      LeftBound29908.bound (LeftBound29908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29908.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound29908.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound29908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound29908.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29912

namespace LeftBound29913
def owner : Owner := ⟨.program ⟨214⟩, ⟨20407⟩⟩
def transferEvent : Nat := 29913
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩ [⟨.result 29905 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29905 .coefficient)
      LeftAuthority29904.bound (LeftAuthority29904.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20404⟩⟩) (rawTerms := some (Proof.Events116.exact29905RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29904.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29904.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29904.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29913

namespace LeftBound29914
def owner : Owner := ⟨.program ⟨214⟩, ⟨20407⟩⟩
def transferEvent : Nat := 29914
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 29913) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29913)
      LeftBound29913.bound (LeftBound29913.actual selector witness) := by
  exact .transfer (LeftBound29913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound29913.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound29913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound29913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29914

namespace LeftBound30009
def owner : Owner := ⟨.program ⟨214⟩, ⟨14805⟩⟩
def transferEvent : Nat := 30009
def frameStart : Nat := 29970
def rule : BoundRule := .identity (.predecessor 0 30008 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30008 .coefficient)
      LeftAuthority30006.bound (LeftAuthority30006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30006.derived selector witness)

def rawBound : CoeffClass := LeftAuthority30006.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority30006.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound30009

namespace LeftBound30026
def owner : Owner := ⟨.program ⟨214⟩, ⟨14844⟩⟩
def transferEvent : Nat := 30026
def frameStart : Nat := 29970
def rule : BoundRule := .sum [.predecessor 0 30024 .coefficient, .predecessor 1 30025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30024 .coefficient)
      LeftBound30009.bound (LeftBound30009.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound30009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30025 .coefficient)
      LeftAuthority30022.bound (LeftAuthority30022.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority30022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30009.bound, LeftAuthority30022.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30009.bound, LeftAuthority30022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30009.actual selector witness, LeftAuthority30022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30026

namespace LeftBound30029
def owner : Owner := ⟨.program ⟨214⟩, ⟨14845⟩⟩
def transferEvent : Nat := 30029
def frameStart : Nat := 29970
def rule : BoundRule := .identity (.predecessor 0 30028 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30028 .coefficient)
      LeftBound30026.bound (LeftBound30026.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound30026.derived selector witness)

def rawBound : CoeffClass := LeftBound30026.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound30026.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound30029

namespace LeftBound30035
def owner : Owner := ⟨.program ⟨214⟩, ⟨14846⟩⟩
def transferEvent : Nat := 30035
def frameStart : Nat := 29970
def rule : BoundRule := .product (.predecessor 0 30033 .coefficient) (.predecessor 1 30034 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30033 .coefficient)
      LeftAuthority30031.bound (LeftAuthority30031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30034 .coefficient)
      LeftBound30029.bound (LeftBound30029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority30031.bound LeftBound30029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30031.bound, LeftBound30029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority30031.actual selector witness) * (LeftBound30029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30035

namespace LeftBound30043
def owner : Owner := ⟨.program ⟨214⟩, ⟨14847⟩⟩
def transferEvent : Nat := 30043
def frameStart : Nat := 29970
def rule : BoundRule := .sum [.predecessor 0 30041 .coefficient, .predecessor 1 30042 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30041 .coefficient)
      LeftAuthority30039.bound (LeftAuthority30039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30042 .coefficient)
      LeftBound30035.bound (LeftBound30035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority30039.bound, LeftBound30035.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30039.bound, LeftBound30035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority30039.actual selector witness, LeftBound30035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30043

namespace LeftBound30047
def owner : Owner := ⟨.program ⟨214⟩, ⟨26395⟩⟩
def transferEvent : Nat := 30047
def frameStart : Nat := 29970
def rule : BoundRule := .product (.predecessor 0 30045 .coefficient) (.predecessor 1 30046 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30045 .coefficient)
      LeftBound30043.bound (LeftBound30043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30046 .coefficient)
      LeftAuthority30020.bound (LeftAuthority30020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30020.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound30043.bound LeftAuthority30020.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30043.bound, LeftAuthority30020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound30043.actual selector witness) * (LeftAuthority30020.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30047

namespace LeftBound30058
def owner : Owner := ⟨.program ⟨214⟩, ⟨15275⟩⟩
def transferEvent : Nat := 30058
def frameStart : Nat := 29970
def rule : BoundRule := .product (.predecessor 0 30056 .coefficient) (.predecessor 1 30057 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30056 .coefficient)
      LeftAuthority30031.bound (LeftAuthority30031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30057 .coefficient)
      LeftAuthority30054.bound (LeftAuthority30054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30054.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority30031.bound LeftAuthority30054.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30031.bound, LeftAuthority30054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority30031.actual selector witness) * (LeftAuthority30054.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound30058

namespace LeftBound30066
def owner : Owner := ⟨.program ⟨214⟩, ⟨15276⟩⟩
def transferEvent : Nat := 30066
def frameStart : Nat := 29970
def rule : BoundRule := .sum [.predecessor 0 30064 .coefficient, .predecessor 1 30065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30064 .coefficient)
      LeftAuthority30062.bound (LeftAuthority30062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30062.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30065 .coefficient)
      LeftBound30058.bound (LeftBound30058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority30062.bound, LeftBound30058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority30062.bound, LeftBound30058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority30062.actual selector witness, LeftBound30058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30066

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
