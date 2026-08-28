import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65190
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65190
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65188, .transfer 65189]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65188)
      LeftBound65188.bound (LeftBound65188.actual selector witness) := by
  exact .transfer (LeftBound65188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65189)
      LeftBound65189.bound (LeftBound65189.actual selector witness) := by
  exact .transfer (LeftBound65189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65188.bound, LeftBound65189.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65188.bound, LeftBound65189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65188.actual selector witness, LeftBound65189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65190

namespace LeftBound65191
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65191
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17714⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 3603 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6459⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3603 .coefficient)
      LeftAuthority3602.bound (LeftAuthority3602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17714⟩⟩) (rawTerms := some (Proof.Events014.exact3603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3602.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority3602.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority3602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority3602.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65191

namespace LeftBound65192
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65190, .transfer 65191]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65190)
      LeftBound65190.bound (LeftBound65190.actual selector witness) := by
  exact .transfer (LeftBound65190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65191)
      LeftBound65191.bound (LeftBound65191.actual selector witness) := by
  exact .transfer (LeftBound65191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65190.bound, LeftBound65191.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65190.bound, LeftBound65191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65190.actual selector witness, LeftBound65191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65192

namespace LeftBound65193
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65193
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17945⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 3611 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6467⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3611 .coefficient)
      LeftAuthority3610.bound (LeftAuthority3610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17945⟩⟩) (rawTerms := some (Proof.Events014.exact3611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority3610.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority3610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority3610.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65193

namespace LeftBound65194
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65192, .transfer 65193]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65192)
      LeftBound65192.bound (LeftBound65192.actual selector witness) := by
  exact .transfer (LeftBound65192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65193)
      LeftBound65193.bound (LeftBound65193.actual selector witness) := by
  exact .transfer (LeftBound65193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65192.bound, LeftBound65193.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65192.bound, LeftBound65193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65192.actual selector witness, LeftBound65193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65194

namespace LeftBound65195
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65195
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17546⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 3619 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6473⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3619 .coefficient)
      LeftAuthority3618.bound (LeftAuthority3618.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17546⟩⟩) (rawTerms := some (Proof.Events014.exact3619RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3618.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority3618.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority3618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority3618.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65195

namespace LeftBound65196
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65194, .transfer 65195]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65194)
      LeftBound65194.bound (LeftBound65194.actual selector witness) := by
  exact .transfer (LeftBound65194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65195)
      LeftBound65195.bound (LeftBound65195.actual selector witness) := by
  exact .transfer (LeftBound65195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65194.bound, LeftBound65195.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65194.bound, LeftBound65195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65194.actual selector witness, LeftBound65195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65196

namespace LeftBound65197
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65197
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6490⟩⟩, ⟨.program ⟨214⟩, ⟨18818⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 3627 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 603 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6490⟩⟩) (rawTerms := some (Proof.Events002.exact603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3627 .coefficient)
      LeftAuthority3626.bound (LeftAuthority3626.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18818⟩⟩) (rawTerms := some (Proof.Events014.exact3627RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3626.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority3626.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority3626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority3626.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65197

namespace LeftBound65198
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65196, .transfer 65197]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65196)
      LeftBound65196.bound (LeftBound65196.actual selector witness) := by
  exact .transfer (LeftBound65196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65197)
      LeftBound65197.bound (LeftBound65197.actual selector witness) := by
  exact .transfer (LeftBound65197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65196.bound, LeftBound65197.bound]
def bound : CoeffClass := .finite ⟨5829664127815216198670160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65196.bound, LeftBound65197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65196.actual selector witness, LeftBound65197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65198

namespace LeftBound65199
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65199
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6494⟩⟩, ⟨.program ⟨214⟩, ⟨17602⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 3635 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 613 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6494⟩⟩) (rawTerms := some (Proof.Events002.exact613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3635 .coefficient)
      LeftAuthority3634.bound (LeftAuthority3634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17602⟩⟩) (rawTerms := some (Proof.Events014.exact3635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority3634.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority3634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority3634.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65199

namespace LeftBound65200
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65200
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65198, .transfer 65199]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65198)
      LeftBound65198.bound (LeftBound65198.actual selector witness) := by
  exact .transfer (LeftBound65198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65199)
      LeftBound65199.bound (LeftBound65199.actual selector witness) := by
  exact .transfer (LeftBound65199.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65198.bound, LeftBound65199.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65198.bound, LeftBound65199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65198.actual selector witness, LeftBound65199.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65200

namespace LeftBound65201
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65201
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6502⟩⟩, ⟨.program ⟨214⟩, ⟨17658⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 3643 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 623 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6502⟩⟩) (rawTerms := some (Proof.Events002.exact623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3643 .coefficient)
      LeftAuthority3642.bound (LeftAuthority3642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17658⟩⟩) (rawTerms := some (Proof.Events014.exact3643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3642.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority3642.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority3642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority3642.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65201

namespace LeftBound65202
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65202
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65200, .transfer 65201]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65200)
      LeftBound65200.bound (LeftBound65200.actual selector witness) := by
  exact .transfer (LeftBound65200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65201)
      LeftBound65201.bound (LeftBound65201.actual selector witness) := by
  exact .transfer (LeftBound65201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65200.bound, LeftBound65201.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65200.bound, LeftBound65201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65200.actual selector witness, LeftBound65201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65202

namespace LeftBound65203
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65203
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6383⟩⟩, ⟨.program ⟨214⟩, ⟨18028⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 3651 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 633 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6383⟩⟩) (rawTerms := some (Proof.Events002.exact633RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3651 .coefficient)
      LeftAuthority3650.bound (LeftAuthority3650.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18028⟩⟩) (rawTerms := some (Proof.Events014.exact3651RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3650.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority3650.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority3650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority3650.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65203

namespace LeftBound65204
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 65202, .transfer 65203]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65202)
      LeftBound65202.bound (LeftBound65202.actual selector witness) := by
  exact .transfer (LeftBound65202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65203)
      LeftBound65203.bound (LeftBound65203.actual selector witness) := by
  exact .transfer (LeftBound65203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65202.bound, LeftBound65203.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65202.bound, LeftBound65203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65202.actual selector witness, LeftBound65203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65204

namespace LeftBound65205
def owner : Owner := ⟨.program ⟨214⟩, ⟨18830⟩⟩
def transferEvent : Nat := 65205
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6387⟩⟩, ⟨.program ⟨214⟩, ⟨17161⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 3659 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 643 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6387⟩⟩) (rawTerms := some (Proof.Events002.exact643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3659 .coefficient)
      LeftAuthority3658.bound (LeftAuthority3658.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17161⟩⟩) (rawTerms := some (Proof.Events014.exact3659RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3658.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority3658.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority3658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority3658.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound65205

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
