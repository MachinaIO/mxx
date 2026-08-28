import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard171
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard173
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard180

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound32943
def owner : Owner := ⟨.program ⟨257⟩, ⟨47575⟩⟩
def transferEvent : Nat := 32943
def frameStart : Nat := 32866
def rule : BoundRule := .product (.predecessor 0 32941 .coefficient) (.predecessor 1 32942 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32941 .coefficient)
      LeftBound32939.bound (LeftBound32939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32942 .coefficient)
      LeftAuthority32916.bound (LeftAuthority32916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32916.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound32939.bound LeftAuthority32916.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32939.bound, LeftAuthority32916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound32939.actual selector witness) * (LeftAuthority32916.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32943

namespace LeftBound32954
def owner : Owner := ⟨.program ⟨257⟩, ⟨45801⟩⟩
def transferEvent : Nat := 32954
def frameStart : Nat := 32866
def rule : BoundRule := .product (.predecessor 0 32952 .coefficient) (.predecessor 1 32953 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32952 .coefficient)
      LeftAuthority32927.bound (LeftAuthority32927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32927.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32953 .coefficient)
      LeftAuthority32950.bound (LeftAuthority32950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32950.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32927.bound LeftAuthority32950.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32927.bound, LeftAuthority32950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority32927.actual selector witness) * (LeftAuthority32950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32954

namespace LeftBound32962
def owner : Owner := ⟨.program ⟨257⟩, ⟨45802⟩⟩
def transferEvent : Nat := 32962
def frameStart : Nat := 32866
def rule : BoundRule := .sum [.predecessor 0 32960 .coefficient, .predecessor 1 32961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32960 .coefficient)
      LeftAuthority32958.bound (LeftAuthority32958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32958.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32961 .coefficient)
      LeftBound32954.bound (LeftBound32954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32958.bound, LeftBound32954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32958.bound, LeftBound32954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority32958.actual selector witness, LeftBound32954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32962

namespace LeftBound32966
def owner : Owner := ⟨.program ⟨257⟩, ⟨47578⟩⟩
def transferEvent : Nat := 32966
def frameStart : Nat := 32866
def rule : BoundRule := .sum [.predecessor 0 32964 .coefficient, .predecessor 1 32965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32964 .coefficient)
      LeftBound32962.bound (LeftBound32962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32965 .coefficient)
      LeftBound32943.bound (LeftBound32943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32962.bound, LeftBound32943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32962.bound, LeftBound32943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32962.actual selector witness, LeftBound32943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32966

namespace LeftBound32979
def owner : Owner := ⟨.program ⟨257⟩, ⟨47577⟩⟩
def transferEvent : Nat := 32979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 32977 .coefficient, .predecessor 1 32978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 32977 .coefficient)
      LeftBound32808.bound (LeftBound32808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 32978 .coefficient)
      LeftBound32791.bound (LeftBound32791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32808.bound, LeftBound32791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32808.bound, LeftBound32791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32808.actual selector witness, LeftBound32791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32979

namespace LeftBound32982
def owner : Owner := ⟨.program ⟨257⟩, ⟨47577⟩⟩
def transferEvent : Nat := 32982
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 32976 .summary, .result 32798 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32976 .summary)
      LeftBound32810.bound (LeftBound32810.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46399⟩⟩) (rawTerms := some (Proof.Events128.exact32976RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32798 .summary)
      LeftBound32793.bound (LeftBound32793.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47576⟩⟩) (rawTerms := some (Proof.Events128.exact32798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32810.bound, LeftBound32793.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32810.bound, LeftBound32793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound32810.actual selector witness, LeftBound32793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32982

namespace LeftBound33006
def owner : Owner := ⟨.program ⟨257⟩, ⟨42693⟩⟩
def transferEvent : Nat := 33006
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 33004 .coefficient) (.predecessor 1 33005 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33004 .coefficient)
      LeftAuthority887.bound (LeftAuthority887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33005 .coefficient)
      LeftBound32026.bound (LeftBound32026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority887.bound LeftBound32026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority887.bound, LeftBound32026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority887.actual selector witness) * (LeftBound32026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33006

namespace LeftBound33011
def owner : Owner := ⟨.program ⟨257⟩, ⟨11616⟩⟩
def transferEvent : Nat := 33011
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33009 .coefficient) (.predecessor 1 33010 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33009 .coefficient)
      LeftBound31897.bound (LeftBound31897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33010 .coefficient)
      LeftBound18081.bound (LeftBound18081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound31897.bound LeftBound18081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31897.bound, LeftBound18081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound31897.actual selector witness) * (LeftBound18081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33011

namespace LeftBound33016
def owner : Owner := ⟨.program ⟨257⟩, ⟨42694⟩⟩
def transferEvent : Nat := 33016
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33014 .coefficient, .predecessor 1 33015 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33014 .coefficient)
      LeftBound33011.bound (LeftBound33011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33015 .coefficient)
      LeftBound33006.bound (LeftBound33006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33011.bound, LeftBound33006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33011.bound, LeftBound33006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound33011.actual selector witness, LeftBound33006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33016

namespace LeftBound33020
def owner : Owner := ⟨.program ⟨257⟩, ⟨42695⟩⟩
def transferEvent : Nat := 33020
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33018 .coefficient, .predecessor 1 33019 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33018 .coefficient)
      LeftBound33016.bound (LeftBound33016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33019 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33016.bound, LeftBound18073.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33016.bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound33016.actual selector witness, LeftBound18073.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33020

namespace LeftBound33021
def owner : Owner := ⟨.program ⟨257⟩, ⟨42695⟩⟩
def transferEvent : Nat := 33021
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩ [⟨.result 18074 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18074 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨109⟩⟩) (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound18073.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound18073.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33021

namespace LeftBound33026
def owner : Owner := ⟨.program ⟨257⟩, ⟨42696⟩⟩
def transferEvent : Nat := 33026
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33024 .coefficient) (.predecessor 1 33025 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33024 .coefficient)
      LeftBound33020.bound (LeftBound33020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact33023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33025 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound33020.bound LeftAuthority890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33020.bound, LeftAuthority890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound33020.actual selector witness) * (LeftAuthority890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33026

namespace LeftBound33027
def owner : Owner := ⟨.program ⟨257⟩, ⟨42696⟩⟩
def transferEvent : Nat := 33027
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14616⟩⟩], []⟩ [⟨.result 891 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 891 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14616⟩⟩) (rawTerms := some (Proof.Events003.exact891RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority890.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority890.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33027

namespace LeftBound33028
def owner : Owner := ⟨.program ⟨257⟩, ⟨42696⟩⟩
def transferEvent : Nat := 33028
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 33023 .summary) (.transfer 33027) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 33023 .summary)
      LeftBound33021.bound (LeftBound33021.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42695⟩⟩) (rawTerms := some (Proof.Events128.exact33023RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 33027)
      LeftBound33027.bound (LeftBound33027.actual selector witness) := by
  exact .transfer (LeftBound33027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound33021.bound LeftBound33027.bound
def bound : CoeffClass := .finite ⟨44302336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33021.bound, LeftBound33027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound33021.actual selector witness) * (LeftBound33027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33028

namespace LeftBound33034
def owner : Owner := ⟨.program ⟨257⟩, ⟨14617⟩⟩
def transferEvent : Nat := 33034
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 33032 .coefficient) (.predecessor 1 33033 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33032 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33033 .coefficient)
      LeftBound32026.bound (LeftBound32026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority890.bound LeftBound32026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority890.bound, LeftBound32026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority890.actual selector witness) * (LeftBound32026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound33034

namespace LeftBound33039
def owner : Owner := ⟨.program ⟨257⟩, ⟨11633⟩⟩
def transferEvent : Nat := 33039
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33037 .coefficient) (.predecessor 1 33038 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 33037 .coefficient)
      LeftBound31897.bound (LeftBound31897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 33038 .coefficient)
      LeftBound18122.bound (LeftBound18122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18122.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound31897.bound LeftBound18122.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31897.bound, LeftBound18122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound31897.actual selector witness) * (LeftBound18122.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33039

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
