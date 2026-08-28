import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard853
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard854
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard856
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard857
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard858
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard859
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard860
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard861
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard862
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard878

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound133976
def owner : Owner := ⟨.program ⟨257⟩, ⟨69853⟩⟩
def transferEvent : Nat := 133976
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133974 .coefficient, .predecessor 1 133975 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133974 .coefficient)
      LeftBound133971.bound (LeftBound133971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133975 .coefficient)
      LeftBound131552.bound (LeftBound131552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events513.exact131559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133971.bound, LeftBound131552.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133971.bound, LeftBound131552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133971.actual selector witness, LeftBound131552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133976

namespace LeftBound133977
def owner : Owner := ⟨.program ⟨257⟩, ⟨69853⟩⟩
def transferEvent : Nat := 133977
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133973 .summary, .result 131559 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133973 .summary)
      LeftBound133972.bound (LeftBound133972.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69852⟩⟩) (rawTerms := some (Proof.Events523.exact133973RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 131559 .summary)
      LeftBound131554.bound (LeftBound131554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30867⟩⟩) (rawTerms := some (Proof.Events513.exact131559RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133972.bound, LeftBound131554.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133972.bound, LeftBound131554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133972.actual selector witness, LeftBound131554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133977

namespace LeftBound133981
def owner : Owner := ⟨.program ⟨257⟩, ⟨69854⟩⟩
def transferEvent : Nat := 133981
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133979 .coefficient, .predecessor 1 133980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133979 .coefficient)
      LeftBound133976.bound (LeftBound133976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133980 .coefficient)
      LeftBound131340.bound (LeftBound131340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events513.exact131347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133976.bound, LeftBound131340.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133976.bound, LeftBound131340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133976.actual selector witness, LeftBound131340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133981

namespace LeftBound133982
def owner : Owner := ⟨.program ⟨257⟩, ⟨69854⟩⟩
def transferEvent : Nat := 133982
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133978 .summary, .result 131347 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133978 .summary)
      LeftBound133977.bound (LeftBound133977.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69853⟩⟩) (rawTerms := some (Proof.Events523.exact133978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 131347 .summary)
      LeftBound131342.bound (LeftBound131342.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36527⟩⟩) (rawTerms := some (Proof.Events513.exact131347RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131342.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133977.bound, LeftBound131342.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133977.bound, LeftBound131342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133977.actual selector witness, LeftBound131342.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133982

namespace LeftBound133986
def owner : Owner := ⟨.program ⟨257⟩, ⟨69855⟩⟩
def transferEvent : Nat := 133986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133984 .coefficient, .predecessor 1 133985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133984 .coefficient)
      LeftBound133981.bound (LeftBound133981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133985 .coefficient)
      LeftBound131128.bound (LeftBound131128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events512.exact131135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133981.bound, LeftBound131128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133981.bound, LeftBound131128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133981.actual selector witness, LeftBound131128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133986

namespace LeftBound133987
def owner : Owner := ⟨.program ⟨257⟩, ⟨69855⟩⟩
def transferEvent : Nat := 133987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133983 .summary, .result 131135 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133983 .summary)
      LeftBound133982.bound (LeftBound133982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69854⟩⟩) (rawTerms := some (Proof.Events523.exact133983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 131135 .summary)
      LeftBound131130.bound (LeftBound131130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39207⟩⟩) (rawTerms := some (Proof.Events512.exact131135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133982.bound, LeftBound131130.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133982.bound, LeftBound131130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133982.actual selector witness, LeftBound131130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133987

namespace LeftBound133991
def owner : Owner := ⟨.program ⟨257⟩, ⟨69856⟩⟩
def transferEvent : Nat := 133991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133989 .coefficient, .predecessor 1 133990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133989 .coefficient)
      LeftBound133986.bound (LeftBound133986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133990 .coefficient)
      LeftBound130916.bound (LeftBound130916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133986.bound, LeftBound130916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133986.bound, LeftBound130916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133986.actual selector witness, LeftBound130916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133991

namespace LeftBound133992
def owner : Owner := ⟨.program ⟨257⟩, ⟨69856⟩⟩
def transferEvent : Nat := 133992
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133988 .summary, .result 130923 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133988 .summary)
      LeftBound133987.bound (LeftBound133987.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69855⟩⟩) (rawTerms := some (Proof.Events523.exact133988RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130923 .summary)
      LeftBound130918.bound (LeftBound130918.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41887⟩⟩) (rawTerms := some (Proof.Events511.exact130923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133987.bound, LeftBound130918.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133987.bound, LeftBound130918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133987.actual selector witness, LeftBound130918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133992

namespace LeftBound133996
def owner : Owner := ⟨.program ⟨257⟩, ⟨69857⟩⟩
def transferEvent : Nat := 133996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133994 .coefficient, .predecessor 1 133995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133994 .coefficient)
      LeftBound133991.bound (LeftBound133991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 133995 .coefficient)
      LeftBound130704.bound (LeftBound130704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events510.exact130711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133991.bound, LeftBound130704.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133991.bound, LeftBound130704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133991.actual selector witness, LeftBound130704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133996

namespace LeftBound133997
def owner : Owner := ⟨.program ⟨257⟩, ⟨69857⟩⟩
def transferEvent : Nat := 133997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133993 .summary, .result 130711 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133993 .summary)
      LeftBound133992.bound (LeftBound133992.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69856⟩⟩) (rawTerms := some (Proof.Events523.exact133993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130711 .summary)
      LeftBound130706.bound (LeftBound130706.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44567⟩⟩) (rawTerms := some (Proof.Events510.exact130711RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130706.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133992.bound, LeftBound130706.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133992.bound, LeftBound130706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133992.actual selector witness, LeftBound130706.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound133997

namespace LeftBound134001
def owner : Owner := ⟨.program ⟨257⟩, ⟨69858⟩⟩
def transferEvent : Nat := 134001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 133999 .coefficient, .predecessor 1 134000 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 133999 .coefficient)
      LeftBound133996.bound (LeftBound133996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact133998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound133996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound133996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 134000 .coefficient)
      LeftBound130492.bound (LeftBound130492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130492.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133996.bound, LeftBound130492.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133996.bound, LeftBound130492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133996.actual selector witness, LeftBound130492.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134001

namespace LeftBound134002
def owner : Owner := ⟨.program ⟨257⟩, ⟨69858⟩⟩
def transferEvent : Nat := 134002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 133998 .summary, .result 130499 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 133998 .summary)
      LeftBound133997.bound (LeftBound133997.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69857⟩⟩) (rawTerms := some (Proof.Events523.exact133998RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound133997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130499 .summary)
      LeftBound130494.bound (LeftBound130494.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47247⟩⟩) (rawTerms := some (Proof.Events509.exact130499RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130494.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound133997.bound, LeftBound130494.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound133997.bound, LeftBound130494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound133997.actual selector witness, LeftBound130494.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134002

namespace LeftBound134006
def owner : Owner := ⟨.program ⟨257⟩, ⟨69859⟩⟩
def transferEvent : Nat := 134006
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 134004 .coefficient, .predecessor 1 134005 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 134004 .coefficient)
      LeftBound134001.bound (LeftBound134001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact134003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 134005 .coefficient)
      LeftBound130280.bound (LeftBound130280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events508.exact130287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound134001.bound, LeftBound130280.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134001.bound, LeftBound130280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound134001.actual selector witness, LeftBound130280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134006

namespace LeftBound134007
def owner : Owner := ⟨.program ⟨257⟩, ⟨69859⟩⟩
def transferEvent : Nat := 134007
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 134003 .summary, .result 130287 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134003 .summary)
      LeftBound134002.bound (LeftBound134002.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69858⟩⟩) (rawTerms := some (Proof.Events523.exact134003RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130287 .summary)
      LeftBound130282.bound (LeftBound130282.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49927⟩⟩) (rawTerms := some (Proof.Events508.exact130287RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound134002.bound, LeftBound130282.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134002.bound, LeftBound130282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound134002.actual selector witness, LeftBound130282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134007

namespace LeftBound134011
def owner : Owner := ⟨.program ⟨257⟩, ⟨71119⟩⟩
def transferEvent : Nat := 134011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 134009 .coefficient, .predecessor 1 134010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 134009 .coefficient)
      LeftBound134006.bound (LeftBound134006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events523.exact134008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 134010 .coefficient)
      LeftBound130068.bound (LeftBound130068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events508.exact130075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound134006.bound, LeftBound130068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134006.bound, LeftBound130068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound134006.actual selector witness, LeftBound130068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134011

namespace LeftBound134012
def owner : Owner := ⟨.program ⟨257⟩, ⟨71119⟩⟩
def transferEvent : Nat := 134012
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 134008 .summary, .result 130075 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134008 .summary)
      LeftBound134007.bound (LeftBound134007.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69859⟩⟩) (rawTerms := some (Proof.Events523.exact134008RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130075 .summary)
      LeftBound130070.bound (LeftBound130070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71117⟩⟩) (rawTerms := some (Proof.Events508.exact130075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound134007.bound, LeftBound130070.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134007.bound, LeftBound130070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound134007.actual selector witness, LeftBound130070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound134012

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
