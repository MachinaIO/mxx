import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard821
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard865

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound132133
def owner : Owner := ⟨.program ⟨257⟩, ⟨64272⟩⟩
def transferEvent : Nat := 132133
def frameStart : Nat := 132068
def rule : BoundRule := .product (.predecessor 0 132131 .coefficient) (.predecessor 1 132132 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132131 .coefficient)
      LeftAuthority132129.bound (LeftAuthority132129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132132 .coefficient)
      LeftBound132127.bound (LeftBound132127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132127.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority132129.bound LeftBound132127.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132129.bound, LeftBound132127.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority132129.actual selector witness) * (LeftBound132127.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132133

namespace LeftBound132141
def owner : Owner := ⟨.program ⟨257⟩, ⟨64273⟩⟩
def transferEvent : Nat := 132141
def frameStart : Nat := 132068
def rule : BoundRule := .sum [.predecessor 0 132139 .coefficient, .predecessor 1 132140 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132139 .coefficient)
      LeftAuthority132137.bound (LeftAuthority132137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132140 .coefficient)
      LeftBound132133.bound (LeftBound132133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132133.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority132137.bound, LeftBound132133.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132137.bound, LeftBound132133.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority132137.actual selector witness, LeftBound132133.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132141

namespace LeftBound132145
def owner : Owner := ⟨.program ⟨257⟩, ⟨64742⟩⟩
def transferEvent : Nat := 132145
def frameStart : Nat := 132068
def rule : BoundRule := .product (.predecessor 0 132143 .coefficient) (.predecessor 1 132144 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132143 .coefficient)
      LeftBound132141.bound (LeftBound132141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132144 .coefficient)
      LeftAuthority132118.bound (LeftAuthority132118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132118.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound132141.bound LeftAuthority132118.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132141.bound, LeftAuthority132118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound132141.actual selector witness) * (LeftAuthority132118.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132145

namespace LeftBound132156
def owner : Owner := ⟨.program ⟨257⟩, ⟨63012⟩⟩
def transferEvent : Nat := 132156
def frameStart : Nat := 132068
def rule : BoundRule := .product (.predecessor 0 132154 .coefficient) (.predecessor 1 132155 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132154 .coefficient)
      LeftAuthority132129.bound (LeftAuthority132129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132129.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132155 .coefficient)
      LeftAuthority132152.bound (LeftAuthority132152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132152.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority132129.bound LeftAuthority132152.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132129.bound, LeftAuthority132152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority132129.actual selector witness) * (LeftAuthority132152.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132156

namespace LeftBound132164
def owner : Owner := ⟨.program ⟨257⟩, ⟨63013⟩⟩
def transferEvent : Nat := 132164
def frameStart : Nat := 132068
def rule : BoundRule := .sum [.predecessor 0 132162 .coefficient, .predecessor 1 132163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132162 .coefficient)
      LeftAuthority132160.bound (LeftAuthority132160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132160.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132163 .coefficient)
      LeftBound132156.bound (LeftBound132156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority132160.bound, LeftBound132156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132160.bound, LeftBound132156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority132160.actual selector witness, LeftBound132156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132164

namespace LeftBound132168
def owner : Owner := ⟨.program ⟨257⟩, ⟨64747⟩⟩
def transferEvent : Nat := 132168
def frameStart : Nat := 132068
def rule : BoundRule := .sum [.predecessor 0 132166 .coefficient, .predecessor 1 132167 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132166 .coefficient)
      LeftBound132164.bound (LeftBound132164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132167 .coefficient)
      LeftBound132145.bound (LeftBound132145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132145.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132164.bound, LeftBound132145.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132164.bound, LeftBound132145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132164.actual selector witness, LeftBound132145.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132168

namespace LeftBound132181
def owner : Owner := ⟨.program ⟨257⟩, ⟨64744⟩⟩
def transferEvent : Nat := 132181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 132179 .coefficient, .predecessor 1 132180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132179 .coefficient)
      LeftBound132010.bound (LeftBound132010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132180 .coefficient)
      LeftBound131993.bound (LeftBound131993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events515.exact132000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound131993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound131993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132010.bound, LeftBound131993.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132010.bound, LeftBound131993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132010.actual selector witness, LeftBound131993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132181

namespace LeftBound132184
def owner : Owner := ⟨.program ⟨257⟩, ⟨64744⟩⟩
def transferEvent : Nat := 132184
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 132178 .summary, .result 132000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132178 .summary)
      LeftBound132012.bound (LeftBound132012.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63595⟩⟩) (rawTerms := some (Proof.Events516.exact132178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132000 .summary)
      LeftBound131995.bound (LeftBound131995.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64743⟩⟩) (rawTerms := some (Proof.Events515.exact132000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound131995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound132012.bound, LeftBound131995.bound]
def bound : CoeffClass := .finite ⟨32190771716940580661919523012608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132012.bound, LeftBound131995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound132012.actual selector witness, LeftBound131995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound132184

namespace LeftBound132188
def owner : Owner := ⟨.program ⟨257⟩, ⟨64745⟩⟩
def transferEvent : Nat := 132188
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 132186 .coefficient) (.predecessor 1 132187 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132186 .coefficient)
      LeftBound132181.bound (LeftBound132181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132187 .coefficient)
      LeftBound15721.bound (LeftBound15721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15721.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound132181.bound LeftBound15721.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132181.bound, LeftBound15721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound132181.actual selector witness) * (LeftBound15721.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132188

namespace LeftBound132189
def owner : Owner := ⟨.program ⟨257⟩, ⟨64745⟩⟩
def transferEvent : Nat := 132189
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩ [⟨.result 15718 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15718 .coefficient)
      LeftAuthority15717.bound (LeftAuthority15717.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7099⟩⟩) (rawTerms := some (Proof.Events061.exact15718RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15717.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15717.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15717.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound132189

namespace LeftBound132190
def owner : Owner := ⟨.program ⟨257⟩, ⟨64745⟩⟩
def transferEvent : Nat := 132190
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 132185 .summary) (.transfer 132189) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132185 .summary)
      LeftBound132184.bound (LeftBound132184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64744⟩⟩) (rawTerms := some (Proof.Events516.exact132185RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound132184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 132189)
      LeftBound132189.bound (LeftBound132189.actual selector witness) := by
  exact .transfer (LeftBound132189.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound132184.bound LeftBound132189.bound
def bound : CoeffClass := .finite ⟨345645779393153907795485959807676889169920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound132184.bound, LeftBound132189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound132184.actual selector witness) * (LeftBound132189.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132190

namespace LeftBound132205
def owner : Owner := ⟨.program ⟨257⟩, ⟨61763⟩⟩
def transferEvent : Nat := 132205
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 132203 .coefficient) (.predecessor 1 132204 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132203 .coefficient)
      LeftBound124872.bound (LeftBound124872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events487.exact124876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound124872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound124872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132204 .coefficient)
      LeftAuthority132201.bound (LeftAuthority132201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132201.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132201.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound124872.bound LeftAuthority132201.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound124872.bound, LeftAuthority132201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound124872.actual selector witness) * (LeftAuthority132201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132205

namespace LeftBound132206
def owner : Owner := ⟨.program ⟨257⟩, ⟨61763⟩⟩
def transferEvent : Nat := 132206
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61761⟩⟩]⟩ [⟨.result 132202 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 132202 .coefficient)
      LeftAuthority132201.bound (LeftAuthority132201.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61761⟩⟩) (rawTerms := some (Proof.Events516.exact132202RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132201.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132201.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority132201.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority132201.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound132206

namespace LeftBound132207
def owner : Owner := ⟨.program ⟨257⟩, ⟨61763⟩⟩
def transferEvent : Nat := 132207
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 124876 .summary) (.transfer 132206) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 124876 .summary)
      LeftBound124875.bound (LeftBound124875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61417⟩⟩) (rawTerms := some (Proof.Events487.exact124876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound124875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 132206)
      LeftBound132206.bound (LeftBound132206.actual selector witness) := by
  exact .transfer (LeftBound132206.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound124875.bound LeftBound132206.bound
def bound : CoeffClass := .finite ⟨32190378816049003834595889643520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound124875.bound, LeftBound132206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound124875.actual selector witness) * (LeftBound132206.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132207

namespace LeftBound132218
def owner : Owner := ⟨.program ⟨257⟩, ⟨60614⟩⟩
def transferEvent : Nat := 132218
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 132216 .coefficient) (.value (.predecessor 1 132217 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132216 .coefficient)
      LeftAuthority132214.bound (LeftAuthority132214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority132214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority132214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132217 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority132214.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority132214.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority132214.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound132218

namespace LeftBound132222
def owner : Owner := ⟨.program ⟨257⟩, ⟨60615⟩⟩
def transferEvent : Nat := 132222
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 132220 .coefficient) (.predecessor 1 132221 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 132220 .coefficient)
      LeftBound119867.bound (LeftBound119867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 132221 .coefficient)
      LeftBound132218.bound (LeftBound132218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events516.exact132219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound132218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound132218.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119867.bound LeftBound132218.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119867.bound, LeftBound132218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119867.actual selector witness) * (LeftBound132218.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound132222

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
