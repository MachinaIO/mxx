import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard625
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard665

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound103338
def owner : Owner := ⟨.program ⟨257⟩, ⟨57224⟩⟩
def transferEvent : Nat := 103338
def frameStart : Nat := 103242
def rule : BoundRule := .sum [.predecessor 0 103336 .coefficient, .predecessor 1 103337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103336 .coefficient)
      LeftAuthority103334.bound (LeftAuthority103334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103337 .coefficient)
      LeftBound103330.bound (LeftBound103330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103334.bound, LeftBound103330.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103334.bound, LeftBound103330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority103334.actual selector witness, LeftBound103330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103338

namespace LeftBound103342
def owner : Owner := ⟨.program ⟨257⟩, ⟨59066⟩⟩
def transferEvent : Nat := 103342
def frameStart : Nat := 103242
def rule : BoundRule := .sum [.predecessor 0 103340 .coefficient, .predecessor 1 103341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103340 .coefficient)
      LeftBound103338.bound (LeftBound103338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103341 .coefficient)
      LeftBound103319.bound (LeftBound103319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103319.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103338.bound, LeftBound103319.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103338.bound, LeftBound103319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound103338.actual selector witness, LeftBound103319.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103342

namespace LeftBound103355
def owner : Owner := ⟨.program ⟨257⟩, ⟨59063⟩⟩
def transferEvent : Nat := 103355
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 103353 .coefficient, .predecessor 1 103354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103353 .coefficient)
      LeftBound103184.bound (LeftBound103184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103354 .coefficient)
      LeftBound103167.bound (LeftBound103167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103184.bound, LeftBound103167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103184.bound, LeftBound103167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound103184.actual selector witness, LeftBound103167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103355

namespace LeftBound103358
def owner : Owner := ⟨.program ⟨257⟩, ⟨59063⟩⟩
def transferEvent : Nat := 103358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 103352 .summary, .result 103174 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103352 .summary)
      LeftBound103186.bound (LeftBound103186.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57815⟩⟩) (rawTerms := some (Proof.Events403.exact103352RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103174 .summary)
      LeftBound103169.bound (LeftBound103169.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59062⟩⟩) (rawTerms := some (Proof.Events403.exact103174RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103186.bound, LeftBound103169.bound]
def bound : CoeffClass := .finite ⟨32190182365603518530196853751808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103186.bound, LeftBound103169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound103186.actual selector witness, LeftBound103169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103358

namespace LeftBound103362
def owner : Owner := ⟨.program ⟨257⟩, ⟨59064⟩⟩
def transferEvent : Nat := 103362
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103360 .coefficient) (.predecessor 1 103361 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103360 .coefficient)
      LeftBound103355.bound (LeftBound103355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103361 .coefficient)
      LeftBound15761.bound (LeftBound15761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound103355.bound LeftBound15761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103355.bound, LeftBound15761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound103355.actual selector witness) * (LeftBound15761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103362

namespace LeftBound103363
def owner : Owner := ⟨.program ⟨257⟩, ⟨59064⟩⟩
def transferEvent : Nat := 103363
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩ [⟨.result 15758 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15758 .coefficient)
      LeftAuthority15757.bound (LeftAuthority15757.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7107⟩⟩) (rawTerms := some (Proof.Events061.exact15758RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15757.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15757.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15757.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15757.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103363

namespace LeftBound103364
def owner : Owner := ⟨.program ⟨257⟩, ⟨59064⟩⟩
def transferEvent : Nat := 103364
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 103359 .summary) (.transfer 103363) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103359 .summary)
      LeftBound103358.bound (LeftBound103358.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59063⟩⟩) (rawTerms := some (Proof.Events403.exact103359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 103363)
      LeftBound103363.bound (LeftBound103363.actual selector witness) := by
  exact .transfer (LeftBound103363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound103358.bound LeftBound103363.bound
def bound : CoeffClass := .finite ⟨345639451281357568474313688265275652177920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103358.bound, LeftBound103363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound103358.actual selector witness) * (LeftBound103363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103364

namespace LeftBound103379
def owner : Owner := ⟨.program ⟨257⟩, ⟨56082⟩⟩
def transferEvent : Nat := 103379
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103377 .coefficient) (.predecessor 1 103378 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103377 .coefficient)
      LeftBound96586.bound (LeftBound96586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103378 .coefficient)
      LeftAuthority103375.bound (LeftAuthority103375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103375.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound96586.bound LeftAuthority103375.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96586.bound, LeftAuthority103375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound96586.actual selector witness) * (LeftAuthority103375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103379

namespace LeftBound103380
def owner : Owner := ⟨.program ⟨257⟩, ⟨56082⟩⟩
def transferEvent : Nat := 103380
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨56080⟩⟩]⟩ [⟨.result 103376 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103376 .coefficient)
      LeftAuthority103375.bound (LeftAuthority103375.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨56080⟩⟩) (rawTerms := some (Proof.Events403.exact103376RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103375.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103375.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority103375.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority103375.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103380

namespace LeftBound103381
def owner : Owner := ⟨.program ⟨257⟩, ⟨56082⟩⟩
def transferEvent : Nat := 103381
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96590 .summary) (.transfer 103380) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96590 .summary)
      LeftBound96589.bound (LeftBound96589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55556⟩⟩) (rawTerms := some (Proof.Events377.exact96590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 103380)
      LeftBound103380.bound (LeftBound103380.actual selector witness) := by
  exact .transfer (LeftBound103380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound96589.bound LeftBound103380.bound
def bound : CoeffClass := .finite ⟨32189789464711941702873220382720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96589.bound, LeftBound103380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound96589.actual selector witness) * (LeftBound103380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103381

namespace LeftBound103392
def owner : Owner := ⟨.program ⟨257⟩, ⟨54834⟩⟩
def transferEvent : Nat := 103392
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 103390 .coefficient) (.value (.predecessor 1 103391 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103390 .coefficient)
      LeftAuthority103388.bound (LeftAuthority103388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103391 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority103388.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103388.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority103388.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound103392

namespace LeftBound103396
def owner : Owner := ⟨.program ⟨257⟩, ⟨54835⟩⟩
def transferEvent : Nat := 103396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103394 .coefficient) (.predecessor 1 103395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103394 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103395 .coefficient)
      LeftBound103392.bound (LeftBound103392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound103392.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound103392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound103392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103396

namespace LeftBound103397
def owner : Owner := ⟨.program ⟨257⟩, ⟨54835⟩⟩
def transferEvent : Nat := 103397
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨54832⟩⟩]⟩ [⟨.result 103389 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103389 .coefficient)
      LeftAuthority103388.bound (LeftAuthority103388.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54832⟩⟩) (rawTerms := some (Proof.Events403.exact103389RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103388.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority103388.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority103388.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103397

namespace LeftBound103398
def owner : Owner := ⟨.program ⟨257⟩, ⟨54835⟩⟩
def transferEvent : Nat := 103398
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90620 .summary) (.transfer 103397) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90620 .summary)
      LeftBound90618.bound (LeftBound90618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9944⟩⟩) (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 103397)
      LeftBound103397.bound (LeftBound103397.actual selector witness) := by
  exact .transfer (LeftBound103397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90618.bound LeftBound103397.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90618.bound, LeftBound103397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90618.actual selector witness) * (LeftBound103397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103398

namespace LeftBound103493
def owner : Owner := ⟨.program ⟨257⟩, ⟨53909⟩⟩
def transferEvent : Nat := 103493
def frameStart : Nat := 103454
def rule : BoundRule := .identity (.predecessor 0 103492 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103492 .coefficient)
      LeftAuthority103490.bound (LeftAuthority103490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103490.derived selector witness)

def rawBound : CoeffClass := LeftAuthority103490.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority103490.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound103493

namespace LeftBound103510
def owner : Owner := ⟨.program ⟨257⟩, ⟨55366⟩⟩
def transferEvent : Nat := 103510
def frameStart : Nat := 103454
def rule : BoundRule := .sum [.predecessor 0 103508 .coefficient, .predecessor 1 103509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 103508 .coefficient)
      LeftBound103493.bound (LeftBound103493.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound103493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 103509 .coefficient)
      LeftAuthority103506.bound (LeftAuthority103506.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority103506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103493.bound, LeftAuthority103506.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103493.bound, LeftAuthority103506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound103493.actual selector witness, LeftAuthority103506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103510

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
