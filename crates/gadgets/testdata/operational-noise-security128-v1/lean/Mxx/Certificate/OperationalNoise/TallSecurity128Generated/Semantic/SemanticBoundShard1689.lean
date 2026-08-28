import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1680
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1681
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1683
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1684
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1685
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1686
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1687
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1688

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound250914
def owner : Owner := ⟨.program ⟨257⟩, ⟨9471⟩⟩
def transferEvent : Nat := 250914
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250912 .coefficient, .predecessor 1 250913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250912 .coefficient)
      LeftBound250908.bound (LeftBound250908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250913 .coefficient)
      LeftBound250908.bound (LeftBound250908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250908.bound, LeftBound250908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250908.bound, LeftBound250908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250908.actual selector witness, LeftBound250908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250914

namespace LeftBound250917
def owner : Owner := ⟨.program ⟨257⟩, ⟨9471⟩⟩
def transferEvent : Nat := 250917
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250911 .summary, .result 250911 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250911 .summary)
      LeftBound250909.bound (LeftBound250909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9366⟩⟩) (rawTerms := some (Proof.Events980.exact250911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250911 .summary)
      LeftBound250909.bound (LeftBound250909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9366⟩⟩) (rawTerms := some (Proof.Events980.exact250911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250909.bound, LeftBound250909.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250909.bound, LeftBound250909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250909.actual selector witness, LeftBound250909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250917

namespace LeftBound250921
def owner : Owner := ⟨.program ⟨257⟩, ⟨17703⟩⟩
def transferEvent : Nat := 250921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250919 .coefficient, .predecessor 1 250920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250919 .coefficient)
      LeftBound250914.bound (LeftBound250914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250920 .coefficient)
      LeftBound250884.bound (LeftBound250884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250914.bound, LeftBound250884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250914.bound, LeftBound250884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250914.actual selector witness, LeftBound250884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250921

namespace LeftBound250922
def owner : Owner := ⟨.program ⟨257⟩, ⟨17703⟩⟩
def transferEvent : Nat := 250922
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250918 .summary, .result 250891 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250918 .summary)
      LeftBound250917.bound (LeftBound250917.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9471⟩⟩) (rawTerms := some (Proof.Events980.exact250918RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250891 .summary)
      LeftBound250886.bound (LeftBound250886.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17702⟩⟩) (rawTerms := some (Proof.Events980.exact250891RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250886.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250917.bound, LeftBound250886.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250917.bound, LeftBound250886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250917.actual selector witness, LeftBound250886.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250922

namespace LeftBound250926
def owner : Owner := ⟨.program ⟨257⟩, ⟨20588⟩⟩
def transferEvent : Nat := 250926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250924 .coefficient, .predecessor 1 250925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250924 .coefficient)
      LeftBound250921.bound (LeftBound250921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250925 .coefficient)
      LeftBound250672.bound (LeftBound250672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events979.exact250679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250921.bound, LeftBound250672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250921.bound, LeftBound250672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250921.actual selector witness, LeftBound250672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250926

namespace LeftBound250927
def owner : Owner := ⟨.program ⟨257⟩, ⟨20588⟩⟩
def transferEvent : Nat := 250927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250923 .summary, .result 250679 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250923 .summary)
      LeftBound250922.bound (LeftBound250922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17703⟩⟩) (rawTerms := some (Proof.Events980.exact250923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250679 .summary)
      LeftBound250674.bound (LeftBound250674.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20587⟩⟩) (rawTerms := some (Proof.Events979.exact250679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250922.bound, LeftBound250674.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250922.bound, LeftBound250674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250922.actual selector witness, LeftBound250674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250927

namespace LeftBound250931
def owner : Owner := ⟨.program ⟨257⟩, ⟨23808⟩⟩
def transferEvent : Nat := 250931
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250929 .coefficient, .predecessor 1 250930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250929 .coefficient)
      LeftBound250926.bound (LeftBound250926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250930 .coefficient)
      LeftBound250460.bound (LeftBound250460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events978.exact250467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250926.bound, LeftBound250460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250926.bound, LeftBound250460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250926.actual selector witness, LeftBound250460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250931

namespace LeftBound250932
def owner : Owner := ⟨.program ⟨257⟩, ⟨23808⟩⟩
def transferEvent : Nat := 250932
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250928 .summary, .result 250467 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250928 .summary)
      LeftBound250927.bound (LeftBound250927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20588⟩⟩) (rawTerms := some (Proof.Events980.exact250928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250467 .summary)
      LeftBound250462.bound (LeftBound250462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23807⟩⟩) (rawTerms := some (Proof.Events978.exact250467RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250462.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250927.bound, LeftBound250462.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250927.bound, LeftBound250462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250927.actual selector witness, LeftBound250462.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250932

namespace LeftBound250936
def owner : Owner := ⟨.program ⟨257⟩, ⟨33828⟩⟩
def transferEvent : Nat := 250936
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250934 .coefficient, .predecessor 1 250935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250934 .coefficient)
      LeftBound250931.bound (LeftBound250931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250935 .coefficient)
      LeftBound250248.bound (LeftBound250248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events977.exact250255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250931.bound, LeftBound250248.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250931.bound, LeftBound250248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250931.actual selector witness, LeftBound250248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250936

namespace LeftBound250937
def owner : Owner := ⟨.program ⟨257⟩, ⟨33828⟩⟩
def transferEvent : Nat := 250937
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250933 .summary, .result 250255 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250933 .summary)
      LeftBound250932.bound (LeftBound250932.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23808⟩⟩) (rawTerms := some (Proof.Events980.exact250933RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250255 .summary)
      LeftBound250250.bound (LeftBound250250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33827⟩⟩) (rawTerms := some (Proof.Events977.exact250255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250932.bound, LeftBound250250.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250932.bound, LeftBound250250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250932.actual selector witness, LeftBound250250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250937

namespace LeftBound250941
def owner : Owner := ⟨.program ⟨257⟩, ⟨52888⟩⟩
def transferEvent : Nat := 250941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250939 .coefficient, .predecessor 1 250940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250939 .coefficient)
      LeftBound250936.bound (LeftBound250936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250940 .coefficient)
      LeftBound250036.bound (LeftBound250036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events976.exact250043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250936.bound, LeftBound250036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250936.bound, LeftBound250036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250936.actual selector witness, LeftBound250036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250941

namespace LeftBound250942
def owner : Owner := ⟨.program ⟨257⟩, ⟨52888⟩⟩
def transferEvent : Nat := 250942
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250938 .summary, .result 250043 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250938 .summary)
      LeftBound250937.bound (LeftBound250937.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33828⟩⟩) (rawTerms := some (Proof.Events980.exact250938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250043 .summary)
      LeftBound250038.bound (LeftBound250038.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52887⟩⟩) (rawTerms := some (Proof.Events976.exact250043RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250038.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250937.bound, LeftBound250038.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250937.bound, LeftBound250038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250937.actual selector witness, LeftBound250038.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250942

namespace LeftBound250946
def owner : Owner := ⟨.program ⟨257⟩, ⟨55868⟩⟩
def transferEvent : Nat := 250946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250944 .coefficient, .predecessor 1 250945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250944 .coefficient)
      LeftBound250941.bound (LeftBound250941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250945 .coefficient)
      LeftBound249824.bound (LeftBound249824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events975.exact249831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound249824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound249824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250941.bound, LeftBound249824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250941.bound, LeftBound249824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250941.actual selector witness, LeftBound249824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250946

namespace LeftBound250947
def owner : Owner := ⟨.program ⟨257⟩, ⟨55868⟩⟩
def transferEvent : Nat := 250947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250943 .summary, .result 249831 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250943 .summary)
      LeftBound250942.bound (LeftBound250942.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52888⟩⟩) (rawTerms := some (Proof.Events980.exact250943RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 249831 .summary)
      LeftBound249826.bound (LeftBound249826.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55867⟩⟩) (rawTerms := some (Proof.Events975.exact249831RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound249826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250942.bound, LeftBound249826.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250942.bound, LeftBound249826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250942.actual selector witness, LeftBound249826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250947

namespace LeftBound250951
def owner : Owner := ⟨.program ⟨257⟩, ⟨58848⟩⟩
def transferEvent : Nat := 250951
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250949 .coefficient, .predecessor 1 250950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250949 .coefficient)
      LeftBound250946.bound (LeftBound250946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250950 .coefficient)
      LeftBound249612.bound (LeftBound249612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events975.exact249619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound249612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound249612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250946.bound, LeftBound249612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250946.bound, LeftBound249612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250946.actual selector witness, LeftBound249612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250951

namespace LeftBound250952
def owner : Owner := ⟨.program ⟨257⟩, ⟨58848⟩⟩
def transferEvent : Nat := 250952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250948 .summary, .result 249619 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250948 .summary)
      LeftBound250947.bound (LeftBound250947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55868⟩⟩) (rawTerms := some (Proof.Events980.exact250948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 249619 .summary)
      LeftBound249614.bound (LeftBound249614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58847⟩⟩) (rawTerms := some (Proof.Events975.exact249619RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound249614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250947.bound, LeftBound249614.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250947.bound, LeftBound249614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250947.actual selector witness, LeftBound249614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250952

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
