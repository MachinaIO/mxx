import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1977
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1978
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1979
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1981
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1983
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1984
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1987
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1993

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound294781
def owner : Owner := ⟨.program ⟨257⟩, ⟨52764⟩⟩
def transferEvent : Nat := 294781
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294777 .summary, .result 293882 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294777 .summary)
      LeftBound294776.bound (LeftBound294776.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33704⟩⟩) (rawTerms := some (Proof.Events1151.exact294777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293882 .summary)
      LeftBound293877.bound (LeftBound293877.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52763⟩⟩) (rawTerms := some (Proof.Events1147.exact293882RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293877.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294776.bound, LeftBound293877.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294776.bound, LeftBound293877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294776.actual selector witness, LeftBound293877.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294781

namespace LeftBound294785
def owner : Owner := ⟨.program ⟨257⟩, ⟨55744⟩⟩
def transferEvent : Nat := 294785
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294783 .coefficient, .predecessor 1 294784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294783 .coefficient)
      LeftBound294780.bound (LeftBound294780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294784 .coefficient)
      LeftBound293663.bound (LeftBound293663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1147.exact293670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294780.bound, LeftBound293663.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294780.bound, LeftBound293663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294780.actual selector witness, LeftBound293663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294785

namespace LeftBound294786
def owner : Owner := ⟨.program ⟨257⟩, ⟨55744⟩⟩
def transferEvent : Nat := 294786
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294782 .summary, .result 293670 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294782 .summary)
      LeftBound294781.bound (LeftBound294781.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52764⟩⟩) (rawTerms := some (Proof.Events1151.exact294782RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293670 .summary)
      LeftBound293665.bound (LeftBound293665.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55743⟩⟩) (rawTerms := some (Proof.Events1147.exact293670RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293665.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294781.bound, LeftBound293665.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294781.bound, LeftBound293665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294781.actual selector witness, LeftBound293665.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294786

namespace LeftBound294790
def owner : Owner := ⟨.program ⟨257⟩, ⟨58724⟩⟩
def transferEvent : Nat := 294790
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294788 .coefficient, .predecessor 1 294789 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294788 .coefficient)
      LeftBound294785.bound (LeftBound294785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294789 .coefficient)
      LeftBound293451.bound (LeftBound293451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1146.exact293458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294785.bound, LeftBound293451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294785.bound, LeftBound293451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294785.actual selector witness, LeftBound293451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294790

namespace LeftBound294791
def owner : Owner := ⟨.program ⟨257⟩, ⟨58724⟩⟩
def transferEvent : Nat := 294791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294787 .summary, .result 293458 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294787 .summary)
      LeftBound294786.bound (LeftBound294786.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55744⟩⟩) (rawTerms := some (Proof.Events1151.exact294787RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293458 .summary)
      LeftBound293453.bound (LeftBound293453.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58723⟩⟩) (rawTerms := some (Proof.Events1146.exact293458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294786.bound, LeftBound293453.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294786.bound, LeftBound293453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294786.actual selector witness, LeftBound293453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294791

namespace LeftBound294795
def owner : Owner := ⟨.program ⟨257⟩, ⟨61704⟩⟩
def transferEvent : Nat := 294795
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294793 .coefficient, .predecessor 1 294794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294793 .coefficient)
      LeftBound294790.bound (LeftBound294790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294794 .coefficient)
      LeftBound293239.bound (LeftBound293239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1145.exact293246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294790.bound, LeftBound293239.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294790.bound, LeftBound293239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294790.actual selector witness, LeftBound293239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294795

namespace LeftBound294796
def owner : Owner := ⟨.program ⟨257⟩, ⟨61704⟩⟩
def transferEvent : Nat := 294796
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294792 .summary, .result 293246 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294792 .summary)
      LeftBound294791.bound (LeftBound294791.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58724⟩⟩) (rawTerms := some (Proof.Events1151.exact294792RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293246 .summary)
      LeftBound293241.bound (LeftBound293241.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61703⟩⟩) (rawTerms := some (Proof.Events1145.exact293246RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293241.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294791.bound, LeftBound293241.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294791.bound, LeftBound293241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294791.actual selector witness, LeftBound293241.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294796

namespace LeftBound294800
def owner : Owner := ⟨.program ⟨257⟩, ⟨64684⟩⟩
def transferEvent : Nat := 294800
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294798 .coefficient, .predecessor 1 294799 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294798 .coefficient)
      LeftBound294795.bound (LeftBound294795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294799 .coefficient)
      LeftBound293027.bound (LeftBound293027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1144.exact293034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound293027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound293027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294795.bound, LeftBound293027.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294795.bound, LeftBound293027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294795.actual selector witness, LeftBound293027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294800

namespace LeftBound294801
def owner : Owner := ⟨.program ⟨257⟩, ⟨64684⟩⟩
def transferEvent : Nat := 294801
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294797 .summary, .result 293034 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294797 .summary)
      LeftBound294796.bound (LeftBound294796.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61704⟩⟩) (rawTerms := some (Proof.Events1151.exact294797RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 293034 .summary)
      LeftBound293029.bound (LeftBound293029.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64683⟩⟩) (rawTerms := some (Proof.Events1144.exact293034RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound293029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294796.bound, LeftBound293029.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294796.bound, LeftBound293029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294796.actual selector witness, LeftBound293029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294801

namespace LeftBound294805
def owner : Owner := ⟨.program ⟨257⟩, ⟨69693⟩⟩
def transferEvent : Nat := 294805
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294803 .coefficient, .predecessor 1 294804 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294803 .coefficient)
      LeftBound294800.bound (LeftBound294800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294804 .coefficient)
      LeftBound292815.bound (LeftBound292815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294800.bound, LeftBound292815.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294800.bound, LeftBound292815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294800.actual selector witness, LeftBound292815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294805

namespace LeftBound294806
def owner : Owner := ⟨.program ⟨257⟩, ⟨69693⟩⟩
def transferEvent : Nat := 294806
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294802 .summary, .result 292822 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294802 .summary)
      LeftBound294801.bound (LeftBound294801.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64684⟩⟩) (rawTerms := some (Proof.Events1151.exact294802RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292822 .summary)
      LeftBound292817.bound (LeftBound292817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69692⟩⟩) (rawTerms := some (Proof.Events1143.exact292822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294801.bound, LeftBound292817.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294801.bound, LeftBound292817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294801.actual selector witness, LeftBound292817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294806

namespace LeftBound294810
def owner : Owner := ⟨.program ⟨257⟩, ⟨69694⟩⟩
def transferEvent : Nat := 294810
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294808 .coefficient, .predecessor 1 294809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294808 .coefficient)
      LeftBound294805.bound (LeftBound294805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294809 .coefficient)
      LeftBound292603.bound (LeftBound292603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294805.bound, LeftBound292603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294805.bound, LeftBound292603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294805.actual selector witness, LeftBound292603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294810

namespace LeftBound294811
def owner : Owner := ⟨.program ⟨257⟩, ⟨69694⟩⟩
def transferEvent : Nat := 294811
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294807 .summary, .result 292610 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294807 .summary)
      LeftBound294806.bound (LeftBound294806.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69693⟩⟩) (rawTerms := some (Proof.Events1151.exact294807RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292610 .summary)
      LeftBound292605.bound (LeftBound292605.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28137⟩⟩) (rawTerms := some (Proof.Events1143.exact292610RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294806.bound, LeftBound292605.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294806.bound, LeftBound292605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294806.actual selector witness, LeftBound292605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294811

namespace LeftBound294815
def owner : Owner := ⟨.program ⟨257⟩, ⟨69695⟩⟩
def transferEvent : Nat := 294815
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294813 .coefficient, .predecessor 1 294814 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294813 .coefficient)
      LeftBound294810.bound (LeftBound294810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294814 .coefficient)
      LeftBound292391.bound (LeftBound292391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294810.bound, LeftBound292391.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294810.bound, LeftBound292391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294810.actual selector witness, LeftBound292391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294815

namespace LeftBound294816
def owner : Owner := ⟨.program ⟨257⟩, ⟨69695⟩⟩
def transferEvent : Nat := 294816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294812 .summary, .result 292398 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294812 .summary)
      LeftBound294811.bound (LeftBound294811.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69694⟩⟩) (rawTerms := some (Proof.Events1151.exact294812RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292398 .summary)
      LeftBound292393.bound (LeftBound292393.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30817⟩⟩) (rawTerms := some (Proof.Events1142.exact292398RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294811.bound, LeftBound292393.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294811.bound, LeftBound292393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294811.actual selector witness, LeftBound292393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294816

namespace LeftBound294820
def owner : Owner := ⟨.program ⟨257⟩, ⟨69696⟩⟩
def transferEvent : Nat := 294820
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294818 .coefficient, .predecessor 1 294819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294818 .coefficient)
      LeftBound294815.bound (LeftBound294815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294819 .coefficient)
      LeftBound292179.bound (LeftBound292179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1141.exact292186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292179.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294815.bound, LeftBound292179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294815.bound, LeftBound292179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294815.actual selector witness, LeftBound292179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294820

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
