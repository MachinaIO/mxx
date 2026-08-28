import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1170
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1172
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1173
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1174
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1176
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1177
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1178
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1180
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1181

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound177797
def owner : Owner := ⟨.program ⟨257⟩, ⟨17871⟩⟩
def transferEvent : Nat := 177797
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177793 .summary, .result 177766 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177793 .summary)
      LeftBound177792.bound (LeftBound177792.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9489⟩⟩) (rawTerms := some (Proof.Events694.exact177793RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177766 .summary)
      LeftBound177761.bound (LeftBound177761.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17870⟩⟩) (rawTerms := some (Proof.Events694.exact177766RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177761.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177792.bound, LeftBound177761.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177792.bound, LeftBound177761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177792.actual selector witness, LeftBound177761.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177797

namespace LeftBound177801
def owner : Owner := ⟨.program ⟨257⟩, ⟨20774⟩⟩
def transferEvent : Nat := 177801
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177799 .coefficient, .predecessor 1 177800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177799 .coefficient)
      LeftBound177796.bound (LeftBound177796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177800 .coefficient)
      LeftBound177547.bound (LeftBound177547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events693.exact177554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177796.bound, LeftBound177547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177796.bound, LeftBound177547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177796.actual selector witness, LeftBound177547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177801

namespace LeftBound177802
def owner : Owner := ⟨.program ⟨257⟩, ⟨20774⟩⟩
def transferEvent : Nat := 177802
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177798 .summary, .result 177554 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177798 .summary)
      LeftBound177797.bound (LeftBound177797.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17871⟩⟩) (rawTerms := some (Proof.Events694.exact177798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177554 .summary)
      LeftBound177549.bound (LeftBound177549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20773⟩⟩) (rawTerms := some (Proof.Events693.exact177554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177797.bound, LeftBound177549.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177797.bound, LeftBound177549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177797.actual selector witness, LeftBound177549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177802

namespace LeftBound177806
def owner : Owner := ⟨.program ⟨257⟩, ⟨23994⟩⟩
def transferEvent : Nat := 177806
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177804 .coefficient, .predecessor 1 177805 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177804 .coefficient)
      LeftBound177801.bound (LeftBound177801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177805 .coefficient)
      LeftBound177335.bound (LeftBound177335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events692.exact177342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177801.bound, LeftBound177335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177801.bound, LeftBound177335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177801.actual selector witness, LeftBound177335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177806

namespace LeftBound177807
def owner : Owner := ⟨.program ⟨257⟩, ⟨23994⟩⟩
def transferEvent : Nat := 177807
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177803 .summary, .result 177342 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177803 .summary)
      LeftBound177802.bound (LeftBound177802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20774⟩⟩) (rawTerms := some (Proof.Events694.exact177803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177342 .summary)
      LeftBound177337.bound (LeftBound177337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23993⟩⟩) (rawTerms := some (Proof.Events692.exact177342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177802.bound, LeftBound177337.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177802.bound, LeftBound177337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177802.actual selector witness, LeftBound177337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177807

namespace LeftBound177811
def owner : Owner := ⟨.program ⟨257⟩, ⟨34014⟩⟩
def transferEvent : Nat := 177811
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177809 .coefficient, .predecessor 1 177810 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177809 .coefficient)
      LeftBound177806.bound (LeftBound177806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177810 .coefficient)
      LeftBound177123.bound (LeftBound177123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events691.exact177130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177806.bound, LeftBound177123.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177806.bound, LeftBound177123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177806.actual selector witness, LeftBound177123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177811

namespace LeftBound177812
def owner : Owner := ⟨.program ⟨257⟩, ⟨34014⟩⟩
def transferEvent : Nat := 177812
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177808 .summary, .result 177130 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177808 .summary)
      LeftBound177807.bound (LeftBound177807.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23994⟩⟩) (rawTerms := some (Proof.Events694.exact177808RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177130 .summary)
      LeftBound177125.bound (LeftBound177125.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34013⟩⟩) (rawTerms := some (Proof.Events691.exact177130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177125.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177807.bound, LeftBound177125.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177807.bound, LeftBound177125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177807.actual selector witness, LeftBound177125.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177812

namespace LeftBound177816
def owner : Owner := ⟨.program ⟨257⟩, ⟨53074⟩⟩
def transferEvent : Nat := 177816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177814 .coefficient, .predecessor 1 177815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177814 .coefficient)
      LeftBound177811.bound (LeftBound177811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177815 .coefficient)
      LeftBound176911.bound (LeftBound176911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events691.exact176918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound176911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound176911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177811.bound, LeftBound176911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177811.bound, LeftBound176911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177811.actual selector witness, LeftBound176911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177816

namespace LeftBound177817
def owner : Owner := ⟨.program ⟨257⟩, ⟨53074⟩⟩
def transferEvent : Nat := 177817
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177813 .summary, .result 176918 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177813 .summary)
      LeftBound177812.bound (LeftBound177812.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34014⟩⟩) (rawTerms := some (Proof.Events694.exact177813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 176918 .summary)
      LeftBound176913.bound (LeftBound176913.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53073⟩⟩) (rawTerms := some (Proof.Events691.exact176918RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound176913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177812.bound, LeftBound176913.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177812.bound, LeftBound176913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177812.actual selector witness, LeftBound176913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177817

namespace LeftBound177821
def owner : Owner := ⟨.program ⟨257⟩, ⟨56054⟩⟩
def transferEvent : Nat := 177821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177819 .coefficient, .predecessor 1 177820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177819 .coefficient)
      LeftBound177816.bound (LeftBound177816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177820 .coefficient)
      LeftBound176699.bound (LeftBound176699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events690.exact176706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound176699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound176699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177816.bound, LeftBound176699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177816.bound, LeftBound176699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177816.actual selector witness, LeftBound176699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177821

namespace LeftBound177822
def owner : Owner := ⟨.program ⟨257⟩, ⟨56054⟩⟩
def transferEvent : Nat := 177822
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177818 .summary, .result 176706 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177818 .summary)
      LeftBound177817.bound (LeftBound177817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53074⟩⟩) (rawTerms := some (Proof.Events694.exact177818RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 176706 .summary)
      LeftBound176701.bound (LeftBound176701.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56053⟩⟩) (rawTerms := some (Proof.Events690.exact176706RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound176701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177817.bound, LeftBound176701.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177817.bound, LeftBound176701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177817.actual selector witness, LeftBound176701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177822

namespace LeftBound177826
def owner : Owner := ⟨.program ⟨257⟩, ⟨59034⟩⟩
def transferEvent : Nat := 177826
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177824 .coefficient, .predecessor 1 177825 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177824 .coefficient)
      LeftBound177821.bound (LeftBound177821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177825 .coefficient)
      LeftBound176487.bound (LeftBound176487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events689.exact176494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound176487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound176487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177821.bound, LeftBound176487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177821.bound, LeftBound176487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177821.actual selector witness, LeftBound176487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177826

namespace LeftBound177827
def owner : Owner := ⟨.program ⟨257⟩, ⟨59034⟩⟩
def transferEvent : Nat := 177827
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177823 .summary, .result 176494 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177823 .summary)
      LeftBound177822.bound (LeftBound177822.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56054⟩⟩) (rawTerms := some (Proof.Events694.exact177823RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 176494 .summary)
      LeftBound176489.bound (LeftBound176489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59033⟩⟩) (rawTerms := some (Proof.Events689.exact176494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound176489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177822.bound, LeftBound176489.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177822.bound, LeftBound176489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177822.actual selector witness, LeftBound176489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177827

namespace LeftBound177831
def owner : Owner := ⟨.program ⟨257⟩, ⟨62014⟩⟩
def transferEvent : Nat := 177831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177829 .coefficient, .predecessor 1 177830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177829 .coefficient)
      LeftBound177826.bound (LeftBound177826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177830 .coefficient)
      LeftBound176275.bound (LeftBound176275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events688.exact176282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound176275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound176275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177826.bound, LeftBound176275.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177826.bound, LeftBound176275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177826.actual selector witness, LeftBound176275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177831

namespace LeftBound177832
def owner : Owner := ⟨.program ⟨257⟩, ⟨62014⟩⟩
def transferEvent : Nat := 177832
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 177828 .summary, .result 176282 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 177828 .summary)
      LeftBound177827.bound (LeftBound177827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59034⟩⟩) (rawTerms := some (Proof.Events694.exact177828RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound177827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 176282 .summary)
      LeftBound176277.bound (LeftBound176277.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62013⟩⟩) (rawTerms := some (Proof.Events688.exact176282RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound176277.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177827.bound, LeftBound176277.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177827.bound, LeftBound176277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177827.actual selector witness, LeftBound176277.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177832

namespace LeftBound177836
def owner : Owner := ⟨.program ⟨257⟩, ⟨64994⟩⟩
def transferEvent : Nat := 177836
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 177834 .coefficient, .predecessor 1 177835 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 177834 .coefficient)
      LeftBound177831.bound (LeftBound177831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events694.exact177833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound177831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound177831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 177835 .coefficient)
      LeftBound176063.bound (LeftBound176063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events687.exact176070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound176063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound176063.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound177831.bound, LeftBound176063.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound177831.bound, LeftBound176063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound177831.actual selector witness, LeftBound176063.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound177836

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
