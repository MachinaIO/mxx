import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard002
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard003

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound1491
def owner : Owner := ⟨.program ⟨257⟩, ⟨32274⟩⟩
def transferEvent : Nat := 1491
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1489 .coefficient, .predecessor 1 1490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1489 .coefficient)
      LeftBound1487.bound (LeftBound1487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1490 .coefficient)
      LeftBound1450.bound (LeftBound1450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1487.bound, LeftBound1450.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1487.bound, LeftBound1450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1487.actual selector witness, LeftBound1450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1491

namespace LeftBound1495
def owner : Owner := ⟨.program ⟨257⟩, ⟨51338⟩⟩
def transferEvent : Nat := 1495
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1493 .coefficient, .predecessor 1 1494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1493 .coefficient)
      LeftBound1491.bound (LeftBound1491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1494 .coefficient)
      LeftBound1442.bound (LeftBound1442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1491.bound, LeftBound1442.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1491.bound, LeftBound1442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1491.actual selector witness, LeftBound1442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1495

namespace LeftBound1499
def owner : Owner := ⟨.program ⟨257⟩, ⟨54318⟩⟩
def transferEvent : Nat := 1499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1497 .coefficient, .predecessor 1 1498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1497 .coefficient)
      LeftBound1495.bound (LeftBound1495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1498 .coefficient)
      LeftBound1434.bound (LeftBound1434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1495.bound, LeftBound1434.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1495.bound, LeftBound1434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1495.actual selector witness, LeftBound1434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1499

namespace LeftBound1503
def owner : Owner := ⟨.program ⟨257⟩, ⟨57298⟩⟩
def transferEvent : Nat := 1503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1501 .coefficient, .predecessor 1 1502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1501 .coefficient)
      LeftBound1499.bound (LeftBound1499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1502 .coefficient)
      LeftBound1426.bound (LeftBound1426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1499.bound, LeftBound1426.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1499.bound, LeftBound1426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1499.actual selector witness, LeftBound1426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1503

namespace LeftBound1507
def owner : Owner := ⟨.program ⟨257⟩, ⟨60278⟩⟩
def transferEvent : Nat := 1507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1505 .coefficient, .predecessor 1 1506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1505 .coefficient)
      LeftBound1503.bound (LeftBound1503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1506 .coefficient)
      LeftBound1418.bound (LeftBound1418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1503.bound, LeftBound1418.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1503.bound, LeftBound1418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1503.actual selector witness, LeftBound1418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1507

namespace LeftBound1511
def owner : Owner := ⟨.program ⟨257⟩, ⟨63258⟩⟩
def transferEvent : Nat := 1511
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1509 .coefficient, .predecessor 1 1510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1509 .coefficient)
      LeftBound1507.bound (LeftBound1507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1510 .coefficient)
      LeftBound1410.bound (LeftBound1410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1507.bound, LeftBound1410.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1507.bound, LeftBound1410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1507.actual selector witness, LeftBound1410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1511

namespace LeftBound1515
def owner : Owner := ⟨.program ⟨257⟩, ⟨67220⟩⟩
def transferEvent : Nat := 1515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1513 .coefficient, .predecessor 1 1514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1513 .coefficient)
      LeftBound1511.bound (LeftBound1511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1514 .coefficient)
      LeftBound1402.bound (LeftBound1402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1511.bound, LeftBound1402.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1511.bound, LeftBound1402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1511.actual selector witness, LeftBound1402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1515

namespace LeftBound1519
def owner : Owner := ⟨.program ⟨257⟩, ⟨67221⟩⟩
def transferEvent : Nat := 1519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1517 .coefficient, .predecessor 1 1518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1517 .coefficient)
      LeftBound1515.bound (LeftBound1515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1518 .coefficient)
      LeftBound1394.bound (LeftBound1394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1515.bound, LeftBound1394.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1515.bound, LeftBound1394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1515.actual selector witness, LeftBound1394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1519

namespace LeftBound1523
def owner : Owner := ⟨.program ⟨257⟩, ⟨67222⟩⟩
def transferEvent : Nat := 1523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1521 .coefficient, .predecessor 1 1522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1521 .coefficient)
      LeftBound1519.bound (LeftBound1519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1522 .coefficient)
      LeftBound1386.bound (LeftBound1386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1519.bound, LeftBound1386.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1519.bound, LeftBound1386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1519.actual selector witness, LeftBound1386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1523

namespace LeftBound1527
def owner : Owner := ⟨.program ⟨257⟩, ⟨67223⟩⟩
def transferEvent : Nat := 1527
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1525 .coefficient, .predecessor 1 1526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1525 .coefficient)
      LeftBound1523.bound (LeftBound1523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1526 .coefficient)
      LeftBound1378.bound (LeftBound1378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1523.bound, LeftBound1378.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1523.bound, LeftBound1378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1523.actual selector witness, LeftBound1378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1527

namespace LeftBound1531
def owner : Owner := ⟨.program ⟨257⟩, ⟨67224⟩⟩
def transferEvent : Nat := 1531
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1529 .coefficient, .predecessor 1 1530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1529 .coefficient)
      LeftBound1527.bound (LeftBound1527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1530 .coefficient)
      LeftBound1370.bound (LeftBound1370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1527.bound, LeftBound1370.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1527.bound, LeftBound1370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1527.actual selector witness, LeftBound1370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1531

namespace LeftBound1535
def owner : Owner := ⟨.program ⟨257⟩, ⟨67225⟩⟩
def transferEvent : Nat := 1535
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1533 .coefficient, .predecessor 1 1534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1533 .coefficient)
      LeftBound1531.bound (LeftBound1531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1534 .coefficient)
      LeftBound1362.bound (LeftBound1362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1531.bound, LeftBound1362.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1531.bound, LeftBound1362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1531.actual selector witness, LeftBound1362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1535

namespace LeftBound1539
def owner : Owner := ⟨.program ⟨257⟩, ⟨67226⟩⟩
def transferEvent : Nat := 1539
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1537 .coefficient, .predecessor 1 1538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1537 .coefficient)
      LeftBound1535.bound (LeftBound1535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1538 .coefficient)
      LeftBound1354.bound (LeftBound1354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1535.bound, LeftBound1354.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1535.bound, LeftBound1354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1535.actual selector witness, LeftBound1354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1539

namespace LeftBound1543
def owner : Owner := ⟨.program ⟨257⟩, ⟨67227⟩⟩
def transferEvent : Nat := 1543
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1541 .coefficient, .predecessor 1 1542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1541 .coefficient)
      LeftBound1539.bound (LeftBound1539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1542 .coefficient)
      LeftBound1346.bound (LeftBound1346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1539.bound, LeftBound1346.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1539.bound, LeftBound1346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1539.actual selector witness, LeftBound1346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1543

namespace LeftBound1547
def owner : Owner := ⟨.program ⟨257⟩, ⟨67228⟩⟩
def transferEvent : Nat := 1547
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1545 .coefficient, .predecessor 1 1546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1545 .coefficient)
      LeftBound1543.bound (LeftBound1543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1546 .coefficient)
      LeftBound1338.bound (LeftBound1338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1543.bound, LeftBound1338.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1543.bound, LeftBound1338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1543.actual selector witness, LeftBound1338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1547

namespace LeftBound1551
def owner : Owner := ⟨.program ⟨257⟩, ⟨67650⟩⟩
def transferEvent : Nat := 1551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 1549 .coefficient, .predecessor 1 1550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1549 .coefficient)
      LeftBound1547.bound (LeftBound1547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1550 .coefficient)
      LeftBound1330.bound (LeftBound1330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound1330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound1330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound1547.bound, LeftBound1330.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound1547.bound, LeftBound1330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound1547.actual selector witness, LeftBound1330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound1551

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
