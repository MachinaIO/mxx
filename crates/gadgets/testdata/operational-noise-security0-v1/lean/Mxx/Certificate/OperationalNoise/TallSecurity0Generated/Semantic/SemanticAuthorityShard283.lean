import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftAuthority91247
def owner : Owner := ⟨.program ⟨214⟩, ⟨28944⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 91247
def resultEvent : Nat := 91248
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28944⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91247

namespace LeftAuthority91260
def owner : Owner := ⟨.program ⟨214⟩, ⟨22048⟩⟩
def authority : Authority := .relationPreimageSource ⟨53⟩
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
def producerEvent : Nat := 91260
def resultEvent : Nat := 91261
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22048⟩⟩] } }]) (recordedCoefficientBound := .finite 136065468) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91260

namespace LeftAuthority91362
def owner : Owner := ⟨.program ⟨214⟩, ⟨16465⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨40, by decide⟩
def producerEvent : Nat := 91362
def resultEvent : Nat := 91363
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16465⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 40) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91362

namespace LeftAuthority91373
def owner : Owner := ⟨.program ⟨214⟩, ⟨24476⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 91373
def resultEvent : Nat := 91374
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24476⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91373

namespace LeftAuthority91376
def owner : Owner := ⟨.program ⟨214⟩, ⟨28944⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 91376
def resultEvent : Nat := 91377
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28944⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91376

namespace LeftAuthority91378
def owner : Owner := ⟨.program ⟨214⟩, ⟨110⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .exactZero
def producerEvent : Nat := 91378
def resultEvent : Nat := 91379
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultCoefficient (by decide) (by rfl) (by rfl)
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91378

namespace LeftAuthority91387
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def authority : Authority := .factStore
def bound : CoeffClass := .large
def producerEvent : Nat := 91387
def resultEvent : Nat := 91388
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91387

namespace LeftAuthority91395
def owner : Owner := ⟨.program ⟨214⟩, ⟨6702⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 91395
def resultEvent : Nat := 91396
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91395

namespace LeftAuthority91410
def owner : Owner := ⟨.program ⟨214⟩, ⟨17550⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨40, by decide⟩
def producerEvent : Nat := 91410
def resultEvent : Nat := 91411
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 40) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91410

namespace LeftAuthority91418
def owner : Owner := ⟨.program ⟨214⟩, ⟨6732⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 91418
def resultEvent : Nat := 91419
def frameStart : Nat := 91326
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91418

namespace LeftAuthority91456
def owner : Owner := ⟨.program ⟨214⟩, ⟨24413⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 91456
def resultEvent : Nat := 91457
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91456

namespace LeftAuthority91459
def owner : Owner := ⟨.program ⟨214⟩, ⟨28727⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 91459
def resultEvent : Nat := 91460
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91459

namespace LeftAuthority91472
def owner : Owner := ⟨.program ⟨214⟩, ⟨21904⟩⟩
def authority : Authority := .relationPreimageSource ⟨51⟩
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
def producerEvent : Nat := 91472
def resultEvent : Nat := 91473
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21904⟩⟩] } }]) (recordedCoefficientBound := .finite 136065468) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91472

namespace LeftAuthority91574
def owner : Owner := ⟨.program ⟨214⟩, ⟨16381⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨36, by decide⟩
def producerEvent : Nat := 91574
def resultEvent : Nat := 91575
def frameStart : Nat := 91538
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 36) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91574

namespace LeftAuthority91585
def owner : Owner := ⟨.program ⟨214⟩, ⟨24413⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 91585
def resultEvent : Nat := 91586
def frameStart : Nat := 91538
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91585

namespace LeftAuthority91588
def owner : Owner := ⟨.program ⟨214⟩, ⟨28727⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 91588
def resultEvent : Nat := 91589
def frameStart : Nat := 91538
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority91588

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
