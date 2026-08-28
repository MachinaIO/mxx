import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftAuthority92224
def owner : Owner := ⟨.program ⟨214⟩, ⟨28076⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 92224
def resultEvent : Nat := 92225
def frameStart : Nat := 92174
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28076⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92224

namespace LeftAuthority92226
def owner : Owner := ⟨.program ⟨214⟩, ⟨110⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .exactZero
def producerEvent : Nat := 92226
def resultEvent : Nat := 92227
def frameStart : Nat := 92174
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
end LeftAuthority92226

namespace LeftAuthority92235
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def authority : Authority := .factStore
def bound : CoeffClass := .large
def producerEvent : Nat := 92235
def resultEvent : Nat := 92236
def frameStart : Nat := 92174
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
end LeftAuthority92235

namespace LeftAuthority92243
def owner : Owner := ⟨.program ⟨214⟩, ⟨6698⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 92243
def resultEvent : Nat := 92244
def frameStart : Nat := 92174
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92243

namespace LeftAuthority92258
def owner : Owner := ⟨.program ⟨214⟩, ⟨18035⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨22, by decide⟩
def producerEvent : Nat := 92258
def resultEvent : Nat := 92259
def frameStart : Nat := 92174
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18035⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 22) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92258

namespace LeftAuthority92266
def owner : Owner := ⟨.program ⟨214⟩, ⟨6724⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 92266
def resultEvent : Nat := 92267
def frameStart : Nat := 92174
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6724⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92266

namespace LeftAuthority92304
def owner : Owner := ⟨.program ⟨214⟩, ⟨24161⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 92304
def resultEvent : Nat := 92305
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24161⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92304

namespace LeftAuthority92307
def owner : Owner := ⟨.program ⟨214⟩, ⟨27859⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 92307
def resultEvent : Nat := 92308
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27859⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92307

namespace LeftAuthority92320
def owner : Owner := ⟨.program ⟨214⟩, ⟨21328⟩⟩
def authority : Authority := .relationPreimageSource ⟨42⟩
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
def producerEvent : Nat := 92320
def resultEvent : Nat := 92321
def frameStart : Nat := 0
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21328⟩⟩] } }]) (recordedCoefficientBound := .finite 136065468) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92320

namespace LeftAuthority92422
def owner : Owner := ⟨.program ⟨214⟩, ⟨15940⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨18, by decide⟩
def producerEvent : Nat := 92422
def resultEvent : Nat := 92423
def frameStart : Nat := 92386
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15940⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 18) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92422

namespace LeftAuthority92433
def owner : Owner := ⟨.program ⟨214⟩, ⟨24161⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 92433
def resultEvent : Nat := 92434
def frameStart : Nat := 92386
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24161⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92433

namespace LeftAuthority92436
def owner : Owner := ⟨.program ⟨214⟩, ⟨27859⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .finite ⟨8192, by decide⟩
def producerEvent : Nat := 92436
def resultEvent : Nat := 92437
def frameStart : Nat := 92386
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27859⟩⟩] } }]) (recordedCoefficientBound := .finite 8192) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92436

namespace LeftAuthority92438
def owner : Owner := ⟨.program ⟨214⟩, ⟨110⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .exactZero
def producerEvent : Nat := 92438
def resultEvent : Nat := 92439
def frameStart : Nat := 92386
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
end LeftAuthority92438

namespace LeftAuthority92447
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def authority : Authority := .factStore
def bound : CoeffClass := .large
def producerEvent : Nat := 92447
def resultEvent : Nat := 92448
def frameStart : Nat := 92386
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
end LeftAuthority92447

namespace LeftAuthority92455
def owner : Owner := ⟨.program ⟨214⟩, ⟨6697⟩⟩
def authority : Authority := .operator
def bound : CoeffClass := .large
def producerEvent : Nat := 92455
def resultEvent : Nat := 92456
def frameStart : Nat := 92386
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }]) (recordedCoefficientBound := .large) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92455

namespace LeftAuthority92470
def owner : Owner := ⟨.program ⟨214⟩, ⟨17165⟩⟩
def authority : Authority := .programFamilyFact
def bound : CoeffClass := .finite ⟨18, by decide⟩
def producerEvent : Nat := 92470
def resultEvent : Nat := 92471
def frameStart : Nat := 92386
theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by
  exact .resultExact (terms := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17165⟩⟩], orderedFactors := [] } }]) (recordedCoefficientBound := .finite 18) (summary := .exactZero) (summaryProducer := none) (by decide) (by rfl) (by rfl) (by simp [bound, RecordedBoundRefines])
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat :=
  witness.authorityMagnitude resultEvent
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
      (actual selector witness) := by
  exact .authority witness.toAuthorityWitness leaf
end LeftAuthority92470

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
