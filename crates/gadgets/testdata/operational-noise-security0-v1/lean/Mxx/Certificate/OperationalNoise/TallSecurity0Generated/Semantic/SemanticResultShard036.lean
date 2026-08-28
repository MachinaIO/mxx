import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard034
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard035

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult4470
def owner : Owner := ⟨.program ⟨214⟩, ⟨15049⟩⟩
def rawTerms : List Term := Proof.Events017.exact4470RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4470.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4467) (rightBinding := 4468)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14888⟩) (rightExpression := ⟨15048⟩)
    (transferEvent := 4469)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4466.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4454.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4470

namespace SemanticResult4474
def owner : Owner := ⟨.program ⟨214⟩, ⟨15210⟩⟩
def rawTerms : List Term := Proof.Events017.exact4474RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4474
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4474.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4471) (rightBinding := 4472)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15049⟩) (rightExpression := ⟨15209⟩)
    (transferEvent := 4473)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4470.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4446.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4474

namespace SemanticResult4478
def owner : Owner := ⟨.program ⟨214⟩, ⟨15518⟩⟩
def rawTerms : List Term := Proof.Events017.exact4478RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4478.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4475) (rightBinding := 4476)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15210⟩) (rightExpression := ⟨15517⟩)
    (transferEvent := 4477)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4474.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4438.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4478

namespace SemanticResult4482
def owner : Owner := ⟨.program ⟨214⟩, ⟨17816⟩⟩
def rawTerms : List Term := Proof.Events017.exact4482RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4482
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4482.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4479) (rightBinding := 4480)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15518⟩) (rightExpression := ⟨17815⟩)
    (transferEvent := 4481)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4478.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4430.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4482

namespace SemanticResult4486
def owner : Owner := ⟨.program ⟨214⟩, ⟨17817⟩⟩
def rawTerms : List Term := Proof.Events017.exact4486RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4486
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4486.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4483) (rightBinding := 4484)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17816⟩) (rightExpression := ⟨17439⟩)
    (transferEvent := 4485)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4482.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4422.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4486

namespace SemanticResult4490
def owner : Owner := ⟨.program ⟨214⟩, ⟨17818⟩⟩
def rawTerms : List Term := Proof.Events017.exact4490RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4490.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4487) (rightBinding := 4488)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17817⟩) (rightExpression := ⟨17222⟩)
    (transferEvent := 4489)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4486.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4414.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4490

namespace SemanticResult4494
def owner : Owner := ⟨.program ⟨214⟩, ⟨17819⟩⟩
def rawTerms : List Term := Proof.Events017.exact4494RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4494
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4494.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4491) (rightBinding := 4492)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17818⟩) (rightExpression := ⟨17166⟩)
    (transferEvent := 4493)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4490.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4406.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4494

namespace SemanticResult4498
def owner : Owner := ⟨.program ⟨214⟩, ⟨18037⟩⟩
def rawTerms : List Term := Proof.Events017.exact4498RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4498.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4495) (rightBinding := 4496)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17819⟩) (rightExpression := ⟨18036⟩)
    (transferEvent := 4497)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4494.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4398.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4498

namespace SemanticResult4502
def owner : Owner := ⟨.program ⟨214⟩, ⟨18038⟩⟩
def rawTerms : List Term := Proof.Events017.exact4502RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4502
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4502.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4499) (rightBinding := 4500)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18037⟩) (rightExpression := ⟨17663⟩)
    (transferEvent := 4501)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4498.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4390.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4502

namespace SemanticResult4506
def owner : Owner := ⟨.program ⟨214⟩, ⟨18039⟩⟩
def rawTerms : List Term := Proof.Events017.exact4506RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4506
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4506.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4503) (rightBinding := 4504)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18038⟩) (rightExpression := ⟨17607⟩)
    (transferEvent := 4505)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4502.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4382.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4506

namespace SemanticResult4510
def owner : Owner := ⟨.program ⟨214⟩, ⟨18834⟩⟩
def rawTerms : List Term := Proof.Events017.exact4510RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4510.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4507) (rightBinding := 4508)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18039⟩) (rightExpression := ⟨18833⟩)
    (transferEvent := 4509)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4506.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4374.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4510

namespace SemanticResult4514
def owner : Owner := ⟨.program ⟨214⟩, ⟨18835⟩⟩
def rawTerms : List Term := Proof.Events017.exact4514RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4514.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4511) (rightBinding := 4512)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18834⟩) (rightExpression := ⟨17551⟩)
    (transferEvent := 4513)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4510.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4366.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4514

namespace SemanticResult4518
def owner : Owner := ⟨.program ⟨214⟩, ⟨18836⟩⟩
def rawTerms : List Term := Proof.Events017.exact4518RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4518.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4515) (rightBinding := 4516)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18835⟩) (rightExpression := ⟨17950⟩)
    (transferEvent := 4517)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4514.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4358.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4518

namespace SemanticResult4522
def owner : Owner := ⟨.program ⟨214⟩, ⟨18837⟩⟩
def rawTerms : List Term := Proof.Events017.exact4522RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4522
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4522.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4519) (rightBinding := 4520)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18836⟩) (rightExpression := ⟨17719⟩)
    (transferEvent := 4521)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4518.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4350.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4522

namespace SemanticResult4526
def owner : Owner := ⟨.program ⟨214⟩, ⟨18838⟩⟩
def rawTerms : List Term := Proof.Events017.exact4526RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4526.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4523) (rightBinding := 4524)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18837⟩) (rightExpression := ⟨17495⟩)
    (transferEvent := 4525)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4522.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4342.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4526

namespace SemanticResult4530
def owner : Owner := ⟨.program ⟨214⟩, ⟨18839⟩⟩
def rawTerms : List Term := Proof.Events017.exact4530RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult4530.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4527) (rightBinding := 4528)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18838⟩) (rightExpression := ⟨16928⟩)
    (transferEvent := 4529)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4526.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4334.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4530

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
