import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard029

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult3724
def owner : Owner := ⟨.program ⟨214⟩, ⟨14883⟩⟩
def rawTerms : List Term := Proof.Events014.exact3724RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3724.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3721) (rightBinding := 3722)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6379⟩) (rightExpression := ⟨14882⟩)
    (transferEvent := 3723)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3720.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3724

namespace SemanticResult3728
def owner : Owner := ⟨.program ⟨214⟩, ⟨15044⟩⟩
def rawTerms : List Term := Proof.Events014.exact3728RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3728.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3725) (rightBinding := 3726)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14883⟩) (rightExpression := ⟨15043⟩)
    (transferEvent := 3727)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3724.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3712.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3728

namespace SemanticResult3732
def owner : Owner := ⟨.program ⟨214⟩, ⟨15205⟩⟩
def rawTerms : List Term := Proof.Events014.exact3732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3729) (rightBinding := 3730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15044⟩) (rightExpression := ⟨15204⟩)
    (transferEvent := 3731)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3704.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3732

namespace SemanticResult3736
def owner : Owner := ⟨.program ⟨214⟩, ⟨15513⟩⟩
def rawTerms : List Term := Proof.Events014.exact3736RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3736
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3736.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3733) (rightBinding := 3734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15205⟩) (rightExpression := ⟨15512⟩)
    (transferEvent := 3735)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3696.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3736

namespace SemanticResult3740
def owner : Owner := ⟨.program ⟨214⟩, ⟨17808⟩⟩
def rawTerms : List Term := Proof.Events014.exact3740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3737) (rightBinding := 3738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15513⟩) (rightExpression := ⟨17807⟩)
    (transferEvent := 3739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3688.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3740

namespace SemanticResult3744
def owner : Owner := ⟨.program ⟨214⟩, ⟨17809⟩⟩
def rawTerms : List Term := Proof.Events014.exact3744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3741) (rightBinding := 3742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17808⟩) (rightExpression := ⟨17435⟩)
    (transferEvent := 3743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3680.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3744

namespace SemanticResult3748
def owner : Owner := ⟨.program ⟨214⟩, ⟨17810⟩⟩
def rawTerms : List Term := Proof.Events014.exact3748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3745) (rightBinding := 3746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17809⟩) (rightExpression := ⟨17218⟩)
    (transferEvent := 3747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3672.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3748

namespace SemanticResult3752
def owner : Owner := ⟨.program ⟨214⟩, ⟨17811⟩⟩
def rawTerms : List Term := Proof.Events014.exact3752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3749) (rightBinding := 3750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17810⟩) (rightExpression := ⟨17162⟩)
    (transferEvent := 3751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3664.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3752

namespace SemanticResult3756
def owner : Owner := ⟨.program ⟨214⟩, ⟨18030⟩⟩
def rawTerms : List Term := Proof.Events014.exact3756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3753) (rightBinding := 3754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17811⟩) (rightExpression := ⟨18029⟩)
    (transferEvent := 3755)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3656.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3756

namespace SemanticResult3760
def owner : Owner := ⟨.program ⟨214⟩, ⟨18031⟩⟩
def rawTerms : List Term := Proof.Events014.exact3760RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3760.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3757) (rightBinding := 3758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18030⟩) (rightExpression := ⟨17659⟩)
    (transferEvent := 3759)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3648.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3760

namespace SemanticResult3764
def owner : Owner := ⟨.program ⟨214⟩, ⟨18032⟩⟩
def rawTerms : List Term := Proof.Events014.exact3764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3764.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3761) (rightBinding := 3762)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18031⟩) (rightExpression := ⟨17603⟩)
    (transferEvent := 3763)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3760.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3640.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3764

namespace SemanticResult3768
def owner : Owner := ⟨.program ⟨214⟩, ⟨18820⟩⟩
def rawTerms : List Term := Proof.Events014.exact3768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3765) (rightBinding := 3766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18032⟩) (rightExpression := ⟨18819⟩)
    (transferEvent := 3767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3632.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3768

namespace SemanticResult3772
def owner : Owner := ⟨.program ⟨214⟩, ⟨18821⟩⟩
def rawTerms : List Term := Proof.Events014.exact3772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3769) (rightBinding := 3770)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18820⟩) (rightExpression := ⟨17547⟩)
    (transferEvent := 3771)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3768.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3772

namespace SemanticResult3776
def owner : Owner := ⟨.program ⟨214⟩, ⟨18822⟩⟩
def rawTerms : List Term := Proof.Events014.exact3776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3776.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3773) (rightBinding := 3774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18821⟩) (rightExpression := ⟨17946⟩)
    (transferEvent := 3775)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3616.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3776

namespace SemanticResult3780
def owner : Owner := ⟨.program ⟨214⟩, ⟨18823⟩⟩
def rawTerms : List Term := Proof.Events014.exact3780RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3780.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3777) (rightBinding := 3778)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18822⟩) (rightExpression := ⟨17715⟩)
    (transferEvent := 3779)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3776.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3608.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3780

namespace SemanticResult3784
def owner : Owner := ⟨.program ⟨214⟩, ⟨18824⟩⟩
def rawTerms : List Term := Proof.Events014.exact3784RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 3784
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult3784.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 3781) (rightBinding := 3782)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18823⟩) (rightExpression := ⟨17491⟩)
    (transferEvent := 3783)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult3780.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3600.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult3784

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
