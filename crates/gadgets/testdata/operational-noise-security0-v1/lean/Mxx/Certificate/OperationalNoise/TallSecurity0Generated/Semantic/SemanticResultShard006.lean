import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard004
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard005

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult716
def owner : Owner := ⟨.program ⟨214⟩, ⟨14906⟩⟩
def rawTerms : List Term := Proof.Events002.exact716RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 716
def producerEvent : Nat := 715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult716.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 0, .finite 2, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult716

namespace SemanticResult721
def owner : Owner := ⟨.program ⟨214⟩, ⟨14907⟩⟩
def rawTerms : List Term := Proof.Events002.exact721RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 721
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult721.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge720.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge720.frameStart)
    (transferEvent := 719) (owner := owner)
    (leftResult := 716) (rightResult := 713)
    (working := LeftOperatorMerge720.working)
    (reconstruction := LeftOperatorMerge720.reconstruction)
    (leftReference := .predecessor 0 717 .coefficient) (rightReference := .predecessor 1 718 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult716.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge720.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult721

namespace SemanticResult723
def owner : Owner := ⟨.program ⟨214⟩, ⟨6378⟩⟩
def rawTerms : List Term := Proof.Events002.exact723RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 723
def producerEvent : Nat := 722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult723.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 0, .finite 1, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult723

namespace SemanticResult728
def owner : Owner := ⟨.program ⟨214⟩, ⟨6379⟩⟩
def rawTerms : List Term := Proof.Events002.exact728RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult728.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge727.working .exactZero) := by
  apply operatorSubMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge727.frameStart)
    (transferEvent := 726) (owner := owner)
    (leftResult := 723) (rightResult := 723)
    (working := LeftOperatorMerge727.working)
    (reconstruction := LeftOperatorMerge727.reconstruction)
    (leftReference := .predecessor 0 724 .coefficient) (rightReference := .predecessor 1 725 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge727.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult728

namespace SemanticResult732
def owner : Owner := ⟨.program ⟨214⟩, ⟨14908⟩⟩
def rawTerms : List Term := Proof.Events002.exact732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 729) (rightBinding := 730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6379⟩) (rightExpression := ⟨14907⟩)
    (transferEvent := 731)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult721.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult732

namespace SemanticResult736
def owner : Owner := ⟨.program ⟨214⟩, ⟨15069⟩⟩
def rawTerms : List Term := Proof.Events002.exact736RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 736
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult736.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 733) (rightBinding := 734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14908⟩) (rightExpression := ⟨15068⟩)
    (transferEvent := 735)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult711.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult736

namespace SemanticResult740
def owner : Owner := ⟨.program ⟨214⟩, ⟨15230⟩⟩
def rawTerms : List Term := Proof.Events002.exact740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 737) (rightBinding := 738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15069⟩) (rightExpression := ⟨15229⟩)
    (transferEvent := 739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult701.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult740

namespace SemanticResult744
def owner : Owner := ⟨.program ⟨214⟩, ⟨15538⟩⟩
def rawTerms : List Term := Proof.Events002.exact744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 741) (rightBinding := 742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15230⟩) (rightExpression := ⟨15537⟩)
    (transferEvent := 743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult691.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult744

namespace SemanticResult748
def owner : Owner := ⟨.program ⟨214⟩, ⟨17848⟩⟩
def rawTerms : List Term := Proof.Events002.exact748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 745) (rightBinding := 746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15538⟩) (rightExpression := ⟨17847⟩)
    (transferEvent := 747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult681.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult748

namespace SemanticResult752
def owner : Owner := ⟨.program ⟨214⟩, ⟨17849⟩⟩
def rawTerms : List Term := Proof.Events002.exact752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 749) (rightBinding := 750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17848⟩) (rightExpression := ⟨17455⟩)
    (transferEvent := 751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult671.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult752

namespace SemanticResult756
def owner : Owner := ⟨.program ⟨214⟩, ⟨17850⟩⟩
def rawTerms : List Term := Proof.Events002.exact756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 753) (rightBinding := 754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17849⟩) (rightExpression := ⟨17238⟩)
    (transferEvent := 755)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult661.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult756

namespace SemanticResult760
def owner : Owner := ⟨.program ⟨214⟩, ⟨17851⟩⟩
def rawTerms : List Term := Proof.Events002.exact760RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult760.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 757) (rightBinding := 758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17850⟩) (rightExpression := ⟨17182⟩)
    (transferEvent := 759)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult651.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult760

namespace SemanticResult764
def owner : Owner := ⟨.program ⟨214⟩, ⟨18065⟩⟩
def rawTerms : List Term := Proof.Events002.exact764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult764.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 761) (rightBinding := 762)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17851⟩) (rightExpression := ⟨18064⟩)
    (transferEvent := 763)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult760.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult764

namespace SemanticResult768
def owner : Owner := ⟨.program ⟨214⟩, ⟨18066⟩⟩
def rawTerms : List Term := Proof.Events003.exact768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 765) (rightBinding := 766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18065⟩) (rightExpression := ⟨17679⟩)
    (transferEvent := 767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult631.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult768

namespace SemanticResult772
def owner : Owner := ⟨.program ⟨214⟩, ⟨18067⟩⟩
def rawTerms : List Term := Proof.Events003.exact772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 769) (rightBinding := 770)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18066⟩) (rightExpression := ⟨17623⟩)
    (transferEvent := 771)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult768.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult621.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult772

namespace SemanticResult776
def owner : Owner := ⟨.program ⟨214⟩, ⟨18895⟩⟩
def rawTerms : List Term := Proof.Events003.exact776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult776.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 773) (rightBinding := 774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18067⟩) (rightExpression := ⟨18894⟩)
    (transferEvent := 775)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult611.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult776

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
