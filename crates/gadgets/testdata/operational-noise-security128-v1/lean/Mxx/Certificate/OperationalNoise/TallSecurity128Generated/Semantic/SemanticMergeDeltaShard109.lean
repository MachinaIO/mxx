import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21659
def owner : Owner := ⟨.program ⟨257⟩, ⟨62239⟩⟩
def mergeEvent : Nat := 21659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21655RawTerms
def rightRaw : List Term := Proof.Events084.exact21612RawTerms
def group : MergeGroup := .operator 21655 21612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21655) (leftOrdinal := 1)
    (rightResult := 21612) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21659

namespace LeftMerge21667
def owner : Owner := ⟨.program ⟨257⟩, ⟨64344⟩⟩
def mergeEvent : Nat := 21667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21661RawTerms
def rightRaw : List Term := Proof.Events084.exact21578RawTerms
def group : MergeGroup := .operator 21661 21578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21661) (leftOrdinal := 1)
    (rightResult := 21578) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21667

namespace LeftMerge21669
def owner : Owner := ⟨.program ⟨257⟩, ⟨64344⟩⟩
def mergeEvent : Nat := 21669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21575RawTerms
def group : MergeGroup := .relation 21668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21668) (rhsResult := 21575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64343⟩⟩) ⟨63877⟩ 21575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨63877⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21669

namespace LeftMerge21670
def owner : Owner := ⟨.program ⟨257⟩, ⟨64344⟩⟩
def mergeEvent : Nat := 21670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21661RawTerms
def rightRaw : List Term := Proof.Events084.exact21578RawTerms
def group : MergeGroup := .operator 21661 21578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21661) (leftOrdinal := 0)
    (rightResult := 21578) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21670

namespace LeftMerge21684
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def mergeEvent : Nat := 21684
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events084.exact21678RawTerms
def group : MergeGroup := .operator 17169 21678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 21678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21684

namespace LeftMerge21763
def owner : Owner := ⟨.program ⟨257⟩, ⟨62232⟩⟩
def mergeEvent : Nat := 21763
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events084.exact21759RawTerms
def rightRaw : List Term := Proof.Events084.exact21756RawTerms
def group : MergeGroup := .operator 21759 21756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21759) (leftOrdinal := 0)
    (rightResult := 21756) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21763

namespace LeftMerge21793
def owner : Owner := ⟨.program ⟨257⟩, ⟨64172⟩⟩
def mergeEvent : Nat := 21793
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21789RawTerms
def rightRaw : List Term := Proof.Events085.exact21787RawTerms
def group : MergeGroup := .operator 21789 21787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21789) (leftOrdinal := 0)
    (rightResult := 21787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21793

namespace LeftMerge21816
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 21816
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21812RawTerms
def rightRaw : List Term := Proof.Events085.exact21809RawTerms
def group : MergeGroup := .operator 21812 21809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21812) (leftOrdinal := 0)
    (rightResult := 21809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21816

namespace LeftMerge21825
def owner : Owner := ⟨.program ⟨257⟩, ⟨64346⟩⟩
def mergeEvent : Nat := 21825
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21821RawTerms
def rightRaw : List Term := Proof.Events085.exact21778RawTerms
def group : MergeGroup := .operator 21821 21778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21821) (leftOrdinal := 1)
    (rightResult := 21778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21825

namespace LeftMerge21827
def owner : Owner := ⟨.program ⟨257⟩, ⟨64346⟩⟩
def mergeEvent : Nat := 21827
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21775RawTerms
def group : MergeGroup := .relation 21826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21826) (rhsResult := 21775)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64343⟩⟩) ⟨63877⟩ 21775) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨63877⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21827

namespace LeftMerge21828
def owner : Owner := ⟨.program ⟨257⟩, ⟨64346⟩⟩
def mergeEvent : Nat := 21828
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21821RawTerms
def rightRaw : List Term := Proof.Events085.exact21778RawTerms
def group : MergeGroup := .operator 21821 21778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21821) (leftOrdinal := 0)
    (rightResult := 21778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64343⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21828

namespace LeftMerge21836
def owner : Owner := ⟨.program ⟨257⟩, ⟨62740⟩⟩
def mergeEvent : Nat := 21836
def frameStart : Nat := 21733
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21789RawTerms
def rightRaw : List Term := Proof.Events085.exact21832RawTerms
def group : MergeGroup := .operator 21789 21832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21789) (leftOrdinal := 0)
    (rightResult := 21832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21836

namespace LeftMerge21853
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def mergeEvent : Nat := 21853
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21850RawTerms
def group : MergeGroup := .relation 21852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21852) (rhsResult := 21850)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩) (none) 21850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨63877⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21853

namespace LeftMerge21854
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def mergeEvent : Nat := 21854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21850RawTerms
def group : MergeGroup := .relation 21852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21852) (rhsResult := 21850)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩) (none) 21850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21854

namespace LeftMerge21855
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def mergeEvent : Nat := 21855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21850RawTerms
def group : MergeGroup := .relation 21852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21852) (rhsResult := 21850)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩) (none) 21850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21855

namespace LeftMerge21856
def owner : Owner := ⟨.program ⟨257⟩, ⟨63285⟩⟩
def mergeEvent : Nat := 21856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21850RawTerms
def group : MergeGroup := .relation 21852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21852) (rhsResult := 21850)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21851 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63282⟩⟩]⟩) (none) 21850) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21856

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
