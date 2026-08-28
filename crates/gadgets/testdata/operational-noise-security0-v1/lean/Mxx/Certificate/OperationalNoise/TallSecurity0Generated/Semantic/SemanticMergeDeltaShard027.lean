import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge5791
def owner : Owner := ⟨.program ⟨214⟩, ⟨6599⟩⟩
def mergeEvent : Nat := 5791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 2 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6427⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6427⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5791

namespace LeftMerge5806
def owner : Owner := ⟨.program ⟨214⟩, ⟨7634⟩⟩
def mergeEvent : Nat := 5806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5802RawTerms
def rightRaw : List Term := Proof.Events022.exact5799RawTerms
def group : MergeGroup := .operator 5802 5799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5802) (leftOrdinal := 0)
    (rightResult := 5799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6714⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6655⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6714⟩⟩, ⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5806

namespace LeftMerge5811
def owner : Owner := ⟨.program ⟨214⟩, ⟨6603⟩⟩
def mergeEvent : Nat := 5811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 2 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6452⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5811

namespace LeftMerge5826
def owner : Owner := ⟨.program ⟨214⟩, ⟨7633⟩⟩
def mergeEvent : Nat := 5826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5822RawTerms
def rightRaw : List Term := Proof.Events022.exact5819RawTerms
def group : MergeGroup := .operator 5822 5819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5822) (leftOrdinal := 0)
    (rightResult := 5819) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6712⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6663⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6712⟩⟩, ⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5826

namespace LeftMerge5831
def owner : Owner := ⟨.program ⟨214⟩, ⟨6607⟩⟩
def mergeEvent : Nat := 5831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 2 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6475⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5831

namespace LeftMerge5846
def owner : Owner := ⟨.program ⟨214⟩, ⟨7632⟩⟩
def mergeEvent : Nat := 5846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5842RawTerms
def rightRaw : List Term := Proof.Events022.exact5839RawTerms
def group : MergeGroup := .operator 5842 5839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5842) (leftOrdinal := 0)
    (rightResult := 5839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6671⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5846

namespace LeftMerge5851
def owner : Owner := ⟨.program ⟨214⟩, ⟨6611⟩⟩
def mergeEvent : Nat := 5851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 2 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6495⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5851

namespace LeftMerge5866
def owner : Owner := ⟨.program ⟨214⟩, ⟨7631⟩⟩
def mergeEvent : Nat := 5866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5862RawTerms
def rightRaw : List Term := Proof.Events022.exact5859RawTerms
def group : MergeGroup := .operator 5862 5859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5862) (leftOrdinal := 0)
    (rightResult := 5859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6679⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5866

namespace LeftMerge5877
def owner : Owner := ⟨.program ⟨214⟩, ⟨7650⟩⟩
def mergeEvent : Nat := 5877
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6760⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5873RawTerms
def rightRaw : List Term := Proof.Events022.exact5873RawTerms
def group : MergeGroup := .operator 5873 5873
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5873) (leftOrdinal := 0)
    (rightResult := 5873) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6760⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6760⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6760⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge5877

namespace LeftMerge5968
def owner : Owner := ⟨.program ⟨214⟩, ⟨7887⟩⟩
def mergeEvent : Nat := 5968
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact5964RawTerms
def rightRaw : List Term := Proof.Events023.exact5961RawTerms
def group : MergeGroup := .operator 5964 5961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5964) (leftOrdinal := 0)
    (rightResult := 5961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7885⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5968

namespace LeftMerge5973
def owner : Owner := ⟨.program ⟨214⟩, ⟨7911⟩⟩
def mergeEvent : Nat := 5973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact5969RawTerms
def rightRaw : List Term := Proof.Events021.exact5487RawTerms
def group : MergeGroup := .operator 5969 5487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5969) (leftOrdinal := 0)
    (rightResult := 5487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7819⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5973

namespace LeftMerge5978
def owner : Owner := ⟨.program ⟨214⟩, ⟨7917⟩⟩
def mergeEvent : Nat := 5978
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact5974RawTerms
def rightRaw : List Term := Proof.Events021.exact5476RawTerms
def group : MergeGroup := .operator 5974 5476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5974) (leftOrdinal := 0)
    (rightResult := 5476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6645⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6745⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7819⟩⟩, ⟨.program ⟨214⟩, ⟨6645⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5978

namespace LeftMerge5983
def owner : Owner := ⟨.program ⟨214⟩, ⟨6615⟩⟩
def mergeEvent : Nat := 5983
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact2RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 2 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2) (leftOrdinal := 0)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6543⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨6543⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge5983

namespace LeftMerge6008
def owner : Owner := ⟨.program ⟨214⟩, ⟨7888⟩⟩
def mergeEvent : Nat := 6008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact6004RawTerms
def rightRaw : List Term := Proof.Events023.exact5961RawTerms
def group : MergeGroup := .operator 6004 5961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6004) (leftOrdinal := 0)
    (rightResult := 5961) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7885⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6008

namespace LeftMerge6013
def owner : Owner := ⟨.program ⟨214⟩, ⟨7912⟩⟩
def mergeEvent : Nat := 6013
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact6009RawTerms
def rightRaw : List Term := Proof.Events023.exact6001RawTerms
def group : MergeGroup := .operator 6009 6001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6009) (leftOrdinal := 0)
    (rightResult := 6001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7821⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6013

namespace LeftMerge6018
def owner : Owner := ⟨.program ⟨214⟩, ⟨7918⟩⟩
def mergeEvent : Nat := 6018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩] } }
def leftRaw : List Term := Proof.Events023.exact6014RawTerms
def rightRaw : List Term := Proof.Events023.exact5991RawTerms
def group : MergeGroup := .operator 6014 5991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6014) (leftOrdinal := 0)
    (rightResult := 5991) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6747⟩⟩, ⟨.program ⟨214⟩, ⟨7885⟩⟩, ⟨.program ⟨214⟩, ⟨7821⟩⟩, ⟨.program ⟨214⟩, ⟨6687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6018

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
