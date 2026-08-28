import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge95737
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def mergeEvent : Nat := 95737
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95731RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 95731 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95731) (leftOrdinal := 1)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95737

namespace LeftMerge95739
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def mergeEvent : Nat := 95739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def rhsRaw : List Term := Proof.Events031.exact7975RawTerms
def group : MergeGroup := .relation 95738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95738) (rhsResult := 7975)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7873⟩⟩) ⟨6787⟩ 7975) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95739

namespace LeftMerge95740
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def mergeEvent : Nat := 95740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95731RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 95731 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95731) (leftOrdinal := 0)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95740

namespace LeftMerge95745
def owner : Owner := ⟨.program ⟨214⟩, ⟨12745⟩⟩
def mergeEvent : Nat := 95745
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95741RawTerms
def rightRaw : List Term := Proof.Events373.exact95711RawTerms
def group : MergeGroup := .operator 95741 95711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95741) (leftOrdinal := 1)
    (rightResult := 95711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95745

namespace LeftMerge95753
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def mergeEvent : Nat := 95753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95747RawTerms
def rightRaw : List Term := Proof.Events373.exact95683RawTerms
def group : MergeGroup := .operator 95747 95683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95747) (leftOrdinal := 1)
    (rightResult := 95683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25514⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95753

namespace LeftMerge95755
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def mergeEvent : Nat := 95755
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95680RawTerms
def group : MergeGroup := .relation 95754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95754) (rhsResult := 95680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25514⟩⟩) ⟨23284⟩ 95680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨23284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95755

namespace LeftMerge95756
def owner : Owner := ⟨.program ⟨214⟩, ⟨25515⟩⟩
def mergeEvent : Nat := 95756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95747RawTerms
def rightRaw : List Term := Proof.Events373.exact95683RawTerms
def group : MergeGroup := .operator 95747 95683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95747) (leftOrdinal := 0)
    (rightResult := 95683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25514⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95756

namespace LeftMerge95770
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def mergeEvent : Nat := 95770
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events374.exact95764RawTerms
def group : MergeGroup := .operator 94462 95764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 95764) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20021⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95770

namespace LeftMerge95825
def owner : Owner := ⟨.program ⟨214⟩, ⟨12739⟩⟩
def mergeEvent : Nat := 95825
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events374.exact95821RawTerms
def rightRaw : List Term := Proof.Events374.exact95818RawTerms
def group : MergeGroup := .operator 95821 95818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95821) (leftOrdinal := 0)
    (rightResult := 95818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95825

namespace LeftMerge95855
def owner : Owner := ⟨.program ⟨214⟩, ⟨12852⟩⟩
def mergeEvent : Nat := 95855
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95851RawTerms
def rightRaw : List Term := Proof.Events374.exact95849RawTerms
def group : MergeGroup := .operator 95851 95849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95851) (leftOrdinal := 0)
    (rightResult := 95849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95855

namespace LeftMerge95878
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def mergeEvent : Nat := 95878
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95874RawTerms
def rightRaw : List Term := Proof.Events374.exact95871RawTerms
def group : MergeGroup := .operator 95874 95871
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95874) (leftOrdinal := 0)
    (rightResult := 95871) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95878

namespace LeftMerge95887
def owner : Owner := ⟨.program ⟨214⟩, ⟨25517⟩⟩
def mergeEvent : Nat := 95887
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95883RawTerms
def rightRaw : List Term := Proof.Events374.exact95840RawTerms
def group : MergeGroup := .operator 95883 95840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95883) (leftOrdinal := 0)
    (rightResult := 95840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25514⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95887

namespace LeftMerge95888
def owner : Owner := ⟨.program ⟨214⟩, ⟨25517⟩⟩
def mergeEvent : Nat := 95888
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95883RawTerms
def rightRaw : List Term := Proof.Events374.exact95840RawTerms
def group : MergeGroup := .operator 95883 95840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95883) (leftOrdinal := 1)
    (rightResult := 95840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25514⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95888

namespace LeftMerge95890
def owner : Owner := ⟨.program ⟨214⟩, ⟨25517⟩⟩
def mergeEvent : Nat := 95890
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }
def rhsRaw : List Term := Proof.Events374.exact95837RawTerms
def group : MergeGroup := .relation 95889
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95889) (rhsResult := 95837)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25514⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25514⟩⟩) ⟨23284⟩ 95837) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23284⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩, ⟨.program ⟨214⟩, ⟨12738⟩⟩], [⟨.program ⟨214⟩, ⟨23284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95890

namespace LeftMerge95898
def owner : Owner := ⟨.program ⟨214⟩, ⟨16625⟩⟩
def mergeEvent : Nat := 95898
def frameStart : Nat := 95807
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events374.exact95851RawTerms
def rightRaw : List Term := Proof.Events374.exact95894RawTerms
def group : MergeGroup := .operator 95851 95894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95851) (leftOrdinal := 0)
    (rightResult := 95894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16623⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16623⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95898

namespace LeftMerge95915
def owner : Owner := ⟨.program ⟨214⟩, ⟨20024⟩⟩
def mergeEvent : Nat := 95915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }
def rhsRaw : List Term := Proof.Events374.exact95912RawTerms
def group : MergeGroup := .relation 95914
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95914) (rhsResult := 95912)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95913 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20021⟩⟩]⟩) (none) 95912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6704⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95915

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
