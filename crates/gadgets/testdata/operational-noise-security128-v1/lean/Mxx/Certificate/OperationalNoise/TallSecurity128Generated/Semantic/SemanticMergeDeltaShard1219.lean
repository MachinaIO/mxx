import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge198894
def owner : Owner := ⟨.program ⟨257⟩, ⟨55276⟩⟩
def mergeEvent : Nat := 198894
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events776.exact198890RawTerms
def rightRaw : List Term := Proof.Events776.exact198888RawTerms
def group : MergeGroup := .operator 198890 198888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198890) (leftOrdinal := 0)
    (rightResult := 198888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198894

namespace LeftMerge198917
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 198917
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198913RawTerms
def rightRaw : List Term := Proof.Events776.exact198910RawTerms
def group : MergeGroup := .operator 198913 198910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198913) (leftOrdinal := 0)
    (rightResult := 198910) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198917

namespace LeftMerge198926
def owner : Owner := ⟨.program ⟨257⟩, ⟨55524⟩⟩
def mergeEvent : Nat := 198926
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198922RawTerms
def rightRaw : List Term := Proof.Events776.exact198879RawTerms
def group : MergeGroup := .operator 198922 198879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198922) (leftOrdinal := 0)
    (rightResult := 198879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55521⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198926

namespace LeftMerge198927
def owner : Owner := ⟨.program ⟨257⟩, ⟨55524⟩⟩
def mergeEvent : Nat := 198927
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198922RawTerms
def rightRaw : List Term := Proof.Events776.exact198879RawTerms
def group : MergeGroup := .operator 198922 198879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198922) (leftOrdinal := 1)
    (rightResult := 198879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55521⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198927

namespace LeftMerge198929
def owner : Owner := ⟨.program ⟨257⟩, ⟨55524⟩⟩
def mergeEvent : Nat := 198929
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198876RawTerms
def group : MergeGroup := .relation 198928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198928) (rhsResult := 198876)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55521⟩⟩) ⟨55001⟩ 198876) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨55001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198929

namespace LeftMerge198937
def owner : Owner := ⟨.program ⟨257⟩, ⟨53886⟩⟩
def mergeEvent : Nat := 198937
def frameStart : Nat := 198834
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events776.exact198890RawTerms
def rightRaw : List Term := Proof.Events777.exact198933RawTerms
def group : MergeGroup := .operator 198890 198933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198890) (leftOrdinal := 0)
    (rightResult := 198933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198937

namespace LeftMerge198954
def owner : Owner := ⟨.program ⟨257⟩, ⟨54452⟩⟩
def mergeEvent : Nat := 198954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events777.exact198951RawTerms
def group : MergeGroup := .relation 198953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198953) (rhsResult := 198951)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩) (none) 198951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198954

namespace LeftMerge198955
def owner : Owner := ⟨.program ⟨257⟩, ⟨54452⟩⟩
def mergeEvent : Nat := 198955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }
def rhsRaw : List Term := Proof.Events777.exact198951RawTerms
def group : MergeGroup := .relation 198953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198953) (rhsResult := 198951)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩) (none) 198951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198955

namespace LeftMerge198956
def owner : Owner := ⟨.program ⟨257⟩, ⟨54452⟩⟩
def mergeEvent : Nat := 198956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }
def rhsRaw : List Term := Proof.Events777.exact198951RawTerms
def group : MergeGroup := .relation 198953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198953) (rhsResult := 198951)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩) (none) 198951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨55001⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198956

namespace LeftMerge198957
def owner : Owner := ⟨.program ⟨257⟩, ⟨54452⟩⟩
def mergeEvent : Nat := 198957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events777.exact198951RawTerms
def group : MergeGroup := .relation 198953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198953) (rhsResult := 198951)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54449⟩⟩]⟩) (none) 198951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198957

namespace LeftMerge198962
def owner : Owner := ⟨.program ⟨257⟩, ⟨55523⟩⟩
def mergeEvent : Nat := 198962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198958RawTerms
def rightRaw : List Term := Proof.Events776.exact198772RawTerms
def group : MergeGroup := .operator 198958 198772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198958) (leftOrdinal := 2)
    (rightResult := 198772) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55001⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24794⟩⟩, ⟨.program ⟨257⟩, ⟨53579⟩⟩], [⟨.program ⟨257⟩, ⟨55001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198962

namespace LeftMerge198963
def owner : Owner := ⟨.program ⟨257⟩, ⟨55523⟩⟩
def mergeEvent : Nat := 198963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198958RawTerms
def rightRaw : List Term := Proof.Events776.exact198772RawTerms
def group : MergeGroup := .operator 198958 198772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198958) (leftOrdinal := 1)
    (rightResult := 198772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55521⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198963

namespace LeftMerge198971
def owner : Owner := ⟨.program ⟨257⟩, ⟨55996⟩⟩
def mergeEvent : Nat := 198971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55994⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198965RawTerms
def rightRaw : List Term := Proof.Events776.exact198688RawTerms
def group : MergeGroup := .operator 198965 198688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198965) (leftOrdinal := 0)
    (rightResult := 198688) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198971

namespace LeftMerge198972
def owner : Owner := ⟨.program ⟨257⟩, ⟨55996⟩⟩
def mergeEvent : Nat := 198972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55994⟩⟩] } }
def leftRaw : List Term := Proof.Events777.exact198965RawTerms
def rightRaw : List Term := Proof.Events776.exact198688RawTerms
def group : MergeGroup := .operator 198965 198688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198965) (leftOrdinal := 1)
    (rightResult := 198688) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198972

namespace LeftMerge198974
def owner : Owner := ⟨.program ⟨257⟩, ⟨55996⟩⟩
def mergeEvent : Nat := 198974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55159⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198685RawTerms
def group : MergeGroup := .relation 198973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198973) (rhsResult := 198685)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55994⟩⟩) ⟨55159⟩ 198685) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55159⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨53884⟩⟩], [⟨.program ⟨257⟩, ⟨55159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198974

namespace LeftMerge198988
def owner : Owner := ⟨.program ⟨257⟩, ⟨54779⟩⟩
def mergeEvent : Nat := 198988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54776⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events777.exact198982RawTerms
def group : MergeGroup := .operator 192995 198982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 198982) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198988

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
