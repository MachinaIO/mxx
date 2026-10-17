import PePreLaneM0Group
import PePreLaneM0G1Group
import PePreLaneM0G2Group
import PePreLaneM0G3Group

namespace PeProof.PreLaneM0

open Mxx.Primitives MxxRuntime MxxExponentLut PeProof

noncomputable def inputsKey
    (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44)
    (K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44) :
    Fin 16 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44 :=
  ![M, parentKeys K00 K01 K10 K11 0, parentKeys K00 K01 K10 K11 1, parentKeys K00 K01 K10 K11 2, parentKeys K00 K01 K10 K11 3, parentKeys K00 K01 K10 K11 4, parentKeys K00 K01 K10 K11 5, parentKeys K00 K01 K10 K11 6, parentKeys K00 K01 K10 K11 7, parentKeys K00 K01 K10 K11 8, parentKeys K00 K01 K10 K11 9, parentKeys K00 K01 K10 K11 10, parentKeys K00 K01 K10 K11 11, parentKeys K00 K01 K10 K11 12, parentKeys K00 K01 K10 K11 13, parentKeys K00 K01 K10 K11 14]

theorem inputsKey_zero (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44)
    (K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44) :
    inputsKey M K00 K01 K10 K11 0 = M := rfl

theorem inputsKey_succ (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44)
    (K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44) (parent : Fin 15) :
    inputsKey M K00 K01 K10 K11 parent.succ = parentKeys K00 K01 K10 K11 parent := by
  exact Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) (Fin.cases (rfl) ((by intro index; exact Fin.elim0 index)))))))))))))))) parent

def outputRows
    (outputs0 : PreLaneM0Group.Output)
    (outputs1 : PreLaneM0G1Group.Output)
    (outputs2 : PreLaneM0G2Group.Output)
    (outputs3 : PreLaneM0G3Group.Output)
    : Fin 16 → Fin 2 → ExactMatrix Q N 1 88 :=
  ![PreLaneL0.row (PreLaneM0Group.outputRows outputs0) 0, PreLaneL0.row (PreLaneM0Group.outputRows outputs0) 1, PreLaneL0.row (PreLaneM0Group.outputRows outputs0) 2, PreLaneL0.row (PreLaneM0Group.outputRows outputs0) 3, PreLaneL0G1.row (PreLaneM0G1Group.outputRows outputs1) 0, PreLaneL0G1.row (PreLaneM0G1Group.outputRows outputs1) 1, PreLaneL0G1.row (PreLaneM0G1Group.outputRows outputs1) 2, PreLaneL0G1.row (PreLaneM0G1Group.outputRows outputs1) 3, PreLaneL0G2.row (PreLaneM0G2Group.outputRows outputs2) 0, PreLaneL0G2.row (PreLaneM0G2Group.outputRows outputs2) 1, PreLaneL0G2.row (PreLaneM0G2Group.outputRows outputs2) 2, PreLaneL0G2.row (PreLaneM0G2Group.outputRows outputs2) 3, PreLaneL0G3.row (PreLaneM0G3Group.outputRows outputs3) 0, PreLaneL0G3.row (PreLaneM0G3Group.outputRows outputs3) 1, PreLaneL0G3.row (PreLaneM0G3Group.outputRows outputs3) 2, PreLaneL0G3.row (PreLaneM0G3Group.outputRows outputs3) 3]

noncomputable def outputKey
    (outputs0 : PreLaneM0Group.Output)
    (outputs1 : PreLaneM0G1Group.Output)
    (outputs2 : PreLaneM0G2Group.Output)
    (outputs3 : PreLaneM0G3Group.Output)
    (r j : Fin 2) : ExactMatrix Q N 1 44 :=
  keyOf (PreLaneM0Group.outputKey outputs0) r j +
  keyOf (PreLaneM0G1Group.outputKey outputs1) r j +
  keyOf (PreLaneM0G2Group.outputKey outputs2) r j +
  keyOf (PreLaneM0G3Group.outputKey outputs3) r j

set_option maxRecDepth 200000 in
set_option maxHeartbeats 16000000 in
theorem lane {model : HashModel} {tape : SampleTape}
    {path0 path1 path2 path3 : List Nat} {params : Stage_preprocess.Params}
    {m : Nat} (hm : m < 7) {key : ByteArray} {tables : Fin 512 → Int}
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    {K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44}
    {A0 A1 : ExactMatrix Q N 1 88}
    {outputs0 : PreLaneM0Group.Output}
    {outputs1 : PreLaneM0G1Group.Output}
    {outputs2 : PreLaneM0G2Group.Output}
    {outputs3 : PreLaneM0G3Group.Output}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0)
    (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    (h0 : Stage_preprocess.parallel_generatedRoot_45
      Backend.backend model tape path0 params m
      (A0, key, A1, K00, K01, K10, K11, tables, ()) outputs0)
    (h1 : Stage_preprocess.parallel_generatedRoot_46
      Backend.backend model tape path1 params m
      (K00, K01, key, K10, K11, tables, ()) outputs1)
    (h2 : Stage_preprocess.parallel_generatedRoot_47
      Backend.backend model tape path2 params m
      (K00, K01, key, K10, K11, tables, ()) outputs2)
    (h3 : Stage_preprocess.parallel_generatedRoot_48
      Backend.backend model tape path3 params m
      (K00, K01, key, K10, K11, tables, ()) outputs3)
    : (∀ slot r jj, outputRows outputs0 outputs1 outputs2 outputs3 slot r 0 jj =
        lutTRow model key ⟨15 + m, by omega⟩ (inputsKey M K00 K01 K10 K11 slot) r jj) ∧
      (∀ r j, outputKey outputs0 outputs1 outputs2 outputs3 r j =
        pairMul (helperKey model key (kSw (15 + m)) r)
          (actionDigits (outputRows outputs0 outputs1 outputs2 outputs3 0 0) (outputRows outputs0 outputs1 outputs2 outputs3 0 1) j) *
          scalarV (branchCoef tables 0 (kSw (15 + m)) (by norm_num)) +
        ∑ parent : Fin 15, pairMul (helperKey model key (kSw (15 + m)) r)
          (actionDigits (outputRows outputs0 outputs1 outputs2 outputs3 parent.succ 0)
            (outputRows outputs0 outputs1 outputs2 outputs3 parent.succ 1) j) *
          scalarV (branchCoef tables 0 (kLeaf2 (parent.val + 15 * m)) (by norm_num))) := by
  obtain ⟨rows0, sum0⟩ := PreLaneM0Group.lane hm hA0 hA1 h0
  obtain ⟨rows1, sum1⟩ := PreLaneM0G1Group.lane hm h1
  obtain ⟨rows2, sum2⟩ := PreLaneM0G2Group.lane hm h2
  obtain ⟨rows3, sum3⟩ := PreLaneM0G3Group.lane hm h3
  constructor
  · exact Fin.cases (rows0 0) (Fin.cases (rows0 1) (Fin.cases (rows0 2) (Fin.cases (rows0 3) (Fin.cases (rows1 0) (Fin.cases (rows1 1) (Fin.cases (rows1 2) (Fin.cases (rows1 3) (Fin.cases (rows2 0) (Fin.cases (rows2 1) (Fin.cases (rows2 2) (Fin.cases (rows2 3) (Fin.cases (rows3 0) (Fin.cases (rows3 1) (Fin.cases (rows3 2) (Fin.cases (rows3 3) ((by intro index; exact Fin.elim0 index)))))))))))))))))
  · intro r j
    unfold outputKey
    rw [sum0 r j, sum1 r j, sum2 r j, sum3 r j]
    simp only [Fin.sum_univ_succ, Fin.sum_univ_zero]
    abel

end PeProof.PreLaneM0
