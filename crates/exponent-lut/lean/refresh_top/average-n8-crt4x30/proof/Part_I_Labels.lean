import TopInst
import TopLabel
import W_Stage_online_eval

/-! The label payloads of the online stage of part `I` (written by `generator/gen_labels.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


theorem labelPay109 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q7 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_109 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_109 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_109.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay380 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q7 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_382 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_382 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_382.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay548 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_550 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_550 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_550.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay603 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_605 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_605 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_605.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay657 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_659 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_659 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_659.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay711 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_713 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_713 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_713.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay765 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_767 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_767 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_767.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay819 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_821 Backend.backend model params i0 i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_821 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_821.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


end RefreshTopProof.Part_I
