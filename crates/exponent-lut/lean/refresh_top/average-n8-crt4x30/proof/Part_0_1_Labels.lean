import TopLabel
import W_Stage_online_eval_0_1

/-! The label payloads of the online stage of part `0_1` (written by `generator/gen_labels.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


theorem labelPay109 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q7 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_109 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_109 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_109.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay380 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q7 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_380 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_380 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_380.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay548 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_548 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_548 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_548.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay603 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_603 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_603 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_603.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay657 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_657 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_657 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_657.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay711 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_711 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_711 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_711.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay765 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_765 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_765 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_765.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


theorem labelPay819 {model : HashModel} {params : W_Stage_online_eval_0_1.Params} (i : Fin 5)
    {out : ExactMatrix Q6 N 1 1}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_819 Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp i)) := by
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_819 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_819.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i


end RefreshTopProof.Part_0_1
