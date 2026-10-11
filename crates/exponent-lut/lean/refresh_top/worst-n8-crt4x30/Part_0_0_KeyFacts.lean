import Stage_preprocess_0_0
import Stage_encrypt_0_0
import Part_0_0_KeysCells

/-! The label keys of the encodings of part `0_0` are hash samples (written by
`generator/gen_key_facts.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


theorem preHash5 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_5 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash6 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_6 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash7 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_7 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash8 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_8 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash368 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_368 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash369 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_369 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash370 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_370 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash371 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_371 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash412 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_412 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash413 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_413 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash414 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_414 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash415 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_415 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash456 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_456 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash457 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_457 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash458 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_458 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash459 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_459 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash500 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_500 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash501 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_501 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash502 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_502 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash503 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_503 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash544 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_544 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash545 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_545 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash546 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_546 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash547 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_547 Backend.backend model params i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash589 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_589 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash590 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_590 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash591 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_591 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash592 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_592 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash597 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_597 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash598 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_598 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash599 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_599 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash600 {model : HashModel} {params : Stage_preprocess_0_0.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_600 Backend.backend model params i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


set_option maxHeartbeats 4000000 in
theorem encKeys375 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_375 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_375 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_375.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys331 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_331 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_331 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_331.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys214 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_214 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_214 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_214.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys257 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_257 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_257 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_257.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys128 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_128 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_128 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_128.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys171 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_171 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_171 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_171.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys42 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_42 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_42 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_42.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys85 {model : HashModel} {params : Stage_encrypt_0_0.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt_0_0.parallel_generatedRoot_85 Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt_0_0.parallel_generatedRoot_85 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt_0_0.parallel_generatedRoot_85.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


end RefreshTopProof.Part_0_0
