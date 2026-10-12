import TopInst
import Stage_preprocess
import Stage_encrypt
import Part_I_KeysCells

/-! The label keys of the encodings of part `I` are hash samples (written by
`generator/gen_key_facts.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


theorem preHash5 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_5 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash6 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_6 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash7 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_7 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash8 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_8 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash368 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_372 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash369 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_373 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash370 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_374 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash371 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_375 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash412 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_416 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash413 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_417 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash414 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_418 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash415 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_419 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash456 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_460 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash457 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_461 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash458 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_462 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash459 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_463 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash500 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_504 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash501 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_505 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash502 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_506 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash503 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_507 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash544 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_548 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash545 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_549 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash546 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_550 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash547 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q6 N 1 12}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_551 Backend.backend model params i0 i key out) :
    out = model.sample Q6 N 1 12 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash589 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_597 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash590 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_598 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash591 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_599 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash592 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_600 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash597 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_605 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (0 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash598 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_606 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (1 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash599 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_607 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (2 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


theorem preHash600 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_preprocess.Params} {i : Nat} {key : ByteArray}
    {out : ExactMatrix Q7 N 1 14}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_407_608 Backend.backend model params i0 i key out) :
    out = model.sample Q7 N 1 14 key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le (3 : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs


set_option maxHeartbeats 4000000 in
theorem encKeys375 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_375 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_375 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_375.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys331 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_331 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_331 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_331.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys214 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_214 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_214 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_214.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys257 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_257 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_257 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_257.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys128 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_128 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_128 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_128.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys171 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_171 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_171 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_171.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys42 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_42 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_42 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_42.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


set_option maxHeartbeats 4000000 in
theorem encKeys85 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_85 Backend.backend model params i0 o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_85 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_85.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩


end RefreshTopProof.Part_I
