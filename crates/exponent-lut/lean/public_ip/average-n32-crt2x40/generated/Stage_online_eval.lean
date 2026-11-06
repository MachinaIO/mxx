import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_online_eval

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_4.constraints_3 (w_32_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Int) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ w_30_0 ∧
  w_30_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_32_0 w_30_0 w_40_0 ∧
  outputs = (w_19_0, (w_26_0, (w_27_0, (w_28_0, (w_35_0, (w_39_0, (w_41_0, ())))))))

abbrev parallel_generatedRoot_4.constraints_2 (i_0 : Nat) (w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_7_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_32_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_1_0 : Int) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Int) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_37_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_25_0 2 3 0 4 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 4 w_29_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  0 ≤ w_30_0 ∧
  w_30_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_30_0 w_31_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_32_0 w_1_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 0 4 w_36_0 ∧
  0 ≤ w_30_0 ∧
  w_30_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_30_0 w_37_0 ∧
  parallel_generatedRoot_4.constraints_3 w_32_0 w_19_0 w_26_0 w_27_0 w_28_0 w_30_0 w_35_0 w_39_0 w_40_0 w_41_0 outputs

abbrev parallel_generatedRoot_4.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_3_0 : ByteArray) (w_7_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_32_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_1_0 : Int) (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Int) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_37_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_15_0 w_16_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_16_0 w_17_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_18_0 1 2 0 4 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_3_0) w_20_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_21_0 ∧
  MxxRuntime.concatRows w_20_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_22_0 w_23_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_23_0 w_24_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_25_0 1 2 0 4 w_26_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_18_0 2 3 0 4 w_27_0 ∧
  parallel_generatedRoot_4.constraints_2 i_0 w_0_0 w_7_0 w_32_0 w_1_0 w_18_0 w_19_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_41_0 outputs

abbrev parallel_generatedRoot_4.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_3_0 : ByteArray) (w_7_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_32_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_1_0 : Int) (w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_6_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_11_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 8) (w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_14_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Int) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_37_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_3_0) w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_5_0 ∧
  MxxRuntime.concatRows w_2_0 w_4_0 w_6_concat_1 ∧
  MxxRuntime.concatRows w_6_concat_1 w_5_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_3_0) w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_10_0 ∧
  MxxRuntime.concatRows w_8_0 w_9_0 w_11_concat_1 ∧
  MxxRuntime.concatRows w_11_concat_1 w_10_0 w_11_0 ∧
  MxxRuntime.concatColumns w_6_0 w_11_0 w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_3_0) w_13_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_14_0 ∧
  MxxRuntime.concatRows w_13_0 w_14_0 w_15_0 ∧
  parallel_generatedRoot_4.constraints_1 backend hashModel i_0 w_0_0 w_3_0 w_7_0 w_32_0 w_1_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_41_0 outputs

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × ByteArray × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := inputs.1
  let w_3_0 : ByteArray := inputs.2.1
  let w_7_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := inputs.2.2.1
  let w_32_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.2.2.1
  let w_1_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(0)]
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
    ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
      ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
        ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4),
          ∃ (w_6_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4),
            ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
              ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                  ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4),
                    ∃ (w_11_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4),
                      ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 8),
                        ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                          ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                            ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4),
                              ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4),
                                ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4),
                                  let w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4 := MxxRuntime.matrixMul w_12_0 w_17_0
                                  ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                    ∃ (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                      ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                        ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4),
                                          ∃ (w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4),
                                            ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4),
                                              let w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4 := MxxRuntime.matrixMul w_12_0 w_24_0
                                              ∃ (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                ∃ (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                      let w_30_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(1)]
                                                      ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                        ∃ (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                          let w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarRight w_31_0 w_33_0
                                                          let w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_29_0 w_34_0
                                                          ∃ (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                            ∃ (w_37_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                              let w_38_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarRight w_37_0 w_33_0
                                                              let w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_36_0 w_38_0
                                                              ∃ (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                let w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixMulScalarLeft w_33_0 w_40_0
                                                                parallel_generatedRoot_4.constraints_0 backend hashModel i_0 w_0_0 w_3_0 w_7_0 w_32_0 w_1_0 w_2_0 w_4_0 w_5_0 w_6_0 w_6_concat_1 w_8_0 w_9_0 w_10_0 w_11_0 w_11_concat_1 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_41_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_4 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_5 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_4_6 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4
  w_8_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4
  w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4
  w_12_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4
  w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 8
  w_14_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4
  w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4
  w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4
  w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4
  w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4
  w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4
  w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_32_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1

abbrev generatedRoot.constraints_2 (w_2_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_3_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_38_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_26_0 0 1 0 4 w_35_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (3) w_36_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (3) w_39_0 ∧
  outputs = (w_20_0, (w_27_0, (w_28_0, (w_29_0, (w_34_0, (w_38_0, (w_40_0, ())))))))

abbrev generatedRoot.constraints_1 (w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_2_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_3_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_4_6 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_32_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_38_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_26_0 1 2 0 4 w_27_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_19_0 2 3 0 4 w_28_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_26_0 2 3 0 4 w_29_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 4 w_30_0 ∧
  MxxRuntime.familyGetStatic w_0_0 (3) w_31_0 ∧
  MxxRuntime.familyGetStatic w_4_6 (0) w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  generatedRoot.constraints_2 w_2_0 w_3_0 w_20_0 w_26_0 w_27_0 w_28_0 w_29_0 w_34_0 w_35_0 w_36_0 w_38_0 w_39_0 w_40_0 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_1_0 : ByteArray) (w_2_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_3_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_4 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_5 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_4_6 : Fin 1 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_8_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_12_concat_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 8) (w_14_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 4) (w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 8 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_32_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_38_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ((w_4_4 i), ((w_4_5 i), ((w_4_6 i), ())))))))) ∧
  MxxRuntime.familyGetStatic w_4_4 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_7_0 ∧
  MxxRuntime.concatRows w_5_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_5 (0) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_10_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_11_0 ∧
  MxxRuntime.concatRows w_9_0 w_10_0 w_12_concat_1 ∧
  MxxRuntime.concatRows w_12_concat_1 w_11_0 w_12_0 ∧
  MxxRuntime.concatColumns w_8_0 w_12_0 w_13_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_14_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_15_0 ∧
  MxxRuntime.concatRows w_14_0 w_15_0 w_16_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_16_0 w_17_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_17_0 w_18_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_19_0 1 2 0 4 w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_21_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_22_0 ∧
  MxxRuntime.concatRows w_21_0 w_22_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_24_0 w_25_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  generatedRoot.constraints_1 w_0_0 w_2_0 w_3_0 w_4_6 w_19_0 w_20_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_38_0 w_39_0 w_40_0 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × ByteArray × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_2_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := inputs.2.2.1
  let w_3_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.2.2.1
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_4_2 := witness.w_4_2
  let w_4_3 := witness.w_4_3
  let w_4_4 := witness.w_4_4
  let w_4_5 := witness.w_4_5
  let w_4_6 := witness.w_4_6
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_8_concat_1 := witness.w_8_concat_1
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_12_concat_1 := witness.w_12_concat_1
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4 := MxxRuntime.matrixMul w_13_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 3 4 := MxxRuntime.matrixMul w_13_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarRight w_31_0 w_32_0
  let w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_30_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarRight w_36_0 w_32_0
  let w_38_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixMulScalarLeft w_32_0 w_39_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_4_4 w_4_5 w_4_6 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_9_0 w_10_0 w_11_0 w_12_0 w_12_concat_1 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_38_0 w_39_0 w_40_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × ByteArray × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_online_eval
