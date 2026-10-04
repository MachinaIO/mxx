import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preprocess

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_3.constraints_2 (i_0 : Nat) (w_115_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_69_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_75_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_81_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_82_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_86_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_87_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_97_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_98_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_110_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_111_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_116_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_120_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_121_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_69_0 w_70_0 ∧
  0 ≤ (10) ∧
  (10) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_74_0 w_75_0 ∧
  0 ≤ (8) ∧
  (8) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_81_0 w_82_0 ∧
  0 ≤ (6) ∧
  (6) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_86_0 w_87_0 ∧
  0 ≤ (4) ∧
  (4) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_92_0 w_93_0 ∧
  0 ≤ (2) ∧
  (2) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_97_0 w_98_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_110_0 ∧
  MxxRuntime.modulusReduceRuns w_115_0 w_116_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_120_0 ∧
  outputs = (w_1_0, (w_2_0, (w_111_0, (w_121_0, ()))))

abbrev parallel_generatedRoot_3.constraints_1 (i_0 : Nat) (w_115_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_34_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_39_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_45_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_50_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_64_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_69_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_75_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_81_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_82_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_86_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_87_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_97_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_98_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_110_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_111_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_116_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_120_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_121_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_34_0 w_35_0 ∧
  0 ≤ (22) ∧
  (22) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_39_0 w_40_0 ∧
  0 ≤ (20) ∧
  (20) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_45_0 w_46_0 ∧
  0 ≤ (18) ∧
  (18) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_50_0 w_51_0 ∧
  0 ≤ (16) ∧
  (16) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_58_0 w_59_0 ∧
  0 ≤ (14) ∧
  (14) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_63_0 w_64_0 ∧
  0 ≤ (12) ∧
  (12) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  parallel_generatedRoot_3.constraints_2 i_0 w_115_0 w_1_0 w_2_0 w_69_0 w_70_0 w_74_0 w_75_0 w_81_0 w_82_0 w_86_0 w_87_0 w_92_0 w_93_0 w_97_0 w_98_0 w_110_0 w_111_0 w_116_0 w_120_0 w_121_0 outputs

abbrev parallel_generatedRoot_3.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_115_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_11_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_17_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_22_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_23_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_28_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_34_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_39_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_45_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_50_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_64_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_69_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_75_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_81_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_82_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_86_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_87_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_97_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_98_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_110_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_111_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_116_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_120_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_121_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 53, 47, 54]) ([.bytes [115, 105, 103, 109, 97], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 53, 47, 54]) ([.bytes [115, 105, 103, 109, 97], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  MxxRuntime.modulusReduceRuns w_3_0 w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_4_0 w_5_0 ∧
  0 ≤ (0) ∧
  (0) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_11_0 w_12_0 ∧
  0 ≤ (30) ∧
  (30) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_16_0 w_17_0 ∧
  0 ≤ (28) ∧
  (28) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_22_0 w_23_0 ∧
  0 ≤ (26) ∧
  (26) < 32 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_27_0 w_28_0 ∧
  0 ≤ (24) ∧
  (24) < 32 ∧
  parallel_generatedRoot_3.constraints_1 i_0 w_115_0 w_1_0 w_2_0 w_34_0 w_35_0 w_39_0 w_40_0 w_45_0 w_46_0 w_50_0 w_51_0 w_58_0 w_59_0 w_63_0 w_64_0 w_69_0 w_70_0 w_74_0 w_75_0 w_81_0 w_82_0 w_86_0 w_87_0 w_92_0 w_93_0 w_97_0 w_98_0 w_110_0 w_111_0 w_116_0 w_120_0 w_121_0 outputs

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.1
  let w_115_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
    ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
        ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
          let w_6_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_5_0
          let w_7_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_1_0
          let w_8_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(549755813024)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
          let w_9_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1030792149421)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
          let w_10_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
          let w_11_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (0) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
          ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
            let w_13_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_10_0 w_12_0
            let w_14_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
            let w_15_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (30) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
            let w_16_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_15_0
            ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
              let w_18_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_14_0 w_17_0
              let w_19_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_13_0 w_18_0
              let w_20_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
              let w_21_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (28) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
              let w_22_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_21_0
              ∃ (w_23_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                let w_24_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_20_0 w_23_0
                let w_25_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                let w_26_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (26) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                let w_27_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_26_0
                ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                  let w_29_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_25_0 w_28_0
                  let w_30_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_24_0 w_29_0
                  let w_31_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_19_0 w_30_0
                  let w_32_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                  let w_33_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (24) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                  let w_34_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_33_0
                  ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                    let w_36_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_32_0 w_35_0
                    let w_37_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                    let w_38_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (22) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                    let w_39_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_38_0
                    ∃ (w_40_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                      let w_41_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_37_0 w_40_0
                      let w_42_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_36_0 w_41_0
                      let w_43_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                      let w_44_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (20) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                      let w_45_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_44_0
                      ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                        let w_47_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_43_0 w_46_0
                        let w_48_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                        let w_49_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (18) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                        let w_50_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_49_0
                        ∃ (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                          let w_52_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_48_0 w_51_0
                          let w_53_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_47_0 w_52_0
                          let w_54_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_42_0 w_53_0
                          let w_55_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_31_0 w_54_0
                          let w_56_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                          let w_57_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (16) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                          let w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_57_0
                          ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                            let w_60_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_56_0 w_59_0
                            let w_61_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                            let w_62_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (14) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                            let w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_62_0
                            ∃ (w_64_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                              let w_65_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_61_0 w_64_0
                              let w_66_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_60_0 w_65_0
                              let w_67_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                              let w_68_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (12) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                              let w_69_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_68_0
                              ∃ (w_70_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                let w_71_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_67_0 w_70_0
                                let w_72_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                let w_73_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (10) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                let w_74_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_73_0
                                ∃ (w_75_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                  let w_76_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_72_0 w_75_0
                                  let w_77_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_71_0 w_76_0
                                  let w_78_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_66_0 w_77_0
                                  let w_79_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                  let w_80_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (8) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                  let w_81_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_80_0
                                  ∃ (w_82_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                    let w_83_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_79_0 w_82_0
                                    let w_84_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                    let w_85_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (6) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                    let w_86_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_85_0
                                    ∃ (w_87_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                      let w_88_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_84_0 w_87_0
                                      let w_89_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_83_0 w_88_0
                                      let w_90_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                      let w_91_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (4) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                      let w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_91_0
                                      ∃ (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                        let w_94_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_90_0 w_93_0
                                        let w_95_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.matrixPolynomial [(2)] : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                        let w_96_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (MxxRuntime.rotationPolynomial (2) : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                        let w_97_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_96_0
                                        ∃ (w_98_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                          let w_99_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_95_0 w_98_0
                                          let w_100_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_94_0 w_99_0
                                          let w_101_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_89_0 w_100_0
                                          let w_102_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_78_0 w_101_0
                                          let w_103_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_55_0 w_102_0
                                          let w_104_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_103_0
                                          let w_105_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_8_0 w_104_0
                                          let w_106_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_105_0 w_4_0
                                          let w_107_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := (1 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1)
                                          let w_108_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_106_0 w_107_0
                                          let w_109_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_7_0 w_108_0
                                          ∃ (w_110_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                            let w_111_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_109_0 w_110_0
                                            let w_112_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixNeg w_5_0
                                            let w_113_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_112_0 w_2_0
                                            let w_114_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_8_0 w_104_0
                                            ∃ (w_116_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                              let w_117_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_114_0 w_116_0
                                              let w_118_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_117_0 w_107_0
                                              let w_119_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_113_0 w_118_0
                                              ∃ (w_120_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                let w_121_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_119_0 w_120_0
                                                parallel_generatedRoot_3.constraints_0 hashModel i_0 w_0_0 w_3_0 w_115_0 w_1_0 w_2_0 w_4_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_22_0 w_23_0 w_27_0 w_28_0 w_34_0 w_35_0 w_39_0 w_40_0 w_45_0 w_46_0 w_50_0 w_51_0 w_58_0 w_59_0 w_63_0 w_64_0 w_69_0 w_70_0 w_74_0 w_75_0 w_81_0 w_82_0 w_86_0 w_87_0 w_92_0 w_93_0 w_97_0 w_98_0 w_110_0 w_111_0 w_116_0 w_120_0 w_121_0 outputs

abbrev parallel_generatedRoot_11.constraints_3 (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_47_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_53_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_54_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_66_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_72_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_74_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_75_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_76_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_75_0 w_76_0 ∧
  outputs = (w_53_0, (w_72_0, (w_74_0, (w_76_0, (w_17_0, (w_54_0, (w_47_0, (w_66_0, ()))))))))

abbrev parallel_generatedRoot_11.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_58_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_47_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_51_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_53_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_54_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_55_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_56_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_59_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_66_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_67_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_68_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_70_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_72_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_73_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_74_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_75_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_76_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_51_0 0 1 0 10 w_52_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_52_0 w_53_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_54_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_55_0 w_56_0 ∧
  MxxRuntime.modulusReduceRuns w_58_0 w_59_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_65_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_66_0 w_67_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_67_0 w_68_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_70_0 0 1 0 10 w_71_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_71_0 w_72_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_51_0 1 2 0 10 w_73_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_73_0 w_74_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_70_0 1 2 0 10 w_75_0 ∧
  parallel_generatedRoot_11.constraints_3 w_17_0 w_47_0 w_53_0 w_54_0 w_66_0 w_72_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_generatedRoot_11.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_21_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_24_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_31_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_25_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_27_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_28_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_29_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_30_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_47_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_48_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_49_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_51_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_53_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_54_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_55_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_56_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_59_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_66_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_67_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_68_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_70_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_72_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_73_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_74_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_75_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_76_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_17_0 w_18_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_18_0 w_19_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 12 ∧
  12 ≤ 12 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 12 w_28_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 12 ∧
  12 ≤ 12 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 12 w_29_0 ∧
  MxxRuntime.concatRows w_28_0 w_29_0 w_30_0 ∧
  MxxRuntime.modulusReduceRuns w_31_0 w_32_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 10 w_33_0 ∧
  MxxRuntime.centeredRebaseRuns w_33_0 w_34_0 ∧
  MxxRuntime.modulusReduceRuns w_41_0 w_42_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_46_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_47_0 w_48_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_48_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  parallel_generatedRoot_11.constraints_2 backend hashModel i_0 w_0_0 w_58_0 w_17_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_59_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_21_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_24_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_31_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_3_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 14) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_8_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 14) (w_11_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_13_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_14_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_15_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_25_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_27_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_28_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_29_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_30_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12) (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_47_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_48_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_49_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_51_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_53_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_54_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_55_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_56_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_59_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_66_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_67_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_68_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10) (w_70_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_72_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_73_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_74_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_75_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_76_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(4), (5), (10), (11), (12), (14), (15), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(4), (5), (10), (11), (12), (14), (15), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_2_0 ∧
  MxxRuntime.concatRows w_1_0 w_2_0 w_3_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(4), (5), (10), (11), (12), (14), (15), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(4), (5), (10), (11), (12), (14), (15), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_8_0 ∧
  MxxRuntime.concatRows w_7_0 w_8_0 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 12 ∧
  12 ≤ 12 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 0 12 w_14_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 12 ∧
  12 ≤ 12 ∧
  MxxRuntime.sliceMatrix w_13_0 1 2 0 12 w_15_0 ∧
  MxxRuntime.concatRows w_14_0 w_15_0 w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_17_0 ∧
  parallel_generatedRoot_11.constraints_1 backend hashModel i_0 w_0_0 w_21_0 w_24_0 w_31_0 w_58_0 w_17_0 w_18_0 w_19_0 w_22_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_41_0 w_42_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_59_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := inputs.2.1
  let w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := inputs.2.2.1
  let w_21_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := inputs.2.2.2.1
  let w_24_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := inputs.2.2.2.2.1
  let w_31_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.1
  let w_38_0 : Int := inputs.2.2.2.2.2.2.1
  let w_40_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_58_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
    ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
      ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 14),
        ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12),
          let w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixMul w_3_0 w_5_0
          ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
            ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
              ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 14),
                ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12),
                  let w_12_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixMul w_9_0 w_11_0
                  let w_13_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixAdd w_6_0 w_12_0
                  ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12),
                    ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12),
                      ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12),
                        ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                          ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                            ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10),
                              let w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixMul w_16_0 w_19_0
                              ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12),
                                let w_23_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixMul w_3_0 w_22_0
                                ∃ (w_25_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12),
                                  let w_26_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixMul w_9_0 w_25_0
                                  let w_27_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12 := MxxRuntime.matrixAdd w_23_0 w_26_0
                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12),
                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12),
                                      ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 12),
                                        ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1),
                                          ∃ (w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
                                            ∃ (w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                              let w_35_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarLeft w_32_0 w_34_0
                                              let w_36_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511621249)] : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1)
                                              let w_37_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarRight w_35_0 w_36_0
                                              let w_39_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.liftInteger w_38_0
                                              let w_41_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.multiplyMonomial w_39_0 w_40_0
                                              ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1),
                                                let w_43_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarRight w_37_0 w_42_0
                                                let w_44_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarLeft w_32_0 w_17_0
                                                let w_45_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixSub w_43_0 w_44_0
                                                ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                  let w_47_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_45_0 w_46_0
                                                  ∃ (w_48_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                                    ∃ (w_49_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10),
                                                      let w_50_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixMul w_30_0 w_49_0
                                                      let w_51_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixAdd w_20_0 w_50_0
                                                      ∃ (w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10),
                                                        ∃ (w_53_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                          ∃ (w_54_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                            ∃ (w_55_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                                              ∃ (w_56_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10),
                                                                let w_57_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixMul w_16_0 w_56_0
                                                                ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 1),
                                                                  let w_60_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarLeft w_59_0 w_34_0
                                                                  let w_61_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarRight w_60_0 w_36_0
                                                                  let w_62_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarRight w_61_0 w_42_0
                                                                  let w_63_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixMulScalarLeft w_32_0 w_54_0
                                                                  let w_64_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixSub w_62_0 w_63_0
                                                                  ∃ (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                                    let w_66_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_64_0 w_65_0
                                                                    ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                                                      ∃ (w_68_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 12 10),
                                                                        let w_69_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixMul w_30_0 w_68_0
                                                                        let w_70_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 2 10 := MxxRuntime.matrixAdd w_57_0 w_69_0
                                                                        ∃ (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10),
                                                                          ∃ (w_72_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                                            ∃ (w_73_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10),
                                                                              ∃ (w_74_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                                                ∃ (w_75_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10),
                                                                                  ∃ (w_76_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                                                    parallel_generatedRoot_11.constraints_0 backend hashModel i_0 w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_58_0 w_1_0 w_2_0 w_3_0 w_5_0 w_7_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_22_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_41_0 w_42_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_59_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev sequential_generatedRoot_24.constraints_0 (w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_7_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_8_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.1
  let w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.2.1
  let w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
    let w_4_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
      let w_8_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
          let w_16_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_24.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev parallel_generatedRoot_27.constraints_3 (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_66_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_67_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_79_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_83_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_85_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_87_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_88_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_89_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_83_0 1 2 0 8 w_88_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249] (1) w_88_0 w_89_0 ∧
  outputs = (w_66_0, (w_85_0, (w_87_0, (w_89_0, (w_30_0, (w_67_0, (w_60_0, (w_79_0, ()))))))))

abbrev parallel_generatedRoot_27.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_61_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_62_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_64_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_66_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_67_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_68_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_69_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_72_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_79_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_80_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_81_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_83_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_84_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_85_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_86_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_87_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_88_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_89_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 524288 w_61_0 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_64_0 0 1 0 8 w_65_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249] (1) w_65_0 w_66_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_67_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_67_0 w_68_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_68_0 w_69_0 ∧
  MxxRuntime.modulusReduceRuns w_71_0 w_72_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_78_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_83_0 0 1 0 8 w_84_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249] (1) w_84_0 w_85_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_64_0 1 2 0 8 w_86_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249] (1) w_86_0 w_87_0 ∧
  parallel_generatedRoot_27.constraints_3 w_30_0 w_60_0 w_66_0 w_67_0 w_79_0 w_83_0 w_85_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_27.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_37_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_27_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_29_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_35_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_38_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_40_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_43_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_45_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_61_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_62_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_64_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_66_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_67_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_68_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_69_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_72_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_79_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_80_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_81_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_83_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_84_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_85_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_86_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_87_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_88_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_89_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_26_0 1 2 0 10 w_28_0 ∧
  MxxRuntime.concatRows w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_30_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_30_0 w_31_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_34_0 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_37_0 w_38_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_40_0 0 1 0 10 w_41_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_40_0 1 2 0 10 w_42_0 ∧
  MxxRuntime.concatRows w_41_0 w_42_0 w_43_0 ∧
  MxxRuntime.modulusReduceRuns w_44_0 w_45_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 8 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.modulusReduceRuns w_54_0 w_55_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_60_0 w_61_0 ∧
  parallel_generatedRoot_27.constraints_2 backend hashModel i_0 w_0_0 w_71_0 w_30_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_13_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_23_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_37_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_3_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12) (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_8_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12) (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_11_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12) (w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 12) (w_14_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_21_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12) (w_22_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 12) (w_24_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_26_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_27_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_29_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_35_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_38_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10) (w_40_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_43_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10) (w_45_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_61_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_62_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_64_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_66_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_67_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_68_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_69_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_72_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_79_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_80_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_81_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8) (w_83_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8) (w_84_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_85_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_86_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_87_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_88_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8) (w_89_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (4), (6), (7), (8), (9), (12), (13), (14), (15), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 12 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_5_0 w_6_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (4), (6), (7), (8), (9), (12), (13), (14), (15), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_9_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_10_0 w_11_0 ∧
  MxxRuntime.concatRows w_8_0 w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (4), (6), (7), (8), (9), (12), (13), (14), (15), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_16_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (4), (6), (7), (8), (9), (12), (13), (14), (15), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_19_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_20_0 w_21_0 ∧
  MxxRuntime.concatRows w_18_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 12 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 10 ∧
  MxxRuntime.sliceMatrix w_26_0 0 1 0 10 w_27_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  parallel_generatedRoot_27.constraints_1 backend hashModel i_0 w_0_0 w_34_0 w_37_0 w_44_0 w_71_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.1
  let w_23_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.1
  let w_34_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.1
  let w_37_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := inputs.2.2.2.2.1
  let w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.1
  let w_51_0 : Int := inputs.2.2.2.2.2.2.1
  let w_53_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
    let w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511619841)] : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12),
        let w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 12),
          let w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := MxxRuntime.matrixMul w_1_0 w_6_0
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12),
            ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
              let w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := MxxRuntime.matrixMul w_9_0 w_6_0
              ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12),
                ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 12),
                  ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                    let w_15_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixMul w_12_0 w_14_0
                    ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                      let w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := MxxRuntime.matrixMul w_16_0 w_6_0
                      ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12),
                        ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                          let w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 := MxxRuntime.matrixMul w_19_0 w_6_0
                          ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 12),
                            ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 12),
                              ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                let w_25_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixMul w_22_0 w_24_0
                                let w_26_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixAdd w_15_0 w_25_0
                                ∃ (w_27_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10),
                                      ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                        ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                          ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8),
                                            let w_33_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixMul w_29_0 w_32_0
                                            ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                              let w_36_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixMul w_12_0 w_35_0
                                              ∃ (w_38_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 12 10),
                                                let w_39_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixMul w_22_0 w_38_0
                                                let w_40_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10 := MxxRuntime.matrixAdd w_36_0 w_39_0
                                                ∃ (w_41_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                  ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10),
                                                    ∃ (w_43_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 10),
                                                      ∃ (w_45_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1),
                                                        ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
                                                          ∃ (w_47_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                            let w_48_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarLeft w_45_0 w_47_0
                                                            let w_49_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511622529)] : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1)
                                                            let w_50_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarRight w_48_0 w_49_0
                                                            let w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.liftInteger w_51_0
                                                            let w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.multiplyMonomial w_52_0 w_53_0
                                                            ∃ (w_55_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1),
                                                              let w_56_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarRight w_50_0 w_55_0
                                                              let w_57_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarLeft w_45_0 w_30_0
                                                              let w_58_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixSub w_56_0 w_57_0
                                                              ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                let w_60_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_58_0 w_59_0
                                                                ∃ (w_61_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                                                  ∃ (w_62_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8),
                                                                    let w_63_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixMul w_43_0 w_62_0
                                                                    let w_64_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixAdd w_33_0 w_63_0
                                                                    ∃ (w_65_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8),
                                                                      ∃ (w_66_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                        ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                          ∃ (w_68_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                                                            ∃ (w_69_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8),
                                                                              let w_70_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixMul w_29_0 w_69_0
                                                                              ∃ (w_72_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 1),
                                                                                let w_73_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarLeft w_72_0 w_47_0
                                                                                let w_74_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarRight w_73_0 w_49_0
                                                                                let w_75_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarRight w_74_0 w_55_0
                                                                                let w_76_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixMulScalarLeft w_45_0 w_67_0
                                                                                let w_77_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixSub w_75_0 w_76_0
                                                                                ∃ (w_78_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                                  let w_79_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_77_0 w_78_0
                                                                                  ∃ (w_80_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                                                                    ∃ (w_81_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 10 8),
                                                                                      let w_82_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixMul w_43_0 w_81_0
                                                                                      let w_83_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 2 8 := MxxRuntime.matrixAdd w_70_0 w_82_0
                                                                                      ∃ (w_84_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8),
                                                                                        ∃ (w_85_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                                          ∃ (w_86_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8),
                                                                                            ∃ (w_87_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                                              ∃ (w_88_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 8),
                                                                                                ∃ (w_89_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                                                                  parallel_generatedRoot_27.constraints_0 backend hashModel i_0 w_0_0 w_13_0 w_23_0 w_34_0 w_37_0 w_44_0 w_71_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev sequential_generatedRoot_40.constraints_0 (w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_4_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_7_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_15_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_40 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.1
  let w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.2.1
  let w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
    let w_4_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
      let w_8_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
          let w_16_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_40.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev parallel_generatedRoot_43.constraints_3 (w_30_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_60_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_66_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_67_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_83_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_85_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_87_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_88_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_89_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_83_0 1 2 0 6 w_88_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529] (1) w_88_0 w_89_0 ∧
  outputs = (w_66_0, (w_85_0, (w_87_0, (w_89_0, (w_30_0, (w_67_0, (w_60_0, (w_79_0, ()))))))))

abbrev parallel_generatedRoot_43.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_60_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_61_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_64_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_65_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_66_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_67_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_68_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_69_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_72_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_80_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_81_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_83_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_84_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_85_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_86_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_87_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_88_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_89_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 524288 w_61_0 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_64_0 0 1 0 6 w_65_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529] (1) w_65_0 w_66_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_67_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_67_0 w_68_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_68_0 w_69_0 ∧
  MxxRuntime.modulusReduceRuns w_71_0 w_72_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_78_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_83_0 0 1 0 6 w_84_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529] (1) w_84_0 w_85_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_64_0 1 2 0 6 w_86_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529] (1) w_86_0 w_87_0 ∧
  parallel_generatedRoot_43.constraints_3 w_30_0 w_60_0 w_66_0 w_67_0 w_79_0 w_83_0 w_85_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_43.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_34_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_27_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_30_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_31_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_32_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_35_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_38_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_40_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_41_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_60_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_61_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_64_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_65_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_66_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_67_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_68_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_69_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_72_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_80_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_81_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_83_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_84_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_85_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_86_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_87_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_88_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_89_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_26_0 1 2 0 8 w_28_0 ∧
  MxxRuntime.concatRows w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_30_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_30_0 w_31_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_34_0 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_37_0 w_38_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_40_0 0 1 0 8 w_41_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_40_0 1 2 0 8 w_42_0 ∧
  MxxRuntime.concatRows w_41_0 w_42_0 w_43_0 ∧
  MxxRuntime.modulusReduceRuns w_44_0 w_45_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 6 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.modulusReduceRuns w_54_0 w_55_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_60_0 w_61_0 ∧
  parallel_generatedRoot_43.constraints_2 backend hashModel i_0 w_0_0 w_71_0 w_30_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_13_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_23_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_34_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_3_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 10) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_8_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_11_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_18_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 10) (w_24_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_26_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_27_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_30_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_31_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_32_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_35_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_38_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8) (w_40_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_41_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8) (w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_60_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_61_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_64_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_65_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_66_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_67_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_68_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_69_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_72_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_80_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_81_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6) (w_83_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6) (w_84_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_85_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_86_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_87_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_88_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6) (w_89_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (5), (8), (9), (10), (11), (12), (15), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 10 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_5_0 w_6_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (5), (8), (9), (10), (11), (12), (15), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_9_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_10_0 w_11_0 ∧
  MxxRuntime.concatRows w_8_0 w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (5), (8), (9), (10), (11), (12), (15), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_16_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (5), (8), (9), (10), (11), (12), (15), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_19_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_20_0 w_21_0 ∧
  MxxRuntime.concatRows w_18_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 10 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 8 ∧
  MxxRuntime.sliceMatrix w_26_0 0 1 0 8 w_27_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  parallel_generatedRoot_43.constraints_1 backend hashModel i_0 w_0_0 w_34_0 w_37_0 w_44_0 w_71_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

def parallel_generatedRoot_43 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.1
  let w_23_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.1
  let w_34_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.1
  let w_37_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := inputs.2.2.2.2.1
  let w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.1
  let w_51_0 : Int := inputs.2.2.2.2.2.2.1
  let w_53_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
    let w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (MxxRuntime.matrixPolynomial [(1208925803713492065601409)] : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10),
        let w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 10),
          let w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10 := MxxRuntime.matrixMul w_1_0 w_6_0
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
            ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
              let w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10 := MxxRuntime.matrixMul w_9_0 w_6_0
              ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
                ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 10),
                  ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                    let w_15_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixMul w_12_0 w_14_0
                    ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                      let w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10 := MxxRuntime.matrixMul w_16_0 w_6_0
                      ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
                        ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                          let w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 10 := MxxRuntime.matrixMul w_19_0 w_6_0
                          ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 10),
                            ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 10),
                              ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                let w_25_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixMul w_22_0 w_24_0
                                let w_26_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixAdd w_15_0 w_25_0
                                ∃ (w_27_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8),
                                      ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                        ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                          ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6),
                                            let w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixMul w_29_0 w_32_0
                                            ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                              let w_36_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixMul w_12_0 w_35_0
                                              ∃ (w_38_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 10 8),
                                                let w_39_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixMul w_22_0 w_38_0
                                                let w_40_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8 := MxxRuntime.matrixAdd w_36_0 w_39_0
                                                ∃ (w_41_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                  ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8),
                                                    ∃ (w_43_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 8),
                                                      ∃ (w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1),
                                                        ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
                                                          ∃ (w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                            let w_48_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarLeft w_45_0 w_47_0
                                                            let w_49_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511622593)] : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1)
                                                            let w_50_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarRight w_48_0 w_49_0
                                                            let w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.liftInteger w_51_0
                                                            let w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.multiplyMonomial w_52_0 w_53_0
                                                            ∃ (w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1),
                                                              let w_56_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarRight w_50_0 w_55_0
                                                              let w_57_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarLeft w_45_0 w_30_0
                                                              let w_58_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixSub w_56_0 w_57_0
                                                              ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                let w_60_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_58_0 w_59_0
                                                                ∃ (w_61_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                                                  ∃ (w_62_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6),
                                                                    let w_63_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixMul w_43_0 w_62_0
                                                                    let w_64_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixAdd w_33_0 w_63_0
                                                                    ∃ (w_65_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6),
                                                                      ∃ (w_66_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                        ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                          ∃ (w_68_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                                                            ∃ (w_69_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6),
                                                                              let w_70_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixMul w_29_0 w_69_0
                                                                              ∃ (w_72_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 1),
                                                                                let w_73_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarLeft w_72_0 w_47_0
                                                                                let w_74_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarRight w_73_0 w_49_0
                                                                                let w_75_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarRight w_74_0 w_55_0
                                                                                let w_76_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixMulScalarLeft w_45_0 w_67_0
                                                                                let w_77_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixSub w_75_0 w_76_0
                                                                                ∃ (w_78_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                                  let w_79_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_77_0 w_78_0
                                                                                  ∃ (w_80_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                                                                    ∃ (w_81_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 8 6),
                                                                                      let w_82_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixMul w_43_0 w_81_0
                                                                                      let w_83_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 2 6 := MxxRuntime.matrixAdd w_70_0 w_82_0
                                                                                      ∃ (w_84_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6),
                                                                                        ∃ (w_85_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                                          ∃ (w_86_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6),
                                                                                            ∃ (w_87_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                                              ∃ (w_88_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 6),
                                                                                                ∃ (w_89_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                                                                  parallel_generatedRoot_43.constraints_0 backend hashModel i_0 w_0_0 w_13_0 w_23_0 w_34_0 w_37_0 w_44_0 w_71_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev sequential_generatedRoot_56.constraints_0 (w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_4_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_7_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_8_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_11_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_12_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_15_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_16_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_56 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.1
  let w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.2.1
  let w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
    let w_4_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
      let w_8_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
          let w_16_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_56.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev parallel_generatedRoot_59.constraints_3 (w_30_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_79_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_83_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_85_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_87_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_88_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_89_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_83_0 1 2 0 4 w_88_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593] (1) w_88_0 w_89_0 ∧
  outputs = (w_66_0, (w_85_0, (w_87_0, (w_89_0, (w_30_0, (w_67_0, (w_60_0, (w_79_0, ()))))))))

abbrev parallel_generatedRoot_59.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_62_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_64_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_65_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_69_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_79_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_80_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_81_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_83_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_84_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_85_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_86_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_87_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_88_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_89_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 524288 w_61_0 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_64_0 0 1 0 4 w_65_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593] (1) w_65_0 w_66_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_67_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_67_0 w_68_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_68_0 w_69_0 ∧
  MxxRuntime.modulusReduceRuns w_71_0 w_72_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_78_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_83_0 0 1 0 4 w_84_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593] (1) w_84_0 w_85_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_64_0 1 2 0 4 w_86_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593] (1) w_86_0 w_87_0 ∧
  parallel_generatedRoot_59.constraints_3 w_30_0 w_60_0 w_66_0 w_67_0 w_79_0 w_83_0 w_85_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_59.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_34_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_37_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_27_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_28_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_29_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_30_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_31_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_32_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_35_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_38_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_40_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_41_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_42_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_45_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_47_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_62_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_64_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_65_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_69_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_79_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_80_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_81_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_83_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_84_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_85_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_86_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_87_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_88_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_89_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_26_0 1 2 0 6 w_28_0 ∧
  MxxRuntime.concatRows w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_30_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_30_0 w_31_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_34_0 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_37_0 w_38_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_40_0 0 1 0 6 w_41_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_40_0 1 2 0 6 w_42_0 ∧
  MxxRuntime.concatRows w_41_0 w_42_0 w_43_0 ∧
  MxxRuntime.modulusReduceRuns w_44_0 w_45_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 4 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.modulusReduceRuns w_54_0 w_55_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_60_0 w_61_0 ∧
  parallel_generatedRoot_59.constraints_2 backend hashModel i_0 w_0_0 w_71_0 w_30_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_59.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_13_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_23_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_34_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_37_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_3_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 8) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 8) (w_14_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_18_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8) (w_21_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8) (w_22_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 8) (w_24_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_26_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_27_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_28_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_29_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_30_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_31_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_32_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_35_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_38_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6) (w_40_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_41_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_42_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6) (w_45_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_47_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_62_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_64_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_65_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_69_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_79_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_80_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_81_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4) (w_83_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4) (w_84_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_85_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_86_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_87_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_88_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4) (w_89_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (13), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 8 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_5_0 w_6_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (13), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_9_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_10_0 w_11_0 ∧
  MxxRuntime.concatRows w_8_0 w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (13), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_16_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (13), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_19_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_20_0 w_21_0 ∧
  MxxRuntime.concatRows w_18_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 8 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_26_0 0 1 0 6 w_27_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  parallel_generatedRoot_59.constraints_1 backend hashModel i_0 w_0_0 w_34_0 w_37_0 w_44_0 w_71_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

def parallel_generatedRoot_59 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.1
  let w_23_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.1
  let w_34_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.1
  let w_37_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := inputs.2.2.2.2.1
  let w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.1
  let w_51_0 : Int := inputs.2.2.2.2.2.2.1
  let w_53_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
    let w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (MxxRuntime.matrixPolynomial [(1329227971958197034497972917778543361)] : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8),
        let w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 8),
          let w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8 := MxxRuntime.matrixMul w_1_0 w_6_0
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
            ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
              let w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8 := MxxRuntime.matrixMul w_9_0 w_6_0
              ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
                ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 8),
                  ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                    let w_15_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixMul w_12_0 w_14_0
                    ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                      let w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8 := MxxRuntime.matrixMul w_16_0 w_6_0
                      ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
                        ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                          let w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 8 := MxxRuntime.matrixMul w_19_0 w_6_0
                          ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 8),
                            ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 8),
                              ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                let w_25_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixMul w_22_0 w_24_0
                                let w_26_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixAdd w_15_0 w_25_0
                                ∃ (w_27_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6),
                                      ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                        ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                          ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4),
                                            let w_33_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixMul w_29_0 w_32_0
                                            ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                              let w_36_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixMul w_12_0 w_35_0
                                              ∃ (w_38_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 8 6),
                                                let w_39_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixMul w_22_0 w_38_0
                                                let w_40_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6 := MxxRuntime.matrixAdd w_36_0 w_39_0
                                                ∃ (w_41_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                  ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6),
                                                    ∃ (w_43_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 6),
                                                      ∃ (w_45_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1),
                                                        ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                                                          ∃ (w_47_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                            let w_48_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarLeft w_45_0 w_47_0
                                                            let w_49_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511623297)] : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1)
                                                            let w_50_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarRight w_48_0 w_49_0
                                                            let w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.liftInteger w_51_0
                                                            let w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.multiplyMonomial w_52_0 w_53_0
                                                            ∃ (w_55_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1),
                                                              let w_56_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarRight w_50_0 w_55_0
                                                              let w_57_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarLeft w_45_0 w_30_0
                                                              let w_58_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixSub w_56_0 w_57_0
                                                              ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                let w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_58_0 w_59_0
                                                                ∃ (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                                                  ∃ (w_62_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4),
                                                                    let w_63_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixMul w_43_0 w_62_0
                                                                    let w_64_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixAdd w_33_0 w_63_0
                                                                    ∃ (w_65_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4),
                                                                      ∃ (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                        ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                          ∃ (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                                                            ∃ (w_69_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4),
                                                                              let w_70_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixMul w_29_0 w_69_0
                                                                              ∃ (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 1),
                                                                                let w_73_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarLeft w_72_0 w_47_0
                                                                                let w_74_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarRight w_73_0 w_49_0
                                                                                let w_75_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarRight w_74_0 w_55_0
                                                                                let w_76_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixMulScalarLeft w_45_0 w_67_0
                                                                                let w_77_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixSub w_75_0 w_76_0
                                                                                ∃ (w_78_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                                  let w_79_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_77_0 w_78_0
                                                                                  ∃ (w_80_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                                                                    ∃ (w_81_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 6 4),
                                                                                      let w_82_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixMul w_43_0 w_81_0
                                                                                      let w_83_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 2 4 := MxxRuntime.matrixAdd w_70_0 w_82_0
                                                                                      ∃ (w_84_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4),
                                                                                        ∃ (w_85_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                                          ∃ (w_86_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4),
                                                                                            ∃ (w_87_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                                              ∃ (w_88_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 4),
                                                                                                ∃ (w_89_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                                                                  parallel_generatedRoot_59.constraints_0 backend hashModel i_0 w_0_0 w_13_0 w_23_0 w_34_0 w_37_0 w_44_0 w_71_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev sequential_generatedRoot_72.constraints_0 (w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_4_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_7_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_8_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_11_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_12_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_15_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_72 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.1
  let w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.2.1
  let w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
    let w_4_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
      let w_8_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
          let w_16_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_72.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev parallel_generatedRoot_75.constraints_3 (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_83_0 1 2 0 2 w_88_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297] (1) w_88_0 w_89_0 ∧
  outputs = (w_66_0, (w_85_0, (w_87_0, (w_89_0, (w_30_0, (w_67_0, (w_60_0, (w_79_0, ()))))))))

abbrev parallel_generatedRoot_75.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_72_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_86_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 524288 w_61_0 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_64_0 0 1 0 2 w_65_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297] (1) w_65_0 w_66_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 52, 47, 53]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_67_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_67_0 w_68_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_68_0 w_69_0 ∧
  MxxRuntime.modulusReduceRuns w_71_0 w_72_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_78_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_83_0 0 1 0 2 w_84_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297] (1) w_84_0 w_85_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_64_0 1 2 0 2 w_86_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297] (1) w_86_0 w_87_0 ∧
  parallel_generatedRoot_75.constraints_3 w_30_0 w_60_0 w_66_0 w_67_0 w_79_0 w_83_0 w_85_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_75.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_34_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_37_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_32_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_35_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_38_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_41_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_42_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_43_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_45_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 2) (w_47_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_72_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_86_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_26_0 1 2 0 4 w_28_0 ∧
  MxxRuntime.concatRows w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 54, 51, 100, 98, 52, 52, 53, 51, 52, 50, 54, 49, 48, 57, 54, 57, 50, 50, 97, 50, 99, 102, 100, 49, 100, 99, 98, 53, 56, 52, 99, 102, 55, 101, 101, 99, 97, 50, 99, 57, 101, 48, 49, 53, 98, 100, 97, 98, 97, 51, 100, 101, 100, 101, 100, 48, 99, 50, 101, 51, 99, 51, 57, 56, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 48, 44, 32, 50, 52, 54, 44, 32, 50, 52, 52, 44, 32, 52, 51, 44, 32, 49, 54, 55, 44, 32, 54, 49, 44, 32, 54, 50, 44, 32, 49, 50, 44, 32, 49, 54, 50, 44, 32, 49, 57, 53, 44, 32, 50, 52, 49, 44, 32, 49, 53, 53, 44, 32, 49, 57, 50, 44, 32, 49, 56, 48, 44, 32, 50, 49, 52, 44, 32, 55, 51, 44, 32, 53, 53, 44, 32, 49, 56, 54, 44, 32, 49, 51, 49, 44, 32, 49, 57, 48, 44, 32, 49, 52, 53, 44, 32, 49, 55, 44, 32, 50, 51, 52, 44, 32, 50, 48, 48, 44, 32, 49, 49, 49, 44, 32, 53, 52, 44, 32, 53, 51, 44, 32, 50, 48, 49, 44, 32, 49, 55, 54, 44, 32, 50, 50, 52, 44, 32, 50, 50, 55, 44, 32, 56, 48, 93, 41, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 52, 47, 53]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_30_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_30_0 w_31_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_34_0 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_37_0 w_38_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_40_0 0 1 0 4 w_41_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_40_0 1 2 0 4 w_42_0 ∧
  MxxRuntime.concatRows w_41_0 w_42_0 w_43_0 ∧
  MxxRuntime.modulusReduceRuns w_44_0 w_45_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 2 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.modulusReduceRuns w_54_0 w_55_0 ∧
  MxxRuntime.gaussianSample ((4/1)) (26) w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 4 w_60_0 w_61_0 ∧
  parallel_generatedRoot_75.constraints_2 backend hashModel i_0 w_0_0 w_71_0 w_30_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev parallel_generatedRoot_75.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_13_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_23_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_34_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_37_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_3_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 6) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_8_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_11_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_12_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 6) (w_14_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_18_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) (w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6) (w_21_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6) (w_22_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 6) (w_24_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_27_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_28_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_29_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_32_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_35_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_38_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4) (w_40_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_41_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_42_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_43_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4) (w_45_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 2) (w_47_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_72_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_86_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(3), (6), (7), (13), (14), (16), (16), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 6 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 14 w_5_0 w_6_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(3), (6), (7), (13), (14), (16), (16), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_9_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_10_0 w_11_0 ∧
  MxxRuntime.concatRows w_8_0 w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(3), (6), (7), (13), (14), (16), (16), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_16_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 14 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(3), (6), (7), (13), (14), (16), (16), (16), (16), (16), (16), (16), (16), (16)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_19_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841] (1) w_20_0 w_21_0 ∧
  MxxRuntime.concatRows w_18_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 6 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 4 ∧
  4 ≤ 4 ∧
  MxxRuntime.sliceMatrix w_26_0 0 1 0 4 w_27_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  parallel_generatedRoot_75.constraints_1 backend hashModel i_0 w_0_0 w_34_0 w_37_0 w_44_0 w_71_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

def parallel_generatedRoot_75 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.1
  let w_23_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.1
  let w_34_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.1
  let w_37_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := inputs.2.2.2.2.1
  let w_44_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.1
  let w_51_0 : Int := inputs.2.2.2.2.2.2.1
  let w_53_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_71_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
    let w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (MxxRuntime.matrixPolynomial [(1461501604243759924967666999996056787893117755073)] : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6),
        let w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 14 6),
          let w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6 := MxxRuntime.matrixMul w_1_0 w_6_0
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
            ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
              let w_10_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6 := MxxRuntime.matrixMul w_9_0 w_6_0
              ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
                ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 6),
                  ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                    let w_15_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixMul w_12_0 w_14_0
                    ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                      let w_17_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6 := MxxRuntime.matrixMul w_16_0 w_6_0
                      ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
                        ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14),
                          let w_20_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 6 := MxxRuntime.matrixMul w_19_0 w_6_0
                          ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 6),
                            ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 6),
                              ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                let w_25_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixMul w_22_0 w_24_0
                                let w_26_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixAdd w_15_0 w_25_0
                                ∃ (w_27_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                  ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                    ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4),
                                      ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                        ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2),
                                          ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2),
                                            let w_33_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixMul w_29_0 w_32_0
                                            ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                              let w_36_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixMul w_12_0 w_35_0
                                              ∃ (w_38_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 6 4),
                                                let w_39_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixMul w_22_0 w_38_0
                                                let w_40_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4 := MxxRuntime.matrixAdd w_36_0 w_39_0
                                                ∃ (w_41_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                  ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4),
                                                    ∃ (w_43_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 4),
                                                      ∃ (w_45_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                        ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 2),
                                                          ∃ (w_47_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                            let w_48_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarLeft w_45_0 w_47_0
                                                            let w_49_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := (MxxRuntime.matrixPolynomial [(1099511625409)] : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1)
                                                            let w_50_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarRight w_48_0 w_49_0
                                                            let w_52_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.liftInteger w_51_0
                                                            let w_54_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := MxxRuntime.multiplyMonomial w_52_0 w_53_0
                                                            ∃ (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                              let w_56_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarRight w_50_0 w_55_0
                                                              let w_57_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarLeft w_45_0 w_30_0
                                                              let w_58_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixSub w_56_0 w_57_0
                                                              ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                let w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_58_0 w_59_0
                                                                ∃ (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2),
                                                                  ∃ (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2),
                                                                    let w_63_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixMul w_43_0 w_62_0
                                                                    let w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixAdd w_33_0 w_63_0
                                                                    ∃ (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2),
                                                                      ∃ (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                        ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                          ∃ (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2),
                                                                            ∃ (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2),
                                                                              let w_70_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixMul w_29_0 w_69_0
                                                                              ∃ (w_72_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                                let w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarLeft w_72_0 w_47_0
                                                                                let w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarRight w_73_0 w_49_0
                                                                                let w_75_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarRight w_74_0 w_55_0
                                                                                let w_76_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixMulScalarLeft w_45_0 w_67_0
                                                                                let w_77_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixSub w_75_0 w_76_0
                                                                                ∃ (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                                  let w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_77_0 w_78_0
                                                                                  ∃ (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 4 2),
                                                                                    ∃ (w_81_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 4 2),
                                                                                      let w_82_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixMul w_43_0 w_81_0
                                                                                      let w_83_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 2 2 := MxxRuntime.matrixAdd w_70_0 w_82_0
                                                                                      ∃ (w_84_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2),
                                                                                        ∃ (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                                          ∃ (w_86_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2),
                                                                                            ∃ (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                                              ∃ (w_88_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 2),
                                                                                                ∃ (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                                                                                  parallel_generatedRoot_75.constraints_0 backend hashModel i_0 w_0_0 w_13_0 w_23_0 w_34_0 w_37_0 w_44_0 w_71_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_35_0 w_38_0 w_40_0 w_41_0 w_42_0 w_43_0 w_45_0 w_46_0 w_47_0 w_54_0 w_55_0 w_59_0 w_60_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_72_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 outputs

abbrev sequential_generatedRoot_88.constraints_0 (w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_88 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.1
  let w_1_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.1
  let w_6_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
    let w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
      let w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
          let w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_88.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_89.constraints_3 (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_43_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_41_0 1 2 0 1 w_70_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_66_0 1 2 0 1 w_73_0 ∧
  outputs = (w_43_0, (w_68_0, (w_71_0, (w_74_0, ()))))

abbrev sequential_generatedRoot_89.constraints_2 (backend : MxxRuntime.BackendContext) (w_50_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_12_0 : Int) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_43_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_52_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_53_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_56_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_57_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_62_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_64_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ w_12_0 ∧
  w_12_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_50_0 w_12_0 w_51_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_51_0 w_52_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_52_0 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_55_0 0 1 0 1 w_56_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_56_0 w_57_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_57_0 w_58_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_58_0 w_59_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_55_0 1 2 0 1 w_61_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_61_0 w_62_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_62_0 w_63_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_63_0 w_64_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_66_0 0 1 0 1 w_67_0 ∧
  sequential_generatedRoot_89.constraints_3 w_41_0 w_43_0 w_66_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 outputs

abbrev sequential_generatedRoot_89.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_45_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_50_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_29_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_37_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_38_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_42_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_43_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_47_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_48_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_52_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_53_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_56_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_57_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_62_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_64_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 1 w_28_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_29_0 w_30_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_30_0 w_31_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_33_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_34_0 ∧
  MxxRuntime.concatRows w_33_0 w_34_0 w_35_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 1 w_36_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_36_0 w_37_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_37_0 w_38_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_38_0 w_39_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_41_0 0 1 0 1 w_42_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_45_0 w_12_0 w_46_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_46_0 w_47_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_47_0 w_48_0 ∧
  sequential_generatedRoot_89.constraints_2 backend w_50_0 w_12_0 w_41_0 w_43_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_61_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 outputs

abbrev sequential_generatedRoot_89.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_11_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_22_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_45_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_50_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2) (w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2) (w_12_0 : Int) (w_13_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2) (w_23_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_24_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_29_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_30_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2) (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_37_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_38_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_42_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_43_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_47_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_48_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_52_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_53_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_56_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_57_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_62_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1) (w_64_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1) (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_70_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_71_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (5), .integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 16 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 1048576 2 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 524288 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  sequential_generatedRoot_89.constraints_1 backend hashModel w_1_0 w_45_0 w_50_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_43_0 w_46_0 w_47_0 w_48_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_61_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 outputs

def sequential_generatedRoot_89 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.1
  let w_44_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.1
  let w_69_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.2.1
  let w_72_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_45_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (15), (17), (21), (25), (27), (45), (51), (55), (61), (63)]
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
    ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
      ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2),
        ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
          ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
            ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2),
              let w_12_0 : Int := (Int.ofNat i_0)
              ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                  ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                    let w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_10_0 w_15_0
                    ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                      ∃ (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                        ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2),
                          ∃ (w_23_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                            ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                              ∃ (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                let w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_21_0 w_25_0
                                let w_27_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixAdd w_16_0 w_26_0
                                ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                  ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                    ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                      ∃ (w_31_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                        let w_32_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_5_0 w_31_0
                                        ∃ (w_33_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                          ∃ (w_34_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2),
                                            ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 2),
                                              ∃ (w_36_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                ∃ (w_37_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                  ∃ (w_38_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                                    ∃ (w_39_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                                      let w_40_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_35_0 w_39_0
                                                      let w_41_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixAdd w_32_0 w_40_0
                                                      ∃ (w_42_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                        let w_43_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixAdd w_0_0 w_42_0
                                                        ∃ (w_46_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                          ∃ (w_47_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                                            ∃ (w_48_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                                              let w_49_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_10_0 w_48_0
                                                              ∃ (w_51_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                                ∃ (w_52_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                                                  ∃ (w_53_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                                                    let w_54_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_21_0 w_53_0
                                                                    let w_55_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixAdd w_49_0 w_54_0
                                                                    ∃ (w_56_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                      ∃ (w_57_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                                        ∃ (w_58_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                                                          ∃ (w_59_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                                                            let w_60_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_5_0 w_59_0
                                                                            ∃ (w_61_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                              ∃ (w_62_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1),
                                                                                ∃ (w_63_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 2 1),
                                                                                  ∃ (w_64_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1),
                                                                                    let w_65_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixMul w_35_0 w_64_0
                                                                                    let w_66_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 2 1 := MxxRuntime.matrixAdd w_60_0 w_65_0
                                                                                    ∃ (w_67_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                                      let w_68_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixAdd w_44_0 w_67_0
                                                                                      ∃ (w_70_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                                        let w_71_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                        ∃ (w_73_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1),
                                                                                          let w_74_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                          sequential_generatedRoot_89.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_43_0 w_46_0 w_47_0 w_48_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_61_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_3_1 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_3_2 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_3_3 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12
  w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12
  w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12
  w_8_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12
  w_11_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_11_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_13_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_15_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_16_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_19_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_21_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_22_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_24_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_24_1 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_24_2 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_24_3 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10
  w_27_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_27_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_28_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_29_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_31_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_32_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_34_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_35_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_37_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_38_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_40_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_40_1 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_40_2 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_40_3 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8
  w_43_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_43_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_44_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_48_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_50_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_51_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_53_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_54_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_56_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_56_1 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_56_2 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_56_3 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6
  w_59_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_59_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_63_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_70_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_72_1 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_72_2 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_72_3 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4
  w_75_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_75_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_76_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_77_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_82_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_83_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_86_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_88_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_88_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_88_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_88_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2
  w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_89_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_89_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_89_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  w_90_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_91_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_3_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_1 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_2 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_3 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_11_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_27_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_43_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_59_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_75_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_90_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_91_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) : Prop :=
  0 ≤ (12) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 × Unit) => sequential_generatedRoot_88 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_75_0, (w_75_1, (w_75_2, (w_75_3, ())))))))) next) (Int.toNat (12)) (w_78_0, (w_81_0, (w_84_0, (w_87_0, ())))) (w_88_0, (w_88_1, (w_88_2, (w_88_3, ())))) ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) => sequential_generatedRoot_89 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_0_0, (w_88_0, (w_88_2, (w_3_0, (w_88_1, (w_88_3, (w_3_2, (w_3_1, (w_3_3, ()))))))))))))) next) (Int.toNat (16)) (w_4_0, (w_4_0, (w_4_0, (w_4_0, ())))) (w_89_0, (w_89_1, (w_89_2, (w_89_3, ())))) ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_89_0 w_90_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_89_1 w_91_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_89_2 w_92_0 ∧
  MxxRuntime.blockModSwitchRuns [1099511626049, 1099511625409] (1) w_89_3 w_93_0 ∧
  outputs = (w_3_0, (w_3_1, (w_3_2, (w_3_3, (w_90_0, (w_91_0, (w_92_0, (w_93_0, (w_11_6, (w_11_7, (w_27_6, (w_27_7, (w_43_6, (w_43_7, (w_59_6, (w_59_7, (w_75_6, (w_75_7, ()))))))))))))))))))

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_57_0 : Fin 14 → Int) (w_58_0 : Fin 14 → Int) (w_73_0 : Fin 14 → Int) (w_74_0 : Fin 14 → Int) (w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_3_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_1 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_2 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_3 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_11_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_27_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_43_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_44_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_48_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_49_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_50_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_51_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_52_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_53_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_54_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_1 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_2 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_3 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_59_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_63_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_70_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_71_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_1 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_2 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_3 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_75_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_76_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_77_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_82_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_86_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_90_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_91_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) : Prop :=
  MxxRuntime.familyGetStatic w_43_0 (0) w_44_0 ∧
  MxxRuntime.familyGetStatic w_43_0 (1) w_45_0 ∧
  MxxRuntime.familyGetStatic w_43_1 (0) w_47_0 ∧
  MxxRuntime.familyGetStatic w_43_1 (1) w_48_0 ∧
  MxxRuntime.familyGetStatic w_43_2 (0) w_50_0 ∧
  MxxRuntime.familyGetStatic w_43_2 (1) w_51_0 ∧
  MxxRuntime.familyGetStatic w_43_3 (0) w_53_0 ∧
  MxxRuntime.familyGetStatic w_43_3 (1) w_54_0 ∧
  0 ≤ (12) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 × Unit) => sequential_generatedRoot_56 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_43_0, (w_43_1, (w_43_2, (w_43_3, ())))))))) next) (Int.toNat (12)) (w_46_0, (w_49_0, (w_52_0, (w_55_0, ())))) (w_56_0, (w_56_1, (w_56_2, (w_56_3, ())))) ∧
  (14) = 14 ∧
  (∀ i : Fin 14, parallel_generatedRoot_59 backend hashModel params i (w_0_0, (w_56_0, (w_56_2, (w_56_1, (w_56_3, (w_1_0, ((w_57_0 i), ((w_58_0 i), (w_2_0, ()))))))))) ((w_59_0 i), ((w_59_1 i), ((w_59_2 i), ((w_59_3 i), ((w_59_4 i), ((w_59_5 i), ((w_59_6 i), ((w_59_7 i), ()))))))))) ∧
  MxxRuntime.familyGetStatic w_59_0 (0) w_60_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (1) w_61_0 ∧
  MxxRuntime.familyGetStatic w_59_1 (0) w_63_0 ∧
  MxxRuntime.familyGetStatic w_59_1 (1) w_64_0 ∧
  MxxRuntime.familyGetStatic w_59_2 (0) w_66_0 ∧
  MxxRuntime.familyGetStatic w_59_2 (1) w_67_0 ∧
  MxxRuntime.familyGetStatic w_59_3 (0) w_69_0 ∧
  MxxRuntime.familyGetStatic w_59_3 (1) w_70_0 ∧
  0 ≤ (12) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 × Unit) => sequential_generatedRoot_72 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_59_0, (w_59_1, (w_59_2, (w_59_3, ())))))))) next) (Int.toNat (12)) (w_62_0, (w_65_0, (w_68_0, (w_71_0, ())))) (w_72_0, (w_72_1, (w_72_2, (w_72_3, ())))) ∧
  (14) = 14 ∧
  (∀ i : Fin 14, parallel_generatedRoot_75 backend hashModel params i (w_0_0, (w_72_0, (w_72_2, (w_72_1, (w_72_3, (w_1_0, ((w_73_0 i), ((w_74_0 i), (w_2_0, ()))))))))) ((w_75_0 i), ((w_75_1 i), ((w_75_2 i), ((w_75_3 i), ((w_75_4 i), ((w_75_5 i), ((w_75_6 i), ((w_75_7 i), ()))))))))) ∧
  MxxRuntime.familyGetStatic w_75_0 (0) w_76_0 ∧
  MxxRuntime.familyGetStatic w_75_0 (1) w_77_0 ∧
  MxxRuntime.familyGetStatic w_75_1 (0) w_79_0 ∧
  MxxRuntime.familyGetStatic w_75_1 (1) w_80_0 ∧
  MxxRuntime.familyGetStatic w_75_2 (0) w_82_0 ∧
  MxxRuntime.familyGetStatic w_75_2 (1) w_83_0 ∧
  MxxRuntime.familyGetStatic w_75_3 (0) w_85_0 ∧
  MxxRuntime.familyGetStatic w_75_3 (1) w_86_0 ∧
  generatedRoot.constraints_2 backend hashModel params w_0_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_11_6 w_11_7 w_27_6 w_27_7 w_43_6 w_43_7 w_59_6 w_59_7 w_75_0 w_75_1 w_75_2 w_75_3 w_75_6 w_75_7 w_78_0 w_81_0 w_84_0 w_87_0 w_88_0 w_88_1 w_88_2 w_88_3 w_89_0 w_89_1 w_89_2 w_89_3 w_90_0 w_91_0 w_92_0 w_93_0 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_9_0 : Fin 14 → Int) (w_10_0 : Fin 14 → Int) (w_25_0 : Fin 14 → Int) (w_26_0 : Fin 14 → Int) (w_41_0 : Fin 14 → Int) (w_42_0 : Fin 14 → Int) (w_57_0 : Fin 14 → Int) (w_58_0 : Fin 14 → Int) (w_73_0 : Fin 14 → Int) (w_74_0 : Fin 14 → Int) (w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_3_0 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_1 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_2 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_3_3 : Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_6_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_7_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_8_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) (w_11_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_11_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_18_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_19_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_20_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_23_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_24_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_24_1 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_24_2 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_24_3 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) (w_27_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_27_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_32_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_34_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_35_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_36_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_38_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_39_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_40_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_40_1 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_40_2 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_40_3 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) (w_43_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_43_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_44_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_45_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_47_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_48_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_49_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_50_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_51_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_52_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_53_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_54_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_1 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_2 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_56_3 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) (w_59_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_59_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_60_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_61_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_63_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_64_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_66_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_67_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_69_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_70_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_71_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_1 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_2 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_72_3 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) (w_75_0 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_1 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_2 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_3 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_4 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_5 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_6 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_75_7 : Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_76_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_77_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_79_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_80_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_81_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_82_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_83_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_84_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_85_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_86_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_88_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) (w_89_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_2 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_89_3 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_90_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_91_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_92_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_93_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (outputs : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) : Prop :=
  (16) = 16 ∧
  (∀ i : Fin 16, parallel_generatedRoot_3 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_3_0 i), ((w_3_1 i), ((w_3_2 i), ((w_3_3 i), ()))))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_5_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_8_0 ∧
  (14) = 14 ∧
  (∀ i : Fin 14, parallel_generatedRoot_11 backend hashModel params i (w_0_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_1_0, ((w_9_0 i), ((w_10_0 i), (w_2_0, ()))))))))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ((w_11_3 i), ((w_11_4 i), ((w_11_5 i), ((w_11_6 i), ((w_11_7 i), ()))))))))) ∧
  MxxRuntime.familyGetStatic w_11_0 (0) w_12_0 ∧
  MxxRuntime.familyGetStatic w_11_0 (1) w_13_0 ∧
  MxxRuntime.familyGetStatic w_11_1 (0) w_15_0 ∧
  MxxRuntime.familyGetStatic w_11_1 (1) w_16_0 ∧
  MxxRuntime.familyGetStatic w_11_2 (0) w_18_0 ∧
  MxxRuntime.familyGetStatic w_11_2 (1) w_19_0 ∧
  MxxRuntime.familyGetStatic w_11_3 (0) w_21_0 ∧
  MxxRuntime.familyGetStatic w_11_3 (1) w_22_0 ∧
  0 ≤ (12) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 × Unit) => sequential_generatedRoot_24 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_11_0, (w_11_1, (w_11_2, (w_11_3, ())))))))) next) (Int.toNat (12)) (w_14_0, (w_17_0, (w_20_0, (w_23_0, ())))) (w_24_0, (w_24_1, (w_24_2, (w_24_3, ())))) ∧
  (14) = 14 ∧
  (∀ i : Fin 14, parallel_generatedRoot_27 backend hashModel params i (w_0_0, (w_24_0, (w_24_2, (w_24_1, (w_24_3, (w_1_0, ((w_25_0 i), ((w_26_0 i), (w_2_0, ()))))))))) ((w_27_0 i), ((w_27_1 i), ((w_27_2 i), ((w_27_3 i), ((w_27_4 i), ((w_27_5 i), ((w_27_6 i), ((w_27_7 i), ()))))))))) ∧
  MxxRuntime.familyGetStatic w_27_0 (0) w_28_0 ∧
  MxxRuntime.familyGetStatic w_27_0 (1) w_29_0 ∧
  MxxRuntime.familyGetStatic w_27_1 (0) w_31_0 ∧
  MxxRuntime.familyGetStatic w_27_1 (1) w_32_0 ∧
  MxxRuntime.familyGetStatic w_27_2 (0) w_34_0 ∧
  MxxRuntime.familyGetStatic w_27_2 (1) w_35_0 ∧
  MxxRuntime.familyGetStatic w_27_3 (0) w_37_0 ∧
  MxxRuntime.familyGetStatic w_27_3 (1) w_38_0 ∧
  0 ≤ (12) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 × Unit) => sequential_generatedRoot_40 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_27_0, (w_27_1, (w_27_2, (w_27_3, ())))))))) next) (Int.toNat (12)) (w_30_0, (w_33_0, (w_36_0, (w_39_0, ())))) (w_40_0, (w_40_1, (w_40_2, (w_40_3, ())))) ∧
  (14) = 14 ∧
  (∀ i : Fin 14, parallel_generatedRoot_43 backend hashModel params i (w_0_0, (w_40_0, (w_40_2, (w_40_1, (w_40_3, (w_1_0, ((w_41_0 i), ((w_42_0 i), (w_2_0, ()))))))))) ((w_43_0 i), ((w_43_1 i), ((w_43_2 i), ((w_43_3 i), ((w_43_4 i), ((w_43_5 i), ((w_43_6 i), ((w_43_7 i), ()))))))))) ∧
  generatedRoot.constraints_1 backend hashModel params w_0_0 w_1_0 w_57_0 w_58_0 w_73_0 w_74_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_11_6 w_11_7 w_27_6 w_27_7 w_43_0 w_43_1 w_43_2 w_43_3 w_43_6 w_43_7 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_56_1 w_56_2 w_56_3 w_59_0 w_59_1 w_59_2 w_59_3 w_59_4 w_59_5 w_59_6 w_59_7 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_72_1 w_72_2 w_72_3 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_88_1 w_88_2 w_88_3 w_89_0 w_89_1 w_89_2 w_89_3 w_90_0 w_91_0 w_92_0 w_93_0 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.1
  let w_9_0 : Fin 14 → Int := inputs.2.2.1
  let w_10_0 : Fin 14 → Int := inputs.2.2.2.1
  let w_25_0 : Fin 14 → Int := inputs.2.2.2.2.1
  let w_26_0 : Fin 14 → Int := inputs.2.2.2.2.2.1
  let w_41_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.1
  let w_42_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.1
  let w_57_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_58_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_73_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (1 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
  let w_3_0 := witness.w_3_0
  let w_3_1 := witness.w_3_1
  let w_3_2 := witness.w_3_2
  let w_3_3 := witness.w_3_3
  let w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := (0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1)
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_11_3 := witness.w_11_3
  let w_11_4 := witness.w_11_4
  let w_11_5 := witness.w_11_5
  let w_11_6 := witness.w_11_6
  let w_11_7 := witness.w_11_7
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_15_0 w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_18_0 w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10 := MxxRuntime.matrixAdd w_21_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_24_3 := witness.w_24_3
  let w_27_0 := witness.w_27_0
  let w_27_1 := witness.w_27_1
  let w_27_2 := witness.w_27_2
  let w_27_3 := witness.w_27_3
  let w_27_4 := witness.w_27_4
  let w_27_5 := witness.w_27_5
  let w_27_6 := witness.w_27_6
  let w_27_7 := witness.w_27_7
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_28_0 w_29_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_31_0 w_32_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_34_0 w_35_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8 := MxxRuntime.matrixAdd w_37_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_40_1 := witness.w_40_1
  let w_40_2 := witness.w_40_2
  let w_40_3 := witness.w_40_3
  let w_43_0 := witness.w_43_0
  let w_43_1 := witness.w_43_1
  let w_43_2 := witness.w_43_2
  let w_43_3 := witness.w_43_3
  let w_43_4 := witness.w_43_4
  let w_43_5 := witness.w_43_5
  let w_43_6 := witness.w_43_6
  let w_43_7 := witness.w_43_7
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_44_0 w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_50_0 w_51_0
  let w_53_0 := witness.w_53_0
  let w_54_0 := witness.w_54_0
  let w_55_0 : Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6 := MxxRuntime.matrixAdd w_53_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_56_1 := witness.w_56_1
  let w_56_2 := witness.w_56_2
  let w_56_3 := witness.w_56_3
  let w_59_0 := witness.w_59_0
  let w_59_1 := witness.w_59_1
  let w_59_2 := witness.w_59_2
  let w_59_3 := witness.w_59_3
  let w_59_4 := witness.w_59_4
  let w_59_5 := witness.w_59_5
  let w_59_6 := witness.w_59_6
  let w_59_7 := witness.w_59_7
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_60_0 w_61_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4 := MxxRuntime.matrixAdd w_69_0 w_70_0
  let w_72_0 := witness.w_72_0
  let w_72_1 := witness.w_72_1
  let w_72_2 := witness.w_72_2
  let w_72_3 := witness.w_72_3
  let w_75_0 := witness.w_75_0
  let w_75_1 := witness.w_75_1
  let w_75_2 := witness.w_75_2
  let w_75_3 := witness.w_75_3
  let w_75_4 := witness.w_75_4
  let w_75_5 := witness.w_75_5
  let w_75_6 := witness.w_75_6
  let w_75_7 := witness.w_75_7
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_76_0 w_77_0
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_81_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_79_0 w_80_0
  let w_82_0 := witness.w_82_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_82_0 w_83_0
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2 := MxxRuntime.matrixAdd w_85_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_88_1 := witness.w_88_1
  let w_88_2 := witness.w_88_2
  let w_88_3 := witness.w_88_3
  let w_89_0 := witness.w_89_0
  let w_89_1 := witness.w_89_1
  let w_89_2 := witness.w_89_2
  let w_89_3 := witness.w_89_3
  let w_90_0 := witness.w_90_0
  let w_91_0 := witness.w_91_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  generatedRoot.constraints_0 backend hashModel params w_0_0 w_1_0 w_9_0 w_10_0 w_25_0 w_26_0 w_41_0 w_42_0 w_57_0 w_58_0 w_73_0 w_74_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_11_0 w_11_1 w_11_2 w_11_3 w_11_4 w_11_5 w_11_6 w_11_7 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_24_3 w_27_0 w_27_1 w_27_2 w_27_3 w_27_4 w_27_5 w_27_6 w_27_7 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_40_1 w_40_2 w_40_3 w_43_0 w_43_1 w_43_2 w_43_3 w_43_4 w_43_5 w_43_6 w_43_7 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_56_1 w_56_2 w_56_3 w_59_0 w_59_1 w_59_2 w_59_3 w_59_4 w_59_5 w_59_6 w_59_7 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_72_1 w_72_2 w_72_3 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_88_1 w_88_2 w_88_3 w_89_0 w_89_1 w_89_2 w_89_3 w_90_0 w_91_0 w_92_0 w_93_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_preprocess
