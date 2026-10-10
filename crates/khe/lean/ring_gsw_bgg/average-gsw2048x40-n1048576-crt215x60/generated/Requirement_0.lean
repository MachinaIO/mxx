import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  «instance_width» : Int
  «witness_width» : Int
  «depth» : Int
  «max_layer_width» : Int

abbrev parallel_generatedRoot_65.constraints_0 (w_3_0 : Bool) (outputs : Bool) : Prop :=
  outputs = w_3_0

def parallel_generatedRoot_65 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := true
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 1
  let w_3_0 : Bool := decide (w_1_0 = w_2_0)
  parallel_generatedRoot_65.constraints_0 w_3_0 outputs

abbrev parallel_sequential_generatedRoot_70_17.constraints_0 (w_16_0 : Fin 5435791071440480 → Int) (w_29_0 : Fin 5435791071440480 → Int) (w_38_0 : Fin 5435791071440480 → Int) (w_13_0 : Int) (w_20_0 : Int) (w_21_0 : Int) (w_30_0 : Int) (w_39_0 : Int) (w_45_0 : Bool) (w_190_0 : Bool) (w_191_0 : Bool) (w_201_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_20_0 ∧
  w_20_0 < 5435791071440480 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_20_0 w_21_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 5435791071440480 ∧
  MxxRuntime.familyGetDynamic w_29_0 w_20_0 w_30_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 5435791071440480 ∧
  MxxRuntime.familyGetDynamic w_38_0 w_20_0 w_39_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_13_0 [w_45_0, w_190_0] w_191_0 ∧
  outputs = w_201_0

def parallel_sequential_generatedRoot_70_17 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Bool × Int × (Fin 5435791071440480 → Int) × Int × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × Int × Bool × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_2_0 : Bool := inputs.1
  let w_9_0 : Int := inputs.2.1
  let w_16_0 : Fin 5435791071440480 → Int := inputs.2.2.1
  let w_17_0 : Int := inputs.2.2.2.1
  let w_29_0 : Fin 5435791071440480 → Int := inputs.2.2.2.2.1
  let w_38_0 : Fin 5435791071440480 → Int := inputs.2.2.2.2.2.1
  let w_101_0 : Int := inputs.2.2.2.2.2.2.1
  let w_197_0 : Bool := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : Bool := true
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := if w_2_0 then 1 else 0
  let w_4_0 : Int := (w_1_0 + w_3_0)
  let w_5_0 : Int := 2
  let w_6_0 : Bool := decide (w_4_0 = w_5_0)
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 : Int := (Int.ofNat i_1)
  let w_10_0 : Int := 1
  let w_11_0 : Int := (w_9_0 - w_10_0)
  let w_12_0 : Bool := decide (w_8_0 ≤ w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_14_0 : Bool := true
  let w_15_0 : Int := if w_14_0 then 1 else 0
  let w_18_0 : Int := params.«max_layer_width»
  let w_19_0 : Int := (w_17_0 * w_18_0)
  let w_20_0 : Int := (w_19_0 + w_8_0)
  ∃ (w_21_0 : Int),
    let w_22_0 : Int := 0
    let w_23_0 : Bool := decide (w_21_0 = w_22_0)
    let w_24_0 : Int := if w_23_0 then 1 else 0
    let w_25_0 : Int := (w_15_0 + w_24_0)
    let w_26_0 : Int := 2
    let w_27_0 : Bool := decide (w_25_0 = w_26_0)
    let w_28_0 : Int := if w_27_0 then 1 else 0
    ∃ (w_30_0 : Int),
      let w_31_0 : Int := 0
      let w_32_0 : Bool := decide (w_30_0 = w_31_0)
      let w_33_0 : Int := if w_32_0 then 1 else 0
      let w_34_0 : Int := (w_28_0 + w_33_0)
      let w_35_0 : Int := 2
      let w_36_0 : Bool := decide (w_34_0 = w_35_0)
      let w_37_0 : Int := if w_36_0 then 1 else 0
      ∃ (w_39_0 : Int),
        let w_40_0 : Int := 0
        let w_41_0 : Bool := decide (w_39_0 = w_40_0)
        let w_42_0 : Int := if w_41_0 then 1 else 0
        let w_43_0 : Int := (w_37_0 + w_42_0)
        let w_44_0 : Int := 2
        let w_45_0 : Bool := decide (w_43_0 = w_44_0)
        let w_46_0 : Int := 0
        let w_47_0 : Bool := true
        let w_48_0 : Int := if w_47_0 then 1 else 0
        let w_49_0 : Int := 0
        let w_50_0 : Bool := decide (w_21_0 = w_49_0)
        let w_51_0 : Int := if w_50_0 then 1 else 0
        let w_52_0 : Int := (w_48_0 + w_51_0)
        let w_53_0 : Int := 2
        let w_54_0 : Bool := decide (w_52_0 = w_53_0)
        let w_55_0 : Int := if w_54_0 then 1 else 0
        let w_56_0 : Int := 0
        let w_57_0 : Bool := decide (w_30_0 = w_56_0)
        let w_58_0 : Int := if w_57_0 then 1 else 0
        let w_59_0 : Int := (w_55_0 + w_58_0)
        let w_60_0 : Int := 2
        let w_61_0 : Bool := decide (w_59_0 = w_60_0)
        let w_62_0 : Int := if w_61_0 then 1 else 0
        let w_63_0 : Bool := decide (w_39_0 = w_56_0)
        let w_64_0 : Int := if w_63_0 then 1 else 0
        let w_65_0 : Int := (w_62_0 + w_64_0)
        let w_66_0 : Int := 2
        let w_67_0 : Bool := decide (w_65_0 = w_66_0)
        let w_68_0 : Int := if w_67_0 then 1 else 0
        let w_69_0 : Int := (w_46_0 + w_68_0)
        let w_70_0 : Bool := true
        let w_71_0 : Int := if w_70_0 then 1 else 0
        let w_72_0 : Int := 1
        let w_73_0 : Bool := decide (w_21_0 = w_72_0)
        let w_74_0 : Int := if w_73_0 then 1 else 0
        let w_75_0 : Int := (w_71_0 + w_74_0)
        let w_76_0 : Int := 2
        let w_77_0 : Bool := decide (w_75_0 = w_76_0)
        let w_78_0 : Int := if w_77_0 then 1 else 0
        let w_79_0 : Int := if w_57_0 then 1 else 0
        let w_80_0 : Int := (w_78_0 + w_79_0)
        let w_81_0 : Int := 2
        let w_82_0 : Bool := decide (w_80_0 = w_81_0)
        let w_83_0 : Int := if w_82_0 then 1 else 0
        let w_84_0 : Int := if w_63_0 then 1 else 0
        let w_85_0 : Int := (w_83_0 + w_84_0)
        let w_86_0 : Int := 2
        let w_87_0 : Bool := decide (w_85_0 = w_86_0)
        let w_88_0 : Int := if w_87_0 then 1 else 0
        let w_89_0 : Int := (w_69_0 + w_88_0)
        let w_90_0 : Bool := true
        let w_91_0 : Int := if w_90_0 then 1 else 0
        let w_92_0 : Int := 2
        let w_93_0 : Bool := decide (w_21_0 = w_92_0)
        let w_94_0 : Int := if w_93_0 then 1 else 0
        let w_95_0 : Int := (w_91_0 + w_94_0)
        let w_96_0 : Int := 2
        let w_97_0 : Bool := decide (w_95_0 = w_96_0)
        let w_98_0 : Int := if w_97_0 then 1 else 0
        let w_99_0 : Bool := decide (w_56_0 ≤ w_30_0)
        let w_100_0 : Int := if w_99_0 then 1 else 0
        let w_102_0 : Int := 1
        let w_103_0 : Int := (w_101_0 - w_102_0)
        let w_104_0 : Bool := decide (w_30_0 ≤ w_103_0)
        let w_105_0 : Int := if w_104_0 then 1 else 0
        let w_106_0 : Int := (w_100_0 + w_105_0)
        let w_107_0 : Int := 2
        let w_108_0 : Bool := decide (w_106_0 = w_107_0)
        let w_109_0 : Int := if w_108_0 then 1 else 0
        let w_110_0 : Int := (w_98_0 + w_109_0)
        let w_111_0 : Int := 2
        let w_112_0 : Bool := decide (w_110_0 = w_111_0)
        let w_113_0 : Int := if w_112_0 then 1 else 0
        let w_114_0 : Int := if w_63_0 then 1 else 0
        let w_115_0 : Int := (w_113_0 + w_114_0)
        let w_116_0 : Int := 2
        let w_117_0 : Bool := decide (w_115_0 = w_116_0)
        let w_118_0 : Int := if w_117_0 then 1 else 0
        let w_119_0 : Int := (w_89_0 + w_118_0)
        let w_120_0 : Bool := true
        let w_121_0 : Int := if w_120_0 then 1 else 0
        let w_122_0 : Int := 3
        let w_123_0 : Bool := decide (w_21_0 = w_122_0)
        let w_124_0 : Int := if w_123_0 then 1 else 0
        let w_125_0 : Int := (w_121_0 + w_124_0)
        let w_126_0 : Int := 2
        let w_127_0 : Bool := decide (w_125_0 = w_126_0)
        let w_128_0 : Int := if w_127_0 then 1 else 0
        let w_129_0 : Int := if w_108_0 then 1 else 0
        let w_130_0 : Int := (w_128_0 + w_129_0)
        let w_131_0 : Int := 2
        let w_132_0 : Bool := decide (w_130_0 = w_131_0)
        let w_133_0 : Int := if w_132_0 then 1 else 0
        let w_134_0 : Int := if w_63_0 then 1 else 0
        let w_135_0 : Int := (w_133_0 + w_134_0)
        let w_136_0 : Int := 2
        let w_137_0 : Bool := decide (w_135_0 = w_136_0)
        let w_138_0 : Int := if w_137_0 then 1 else 0
        let w_139_0 : Int := (w_119_0 + w_138_0)
        let w_140_0 : Bool := true
        let w_141_0 : Int := if w_140_0 then 1 else 0
        let w_142_0 : Int := 4
        let w_143_0 : Bool := decide (w_21_0 = w_142_0)
        let w_144_0 : Int := if w_143_0 then 1 else 0
        let w_145_0 : Int := (w_141_0 + w_144_0)
        let w_146_0 : Int := 2
        let w_147_0 : Bool := decide (w_145_0 = w_146_0)
        let w_148_0 : Int := if w_147_0 then 1 else 0
        let w_149_0 : Int := if w_108_0 then 1 else 0
        let w_150_0 : Int := (w_148_0 + w_149_0)
        let w_151_0 : Int := 2
        let w_152_0 : Bool := decide (w_150_0 = w_151_0)
        let w_153_0 : Int := if w_152_0 then 1 else 0
        let w_154_0 : Bool := decide (w_56_0 ≤ w_39_0)
        let w_155_0 : Int := if w_154_0 then 1 else 0
        let w_156_0 : Int := 1
        let w_157_0 : Int := (w_101_0 - w_156_0)
        let w_158_0 : Bool := decide (w_39_0 ≤ w_157_0)
        let w_159_0 : Int := if w_158_0 then 1 else 0
        let w_160_0 : Int := (w_155_0 + w_159_0)
        let w_161_0 : Int := 2
        let w_162_0 : Bool := decide (w_160_0 = w_161_0)
        let w_163_0 : Int := if w_162_0 then 1 else 0
        let w_164_0 : Int := (w_153_0 + w_163_0)
        let w_165_0 : Int := 2
        let w_166_0 : Bool := decide (w_164_0 = w_165_0)
        let w_167_0 : Int := if w_166_0 then 1 else 0
        let w_168_0 : Int := (w_139_0 + w_167_0)
        let w_169_0 : Bool := true
        let w_170_0 : Int := if w_169_0 then 1 else 0
        let w_171_0 : Int := 5
        let w_172_0 : Bool := decide (w_21_0 = w_171_0)
        let w_173_0 : Int := if w_172_0 then 1 else 0
        let w_174_0 : Int := (w_170_0 + w_173_0)
        let w_175_0 : Int := 2
        let w_176_0 : Bool := decide (w_174_0 = w_175_0)
        let w_177_0 : Int := if w_176_0 then 1 else 0
        let w_178_0 : Int := if w_108_0 then 1 else 0
        let w_179_0 : Int := (w_177_0 + w_178_0)
        let w_180_0 : Int := 2
        let w_181_0 : Bool := decide (w_179_0 = w_180_0)
        let w_182_0 : Int := if w_181_0 then 1 else 0
        let w_183_0 : Int := if w_162_0 then 1 else 0
        let w_184_0 : Int := (w_182_0 + w_183_0)
        let w_185_0 : Int := 2
        let w_186_0 : Bool := decide (w_184_0 = w_185_0)
        let w_187_0 : Int := if w_186_0 then 1 else 0
        let w_188_0 : Int := (w_168_0 + w_187_0)
        let w_189_0 : Int := 1
        let w_190_0 : Bool := decide (w_188_0 = w_189_0)
        ∃ (w_191_0 : Bool),
          let w_192_0 : Int := if w_191_0 then 1 else 0
          let w_193_0 : Int := (w_7_0 + w_192_0)
          let w_194_0 : Int := 2
          let w_195_0 : Bool := decide (w_193_0 = w_194_0)
          let w_196_0 : Int := if w_195_0 then 1 else 0
          let w_198_0 : Int := if w_197_0 then 1 else 0
          let w_199_0 : Int := (w_196_0 + w_198_0)
          let w_200_0 : Int := 2
          let w_201_0 : Bool := decide (w_199_0 = w_200_0)
          parallel_sequential_generatedRoot_70_17.constraints_0 w_16_0 w_29_0 w_38_0 w_13_0 w_20_0 w_21_0 w_30_0 w_39_0 w_45_0 w_190_0 w_191_0 w_201_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_70.Witness where
  w_3_0 : Int
  w_17_0 : Fin 15057592995680 → Bool

abbrev sequential_generatedRoot_70.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 15057592995680 → Bool) (w_7_0 : Int) (w_1_0 : Fin 361 → Int) (w_4_0 : Fin 5435791071440480 → Int) (w_5_0 : Fin 5435791071440480 → Int) (w_6_0 : Fin 5435791071440480 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_16_0 : Bool) (w_17_0 : Fin 15057592995680 → Bool) (outputs : (Fin 15057592995680 → Bool) × Int × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 361 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  (params.«max_layer_width») = 15057592995680 ∧
  (∀ i : Fin 15057592995680, parallel_sequential_generatedRoot_70_17 tape (path ++ [17, i.val]) params i_0 i ((w_0_0 i), (w_3_0, (w_4_0, (w_2_0, (w_5_0, (w_6_0, (w_7_0, (w_16_0, ())))))))) (w_17_0 i)) ∧
  outputs = (w_17_0, (w_3_0, ()))

abbrev sequential_generatedRoot_70.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 15057592995680 → Bool) × Int × (Fin 361 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × Int × Unit) (outputs : (Fin 15057592995680 → Bool) × Int × Unit) (witness : sequential_generatedRoot_70.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 15057592995680 → Bool := inputs.1
  let w_7_0 : Int := inputs.2.1
  let w_1_0 : Fin 361 → Int := inputs.2.2.1
  let w_4_0 : Fin 5435791071440480 → Int := inputs.2.2.2.1
  let w_5_0 : Fin 5435791071440480 → Int := inputs.2.2.2.2.1
  let w_6_0 : Fin 5435791071440480 → Int := inputs.2.2.2.2.2.1
  let w_11_0 : Int := inputs.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_8_0 : Int := 1
  let w_9_0 : Bool := decide (w_8_0 ≤ w_3_0)
  let w_10_0 : Int := if w_9_0 then 1 else 0
  let w_12_0 : Bool := decide (w_3_0 ≤ w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_14_0 : Int := (w_10_0 + w_13_0)
  let w_15_0 : Int := 2
  let w_16_0 : Bool := decide (w_14_0 = w_15_0)
  let w_17_0 := witness.w_17_0
  sequential_generatedRoot_70.constraints_0 tape path params i_0 w_0_0 w_7_0 w_1_0 w_4_0 w_5_0 w_6_0 w_2_0 w_3_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_70 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 15057592995680 → Bool) × Int × (Fin 361 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × Int × Unit) (outputs : (Fin 15057592995680 → Bool) × Int × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_70.Witness,
    sequential_generatedRoot_70.body tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_71.constraints_0 (w_2_0 : Fin 15057592995680 → Bool) (w_3_0 : Int) (w_4_0 : Bool) (w_8_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 15057592995680 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  outputs = w_8_0

def sequential_generatedRoot_71 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 15057592995680 → Bool) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 15057592995680 → Bool := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Bool),
    let w_5_0 : Int := if w_4_0 then 1 else 0
    let w_6_0 : Int := (w_1_0 + w_5_0)
    let w_7_0 : Int := 2
    let w_8_0 : Bool := decide (w_6_0 = w_7_0)
    sequential_generatedRoot_71.constraints_0 w_2_0 w_3_0 w_4_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_65_0 : Fin 15057592995680 → Bool
  w_70_0 : Fin 15057592995680 → Bool
  w_70_1 : Int
  w_71_0 : Bool
  w_79_0 : Int

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_66_0 : Fin 361 → Int) (w_67_0 : Fin 5435791071440480 → Int) (w_68_0 : Fin 5435791071440480 → Int) (w_69_0 : Fin 5435791071440480 → Int) (w_78_0 : Fin 1 → Int) (w_27_0 : Int) (w_47_0 : Int) (w_64_0 : Bool) (w_65_0 : Fin 15057592995680 → Bool) (w_70_0 : Fin 15057592995680 → Bool) (w_70_1 : Int) (w_71_0 : Bool) (w_79_0 : Int) (w_92_0 : Bool) (outputs : Bool) : Prop :=
  (params.«max_layer_width») = 15057592995680 ∧
  (∀ i : Fin 15057592995680, parallel_generatedRoot_65 tape (path ++ [65, i.val]) params i () (w_65_0 i)) ∧
  0 ≤ (params.«depth») ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : (Fin 15057592995680 → Bool) × Int × Unit) => sequential_generatedRoot_70 tape (path ++ [70, i]) params i (current.1, (current.2.1, (w_66_0, (w_67_0, (w_68_0, (w_69_0, (w_47_0, ()))))))) next) (Int.toNat (params.«depth»)) (w_65_0, (w_27_0, ())) (w_70_0, (w_70_1, ())) ∧
  0 ≤ (params.«max_layer_width») ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_71 tape (path ++ [71, i]) params i (current, (w_70_0, ())) next) (Int.toNat (params.«max_layer_width»)) w_64_0 w_71_0 ∧
  MxxRuntime.familyGetStatic w_78_0 (0) w_79_0 ∧
  outputs = w_92_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 361 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_66_0 : Fin 361 → Int := inputs.1
  let w_67_0 : Fin 5435791071440480 → Int := inputs.2.1
  let w_68_0 : Fin 5435791071440480 → Int := inputs.2.2.1
  let w_69_0 : Fin 5435791071440480 → Int := inputs.2.2.2.1
  let w_78_0 : Fin 1 → Int := inputs.2.2.2.2.1
  let w_0_0 : Bool := true
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Bool := true
  let w_3_0 : Int := if w_2_0 then 1 else 0
  let w_4_0 : Int := 0
  let w_5_0 : Int := params.«instance_width»
  let w_6_0 : Int := 0
  let w_7_0 : Int := (w_5_0 + w_6_0)
  let w_8_0 : Bool := decide (w_4_0 ≤ w_7_0)
  let w_9_0 : Int := if w_8_0 then 1 else 0
  let w_10_0 : Int := (w_3_0 + w_9_0)
  let w_11_0 : Int := 2
  let w_12_0 : Bool := decide (w_10_0 = w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_14_0 : Int := 0
  let w_15_0 : Int := params.«witness_width»
  let w_16_0 : Int := 0
  let w_17_0 : Int := (w_15_0 + w_16_0)
  let w_18_0 : Bool := decide (w_14_0 ≤ w_17_0)
  let w_19_0 : Int := if w_18_0 then 1 else 0
  let w_20_0 : Int := (w_13_0 + w_19_0)
  let w_21_0 : Int := 2
  let w_22_0 : Bool := decide (w_20_0 = w_21_0)
  let w_23_0 : Int := if w_22_0 then 1 else 0
  let w_24_0 : Int := 1
  let w_25_0 : Int := (params.«instance_width» + params.«witness_width»)
  let w_26_0 : Int := 0
  let w_27_0 : Int := (w_25_0 + w_26_0)
  let w_28_0 : Bool := decide (w_24_0 ≤ w_27_0)
  let w_29_0 : Int := if w_28_0 then 1 else 0
  let w_30_0 : Int := (w_23_0 + w_29_0)
  let w_31_0 : Int := 2
  let w_32_0 : Bool := decide (w_30_0 = w_31_0)
  let w_33_0 : Int := if w_32_0 then 1 else 0
  let w_34_0 : Int := 1
  let w_35_0 : Int := params.«depth»
  let w_36_0 : Int := 0
  let w_37_0 : Int := (w_35_0 + w_36_0)
  let w_38_0 : Bool := decide (w_34_0 ≤ w_37_0)
  let w_39_0 : Int := if w_38_0 then 1 else 0
  let w_40_0 : Int := (w_33_0 + w_39_0)
  let w_41_0 : Int := 2
  let w_42_0 : Bool := decide (w_40_0 = w_41_0)
  let w_43_0 : Int := if w_42_0 then 1 else 0
  let w_44_0 : Int := 1
  let w_45_0 : Int := params.«max_layer_width»
  let w_46_0 : Int := 0
  let w_47_0 : Int := (w_45_0 + w_46_0)
  let w_48_0 : Bool := decide (w_44_0 ≤ w_47_0)
  let w_49_0 : Int := if w_48_0 then 1 else 0
  let w_50_0 : Int := (w_43_0 + w_49_0)
  let w_51_0 : Int := 2
  let w_52_0 : Bool := decide (w_50_0 = w_51_0)
  let w_53_0 : Int := if w_52_0 then 1 else 0
  let w_54_0 : Bool := decide (w_27_0 ≤ w_47_0)
  let w_55_0 : Int := if w_54_0 then 1 else 0
  let w_56_0 : Int := (w_53_0 + w_55_0)
  let w_57_0 : Int := 2
  let w_58_0 : Bool := decide (w_56_0 = w_57_0)
  let w_59_0 : Int := if w_58_0 then 1 else 0
  let w_60_0 : Int := (w_1_0 + w_59_0)
  let w_61_0 : Int := 2
  let w_62_0 : Bool := decide (w_60_0 = w_61_0)
  let w_63_0 : Int := if w_62_0 then 1 else 0
  let w_64_0 : Bool := true
  let w_65_0 := witness.w_65_0
  let w_70_0 := witness.w_70_0
  let w_70_1 := witness.w_70_1
  let w_71_0 := witness.w_71_0
  let w_72_0 : Int := if w_71_0 then 1 else 0
  let w_73_0 : Int := (w_63_0 + w_72_0)
  let w_74_0 : Int := 2
  let w_75_0 : Bool := decide (w_73_0 = w_74_0)
  let w_76_0 : Int := if w_75_0 then 1 else 0
  let w_77_0 : Int := 0
  let w_79_0 := witness.w_79_0
  let w_80_0 : Bool := decide (w_77_0 ≤ w_79_0)
  let w_81_0 : Int := if w_80_0 then 1 else 0
  let w_82_0 : Int := 1
  let w_83_0 : Int := (w_70_1 - w_82_0)
  let w_84_0 : Bool := decide (w_79_0 ≤ w_83_0)
  let w_85_0 : Int := if w_84_0 then 1 else 0
  let w_86_0 : Int := (w_81_0 + w_85_0)
  let w_87_0 : Int := 2
  let w_88_0 : Bool := decide (w_86_0 = w_87_0)
  let w_89_0 : Int := if w_88_0 then 1 else 0
  let w_90_0 : Int := (w_76_0 + w_89_0)
  let w_91_0 : Int := 2
  let w_92_0 : Bool := decide (w_90_0 = w_91_0)
  generatedRoot.constraints_0 tape path params w_66_0 w_67_0 w_68_0 w_69_0 w_78_0 w_27_0 w_47_0 w_64_0 w_65_0 w_70_0 w_70_1 w_71_0 w_79_0 w_92_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 361 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 5435791071440480 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
