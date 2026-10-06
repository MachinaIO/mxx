import LwrLutE
import LwrPreprocessE
import LwrEncryptE

/-!
# The model of a run

Every input of the error analysis as a function of the sample tape: the payload and mask secrets
and the Gaussian errors, each clamped to its sampler's range (which every run reads unchanged),
and the hash-derived keys.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape} {path : List Nat}

/-- `rns-bgg/lut-b/LutId(0)/34/35` -/
def lutBPrefix : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48,
    41, 47, 51, 52, 47, 51, 53]

/-- The hash-derived switching tops `B_{0j}` of branch `i` of the readout LUT. -/
noncomputable def lutB (model : HashModel) (key : ByteArray) (i : Fin 128) (j : Fin 2) :
    ExactMatrix Q1 N 1 4 :=
  model.sample Q1 N 1 4 key (completeHashTag lutBPrefix
    [.bytes [115, 105, 103, 109, 97], .u64Le (sigma i), .bytes [98, 114, 97, 110, 99, 104],
      .u64Le (i : Int), .bytes componentBytes, .u64Le (j : Int)])

set_option maxHeartbeats 8000000 in
/-- The switching tops of the readout LUT are hash-derived. -/
theorem preprocess_switchTop {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q36 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 b5 x5 b6 x6 b7 x7 b8 x8 b9 x9 b10 x10 b11 x11 b12 x12 b13 x13 b14 x14 b15 x15 b16 x16 b17 x17 b18 x18 b19 x19 b20 x20 b21 x21 b22 x22 b23 x23 b24 x24 b25 x25 b26 x26 b27 x27 b28 x28 b29 x29 b30 x30 b31 x31 b32 x32 b33 x33 : Fin 113 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, b5, x5, b6, x6, b7, x7, b8, x8, b9, x9, b10, x10, b11, x11, b12, x12, b13, x13, b14, x14, b15, x15, b16, x16, b17, x17, b18, x18, b19, x19, b20, x20, b21, x21, b22, x22, b23, x23, b24, x24, b25, x25, b26, x26, b27, x27, b28, x28, b29, x29, b30, x30, b31, x31, b32, x32, b33, x33, ()) outputs) (i : Fin 128) :
    ![outputs.1 i, outputs.2.1 i] = lutB model key i := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h3_3 := hb.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout
  have hcell := h3_3 i
  unfold Stage_preprocess.parallel_generatedRoot_3 at hcell
  try extract_lets at hcell
  obtain ⟨w_1_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_2_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_4_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_5_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_12_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_17_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_23_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_28_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_35_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_40_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_46_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_51_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_59_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_64_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_70_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_75_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_82_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_87_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_93_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_98_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_107_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_112_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_118_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_123_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_130_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_135_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_141_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_146_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_154_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_159_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_165_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_170_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_177_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_182_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_188_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_193_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_203_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_208_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_214_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_219_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_226_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_231_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_237_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_242_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_250_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_255_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_261_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_266_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_273_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_278_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_284_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_289_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_298_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_303_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_309_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_314_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_321_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_326_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_332_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_337_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_345_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_350_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_356_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_361_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_368_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_373_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_379_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_384_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_395_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_400_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_406_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_411_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_418_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_423_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_429_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_434_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_442_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_447_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_453_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_458_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_465_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_470_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_476_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_481_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_490_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_495_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_501_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_506_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_513_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_518_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_524_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_529_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_537_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_542_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_548_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_553_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_560_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_565_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_571_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_576_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_586_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_591_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_597_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_602_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_609_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_614_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_620_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_625_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_633_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_638_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_644_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_649_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_656_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_661_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_667_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_672_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_681_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_686_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_692_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_697_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_704_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_709_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_715_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_720_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_728_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_733_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_739_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_744_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_751_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_756_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_762_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_767_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_779_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_782_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_788_0, hcell⟩ := hcell
  try extract_lets at hcell
  obtain ⟨w_792_0, hcell⟩ := hcell
  try extract_lets at hcell
  have h1_0 := hcell.2.2.1
  have h2_0 := hcell.2.2.2.2.2.1
  have hout := hcell.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  funext j
  fin_cases j
  · simp only []
    rw [ho0, hashSample_eq h1_0]
    rfl
  · simp only []
    rw [ho1, hashSample_eq h2_0]
    rfl

/-! ## Inputs -/

/-- The payload secret of the model: its samples clamped to `[-1, 1]`. -/
noncomputable def payloadM (tape : SampleTape) : ErrorPoly N :=
  clampedWitness (rows := 1) (columns := 1) tape [0, 0] (.interval (-1) 1) (-1) 1 0 0

/-- The mask secret of the model: its samples clamped to `[-1, 1]`. -/
noncomputable def maskM (tape : SampleTape) : ErrorPoly N :=
  clampedWitness (rows := 1) (columns := 1) tape [1, 0] (.interval (-1) 1) (-1) 1 0 0

theorem secret_model {site : List Nat} {out : ExactMatrix Q36 N 1 1}
    (h : uniformIntervalSampleAt tape site (-1) 1 out) :
    out = (fun _ _ ↦ reducePoly Q36 N
      (clampedWitness (rows := 1) (columns := 1) tape site (.interval (-1) 1) (-1) 1 0 0)) := by
  obtain ⟨hw, -⟩ := ternarySampleAt_explicit hN h
  rw [hw, clampedWitness_eq_of_interval hN (rows := 1) (columns := 1) h]

theorem payload_model {out : ExactMatrix Q36 N 1 1}
    (h : Stage_payload.generatedRoot tape [0] GeneratedClaim.stage_0_params () out) :
    out = fun _ _ ↦ reducePoly Q36 N (payloadM tape) := by
  obtain ⟨w, hw, hout⟩ := h
  rw [hout]
  exact secret_model hw

theorem mask_model {out : ExactMatrix Q36 N 1 1}
    (h : Stage_mask.generatedRoot tape [1] GeneratedClaim.stage_1_params () out) :
    out = fun _ _ ↦ reducePoly Q36 N (maskM tape) := by
  obtain ⟨w, hw, hout⟩ := h
  rw [hout]
  exact secret_model hw

/-- An identity `X - Y + s Z + U = p v` entrywise is `v = divInt p (X - Y + s • Z + U)`. -/
theorem div_of_identity {rows columns : Nat} {p : ℤ} (hp : p ≠ 0) {sm : ErrorPoly N}
    {X Y Z U v : ErrorMatrix N rows columns}
    (h : ∀ row column, X row column - Y row column + sm * Z row column + U row column =
      ((p : ℤ) : ErrorPoly N) * v row column) :
    v = divInt p (X - Y + sm • Z + U) :=
  (divInt_eq hN hp fun row column ↦ by
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
    exact h row column).symm

section Model

variable (model : HashModel) (key : ByteArray) (ext : GeneratedClaim.ExternalInputs)

/-! ## The plaintexts and the constants -/

/-- The plaintext `X^(512 a_k)` of member `k` of `a`. -/
noncomputable def alphaOf (k : Fin 1025) : ErrorPoly N := Xe (512 * (ext.input_3 k).toNat)

/-- The selector bits of every block. -/
def selOf : Fin 34 → Fin 113 → Int :=
  ![ext.input_6, ext.input_8, ext.input_10, ext.input_12,
    ext.input_14, ext.input_16, ext.input_18, ext.input_20, ext.input_22, ext.input_24,
    ext.input_26, ext.input_28, ext.input_30, ext.input_32, ext.input_34, ext.input_36,
    ext.input_38, ext.input_40, ext.input_42, ext.input_44, ext.input_46, ext.input_48,
    ext.input_50, ext.input_52, ext.input_54, ext.input_56, ext.input_58, ext.input_60,
    ext.input_62, ext.input_64, ext.input_66, ext.input_68, ext.input_70, ext.input_72]

/-- The selector plaintext of cell `i` of block `b`. -/
noncomputable def nuOf (b : Fin 34) (i : Fin 113) : ErrorPoly N := (selOf ext b i : ErrorPoly N)

/-- The sum of the coordinates selected by the blocks before `b`. -/
def zOf (b : Nat) : Nat :=
  ((Finset.univ.filter fun b' : Fin 34 ↦ b'.val < b).sum fun b' ↦
    ∑ i : Fin 113, selOf ext b' i * ext.input_3 ⟨coords b' i, coords_lt b' i⟩).toNat

/-- The state plaintext `X^(512 z_b)` entering block `b`. -/
noncomputable def muOf (b : Nat) : ErrorPoly N := Xe (512 * zOf ext b)

/-- The native gadget of `Q_q'` rebased to `Q_q`. -/
noncomputable abbrev gReb (q q' m : Nat) : ExactMatrix q N 1 m :=
  theOutput centeredRebaseRuns (gadgetOf q' m)

end Model

end LwrProof
