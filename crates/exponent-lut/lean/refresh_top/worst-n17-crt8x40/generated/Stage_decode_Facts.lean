import Stage_decode

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Stage_decode

local notation "mxxNat0" => 1067915858319243662302466808359780720911201935418912308884990971092217754157489135095283728646144
local notation "mxxNat1" => 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 28
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 56
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 28
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 28

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_28.Ctx (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) where
  inputs : Int × mxxMat0 × Int × Int × Unit
  witness : sequential_generatedRoot_28.Witness

def sequential_generatedRoot_28.w_0_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Int :=
  ctx.inputs.1

def sequential_generatedRoot_28.w_1_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def sequential_generatedRoot_28.w_4_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Int :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_28.w_5_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Int :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_28.w_2_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def sequential_generatedRoot_28.w_3_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Fin 131072 → Int :=
  ctx.witness.w_3_0

def sequential_generatedRoot_28.w_6_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Int :=
  ctx.witness.w_6_0

def sequential_generatedRoot_28.outs {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) : Int :=
  (sequential_generatedRoot_28.w_6_0 ctx)

structure sequential_generatedRoot_28.Facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) (outputs : Int) : Prop where
  t0 : 0 ≤ 0
  t1 : 0 < 1
  t2 : 1 ≤ 1
  t3 : 0 ≤ (Int.ofNat i_0)
  t4 : (Int.ofNat i_0) < (1 + (Int.ofNat i_0))
  t5 : (1 + (Int.ofNat i_0)) ≤ 28
  h2_0 : MxxRuntime.sliceMatrix (sequential_generatedRoot_28.w_1_0 ctx) 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) (sequential_generatedRoot_28.w_2_0 ctx)
  h3_0 : MxxRuntime.polynomialValues false (sequential_generatedRoot_28.w_2_0 ctx) (sequential_generatedRoot_28.w_3_0 ctx)
  t8 : 0 ≤ (131072)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_28_6 backend params i_0 i (current, ((sequential_generatedRoot_28.w_3_0 ctx), ((sequential_generatedRoot_28.w_4_0 ctx), ((sequential_generatedRoot_28.w_5_0 ctx), ())))) next) (Int.toNat (131072)) (sequential_generatedRoot_28.w_0_0 ctx) (sequential_generatedRoot_28.w_6_0 ctx)
  hout : outputs = sequential_generatedRoot_28.outs ctx

theorem sequential_generatedRoot_28.facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_28.Ctx backend params i_0} {outputs : Int}
    (h : sequential_generatedRoot_28.constraints_0 backend params i_0 (sequential_generatedRoot_28.w_0_0 ctx) (sequential_generatedRoot_28.w_1_0 ctx) (sequential_generatedRoot_28.w_4_0 ctx) (sequential_generatedRoot_28.w_5_0 ctx) (sequential_generatedRoot_28.w_2_0 ctx) (sequential_generatedRoot_28.w_3_0 ctx) (sequential_generatedRoot_28.w_6_0 ctx) outputs) :
    sequential_generatedRoot_28.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_28.Facts {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) (outputs : Int) : Prop where
  c0 : sequential_generatedRoot_28.Facts_0 ctx outputs

theorem sequential_generatedRoot_28.facts {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx backend params i_0) {outputs : Int}
    (h : sequential_generatedRoot_28.body backend params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_28.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_28.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_46.Ctx (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) where
  inputs : Int × mxxMat0 × Int × Int × Unit
  witness : sequential_generatedRoot_46.Witness

def sequential_generatedRoot_46.w_0_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Int :=
  ctx.inputs.1

def sequential_generatedRoot_46.w_1_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : mxxMat0 :=
  ctx.inputs.2.1

def sequential_generatedRoot_46.w_4_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Int :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_46.w_5_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Int :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_46.w_2_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : mxxMat1 :=
  ctx.witness.w_2_0

def sequential_generatedRoot_46.w_3_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Fin 131072 → Int :=
  ctx.witness.w_3_0

def sequential_generatedRoot_46.w_6_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Int :=
  ctx.witness.w_6_0

def sequential_generatedRoot_46.outs {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) : Int :=
  (sequential_generatedRoot_46.w_6_0 ctx)

structure sequential_generatedRoot_46.Facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) (outputs : Int) : Prop where
  t0 : 0 ≤ 0
  t1 : 0 < 1
  t2 : 1 ≤ 1
  t3 : 0 ≤ (Int.ofNat i_0)
  t4 : (Int.ofNat i_0) < (1 + (Int.ofNat i_0))
  t5 : (1 + (Int.ofNat i_0)) ≤ 28
  h2_0 : MxxRuntime.sliceMatrix (sequential_generatedRoot_46.w_1_0 ctx) 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) (sequential_generatedRoot_46.w_2_0 ctx)
  h3_0 : MxxRuntime.polynomialValues false (sequential_generatedRoot_46.w_2_0 ctx) (sequential_generatedRoot_46.w_3_0 ctx)
  t8 : 0 ≤ (131072)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_46_6 backend params i_0 i (current, ((sequential_generatedRoot_46.w_3_0 ctx), ((sequential_generatedRoot_46.w_4_0 ctx), ((sequential_generatedRoot_46.w_5_0 ctx), ())))) next) (Int.toNat (131072)) (sequential_generatedRoot_46.w_0_0 ctx) (sequential_generatedRoot_46.w_6_0 ctx)
  hout : outputs = sequential_generatedRoot_46.outs ctx

theorem sequential_generatedRoot_46.facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_46.Ctx backend params i_0} {outputs : Int}
    (h : sequential_generatedRoot_46.constraints_0 backend params i_0 (sequential_generatedRoot_46.w_0_0 ctx) (sequential_generatedRoot_46.w_1_0 ctx) (sequential_generatedRoot_46.w_4_0 ctx) (sequential_generatedRoot_46.w_5_0 ctx) (sequential_generatedRoot_46.w_2_0 ctx) (sequential_generatedRoot_46.w_3_0 ctx) (sequential_generatedRoot_46.w_6_0 ctx) outputs) :
    sequential_generatedRoot_46.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_46.Facts {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) (outputs : Int) : Prop where
  c0 : sequential_generatedRoot_46.Facts_0 ctx outputs

theorem sequential_generatedRoot_46.facts {backend : MxxRuntime.BackendContext} {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx backend params i_0) {outputs : Int}
    (h : sequential_generatedRoot_46.body backend params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_46.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_46.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (params : Params) where
  inputs : (Fin 16 → mxxMat0) × (Fin 131072 → Int) × mxxMat2 × mxxMat1 × Int × Int × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Fin 16 → mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Fin 131072 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat2 :=
  ctx.inputs.2.2.1

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  0

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  0

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_3_0

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_4_0

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  ctx.witness.w_6_0

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat3 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_9_0

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_6_0 ctx) (generatedRoot.w_9_0 ctx)

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_11_0

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx)

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1099489607681)] : mxxMat1)

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.liftInteger (generatedRoot.w_16_0 ctx)

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.multiplyMonomial (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_15_0 ctx) (generatedRoot.w_19_0 ctx)

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat4 :=
  ctx.witness.w_21_0

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_22_0

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_12_0 ctx) (generatedRoot.w_23_0 ctx)

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_4_0 ctx) (generatedRoot.w_24_0 ctx)

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  mxxNat0

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  mxxNat1

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ((generatedRoot.w_0_0 ctx) + (generatedRoot.w_28_0 ctx))

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  0

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_31_0

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_32_0

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat3 :=
  ctx.witness.w_33_0

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_34_0

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_6_0 ctx) (generatedRoot.w_34_0 ctx)

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  ctx.witness.w_36_0

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx)

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  (1 : mxxMat1)

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_13_0 ctx) (generatedRoot.w_38_0 ctx)

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_39_0 ctx) (generatedRoot.w_19_0 ctx)

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_40_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_37_0 ctx) (generatedRoot.w_41_0 ctx)

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_32_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  mxxNat0

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  mxxNat1

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ctx.witness.w_46_0

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  ((generatedRoot.w_29_0 ctx) + (generatedRoot.w_46_0 ctx))

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Int :=
  (generatedRoot.w_47_0 ctx)

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) : Prop where
  h3_0 : MxxRuntime.familyGetStatic (generatedRoot.w_2_0 ctx) (0) (generatedRoot.w_3_0 ctx)
  t1 : 0 ≤ (7)
  h4_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_4 backend params i (current, ((generatedRoot.w_2_0 ctx), ())) next) (Int.toNat (7)) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx)
  h6_0 : MxxRuntime.polynomialFromValues false (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  t4 : 0 ≤ 0
  t5 : 0 < 2
  t6 : 2 ≤ 2
  t7 : 0 ≤ 0
  t8 : 0 < 28
  t9 : 28 ≤ 56
  h8_0 : MxxRuntime.sliceMatrix (generatedRoot.w_7_0 ctx) 0 2 0 28 (generatedRoot.w_8_0 ctx)
  t11 : 0 ≤ 0
  t12 : 0 < 1
  t13 : 1 ≤ 2
  t14 : 0 ≤ 0
  t15 : 0 < 28
  t16 : 28 ≤ 28
  h9_0 : MxxRuntime.sliceMatrix (generatedRoot.w_8_0 ctx) 0 1 0 28 (generatedRoot.w_9_0 ctx)
  t18 : 0 ≤ 1
  t19 : 1 < 2
  t20 : 2 ≤ 2
  t21 : 0 ≤ 0
  t22 : 0 < 28
  t23 : 28 ≤ 28
  h11_0 : MxxRuntime.sliceMatrix (generatedRoot.w_8_0 ctx) 1 2 0 28 (generatedRoot.w_11_0 ctx)
  h21_0 : MxxRuntime.gadgetMatrixRuns backend 1024 28 (generatedRoot.w_21_0 ctx)
  h22_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx)
  t27 : 0 ≤ (28)
  h28_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_28 backend params i (current, ((generatedRoot.w_25_0 ctx), ((generatedRoot.w_26_0 ctx), ((generatedRoot.w_27_0 ctx), ())))) next) (Int.toNat (28)) (generatedRoot.w_1_0 ctx) (generatedRoot.w_28_0 ctx)
  h31_0 : MxxRuntime.familyGetStatic (generatedRoot.w_2_0 ctx) (1) (generatedRoot.w_31_0 ctx)
  t30 : 0 ≤ (7)
  h32_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_32 backend params i (current, ((generatedRoot.w_2_0 ctx), ())) next) (Int.toNat (7)) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx)

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {params : Params} {ctx : generatedRoot.Ctx backend params} {outputs : Int}
    (h : generatedRoot.constraints_0 backend params (generatedRoot.w_2_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) (outputs : Int) : Prop where
  t32 : 0 ≤ 0
  t33 : 0 < 2
  t34 : 2 ≤ 2
  t35 : 0 ≤ 28
  t36 : 28 < 56
  t37 : 56 ≤ 56
  h33_0 : MxxRuntime.sliceMatrix (generatedRoot.w_7_0 ctx) 0 2 28 56 (generatedRoot.w_33_0 ctx)
  t39 : 0 ≤ 0
  t40 : 0 < 1
  t41 : 1 ≤ 2
  t42 : 0 ≤ 0
  t43 : 0 < 28
  t44 : 28 ≤ 28
  h34_0 : MxxRuntime.sliceMatrix (generatedRoot.w_33_0 ctx) 0 1 0 28 (generatedRoot.w_34_0 ctx)
  t46 : 0 ≤ 1
  t47 : 1 < 2
  t48 : 2 ≤ 2
  t49 : 0 ≤ 0
  t50 : 0 < 28
  t51 : 28 ≤ 28
  h36_0 : MxxRuntime.sliceMatrix (generatedRoot.w_33_0 ctx) 1 2 0 28 (generatedRoot.w_36_0 ctx)
  t53 : 0 ≤ (28)
  h46_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_46 backend params i (current, ((generatedRoot.w_43_0 ctx), ((generatedRoot.w_44_0 ctx), ((generatedRoot.w_45_0 ctx), ())))) next) (Int.toNat (28)) (generatedRoot.w_30_0 ctx) (generatedRoot.w_46_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {params : Params} {ctx : generatedRoot.Ctx backend params} {outputs : Int}
    (h : generatedRoot.constraints_1 backend params (generatedRoot.w_7_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) (outputs : Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {params : Params} (ctx : generatedRoot.Ctx backend params) {outputs : Int}
    (h : generatedRoot.body backend params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end Stage_decode

end
