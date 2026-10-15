import W_Requirement_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace W_Requirement_0

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_41.Ctx (params : Params) (i_0 : Nat) where
  inputs : (Fin 8193 → Int) × (Fin 8389632 → Int) × (Fin 8193 → Int) × Unit
  witness : parallel_generatedRoot_41.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_41Ctx" => (parallel_generatedRoot_41.Ctx params i_0)

def parallel_generatedRoot_41.w_1_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 8193 → Int :=
  ctx.inputs.1

def parallel_generatedRoot_41.w_6_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 8389632 → Int :=
  ctx.inputs.2.1

def parallel_generatedRoot_41.w_9_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 8193 → Int :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_41.w_0_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  0

def parallel_generatedRoot_41.w_2_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  (Int.ofNat i_0)

def parallel_generatedRoot_41.w_3_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  8193

def parallel_generatedRoot_41.w_4_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_2_0 ctx) % (parallel_generatedRoot_41.w_3_0 ctx))

def parallel_generatedRoot_41.w_5_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_5_0

def parallel_generatedRoot_41.w_7_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  8193

def parallel_generatedRoot_41.w_8_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_2_0 ctx) / (parallel_generatedRoot_41.w_7_0 ctx))

def parallel_generatedRoot_41.w_10_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_41.w_11_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_11_0

def parallel_generatedRoot_41.w_12_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  256

def parallel_generatedRoot_41.w_13_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_11_0 ctx) % (parallel_generatedRoot_41.w_12_0 ctx))

def parallel_generatedRoot_41.outs  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  (parallel_generatedRoot_41.w_13_0 ctx)

structure parallel_generatedRoot_41.Facts_0  (ctx : parallel_generatedRoot_41Ctx) (outputs : Int) : Prop where
  h3_0 : (parallel_generatedRoot_41.w_3_0 ctx) ≠ 0
  h4_0 : 0 ≤ (parallel_generatedRoot_41.w_4_0 ctx)
  h4_0_1 : (parallel_generatedRoot_41.w_4_0 ctx) < 8193
  h5_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_41.w_1_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_5_0 ctx)
  h7_0 : (parallel_generatedRoot_41.w_7_0 ctx) ≠ 0
  h4_0_2 : 0 ≤ (parallel_generatedRoot_41.w_4_0 ctx)
  h4_0_3 : (parallel_generatedRoot_41.w_4_0 ctx) < 8193
  h10_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_41.w_9_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_10_0 ctx)
  t8 : 0 ≤ (4253)
  h11_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_parallel_generatedRoot_41_11 params i_0 i (current, ((parallel_generatedRoot_41.w_5_0 ctx), ((parallel_generatedRoot_41.w_6_0 ctx), ((parallel_generatedRoot_41.w_8_0 ctx), ((parallel_generatedRoot_41.w_10_0 ctx), ()))))) next) (Int.toNat (4253)) (parallel_generatedRoot_41.w_0_0 ctx) (parallel_generatedRoot_41.w_11_0 ctx)
  h12_0 : (parallel_generatedRoot_41.w_12_0 ctx) ≠ 0
  hout : outputs = parallel_generatedRoot_41.outs ctx

theorem parallel_generatedRoot_41.facts_0  {ctx : parallel_generatedRoot_41Ctx} {outputs : Int}
    (h : parallel_generatedRoot_41.constraints_0 params i_0 (parallel_generatedRoot_41.w_1_0 ctx) (parallel_generatedRoot_41.w_6_0 ctx) (parallel_generatedRoot_41.w_9_0 ctx) (parallel_generatedRoot_41.w_0_0 ctx) (parallel_generatedRoot_41.w_3_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_5_0 ctx) (parallel_generatedRoot_41.w_7_0 ctx) (parallel_generatedRoot_41.w_8_0 ctx) (parallel_generatedRoot_41.w_10_0 ctx) (parallel_generatedRoot_41.w_11_0 ctx) (parallel_generatedRoot_41.w_12_0 ctx) (parallel_generatedRoot_41.w_13_0 ctx) outputs) :
    parallel_generatedRoot_41.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_41.Facts  (ctx : parallel_generatedRoot_41Ctx) (outputs : Int) : Prop where
  c0 : parallel_generatedRoot_41.Facts_0 ctx outputs

theorem parallel_generatedRoot_41.facts  (ctx : parallel_generatedRoot_41Ctx) {outputs : Int}
    (h : parallel_generatedRoot_41.body params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_41.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_41.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_74.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_74.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_74Ctx" => (sequential_generatedRoot_74.Ctx params i_0)

def sequential_generatedRoot_74.w_0_0  (ctx : sequential_generatedRoot_74Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_74.w_3_0  (ctx : sequential_generatedRoot_74Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_74.w_1_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  if (sequential_generatedRoot_74.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_74.w_2_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  0

def sequential_generatedRoot_74.w_4_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_74.w_5_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_74.w_6_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  1

def sequential_generatedRoot_74.w_7_0  (ctx : sequential_generatedRoot_74Ctx) : Bool :=
  decide ((sequential_generatedRoot_74.w_5_0 ctx) = (sequential_generatedRoot_74.w_6_0 ctx))

def sequential_generatedRoot_74.w_8_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  if (sequential_generatedRoot_74.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_74.w_9_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  ((sequential_generatedRoot_74.w_1_0 ctx) + (sequential_generatedRoot_74.w_8_0 ctx))

def sequential_generatedRoot_74.w_10_0  (ctx : sequential_generatedRoot_74Ctx) : Int :=
  2

def sequential_generatedRoot_74.w_11_0  (ctx : sequential_generatedRoot_74Ctx) : Bool :=
  decide ((sequential_generatedRoot_74.w_9_0 ctx) = (sequential_generatedRoot_74.w_10_0 ctx))

def sequential_generatedRoot_74.outs  (ctx : sequential_generatedRoot_74Ctx) : Bool :=
  (sequential_generatedRoot_74.w_11_0 ctx)

structure sequential_generatedRoot_74.Facts_0  (ctx : sequential_generatedRoot_74Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 params i_0 i (current, ((sequential_generatedRoot_74.w_3_0 ctx), ((sequential_generatedRoot_74.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_74.outs ctx

theorem sequential_generatedRoot_74.facts_0  {ctx : sequential_generatedRoot_74Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_74.constraints_0 params i_0 (sequential_generatedRoot_74.w_3_0 ctx) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_4_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx) (sequential_generatedRoot_74.w_11_0 ctx) outputs) :
    sequential_generatedRoot_74.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_74.Facts  (ctx : sequential_generatedRoot_74Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_74.Facts_0 ctx outputs

theorem sequential_generatedRoot_74.facts  (ctx : sequential_generatedRoot_74Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_74.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_74.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_74.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_76.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_76.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_76Ctx" => (sequential_generatedRoot_76.Ctx params i_0)

def sequential_generatedRoot_76.w_0_0  (ctx : sequential_generatedRoot_76Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_76.w_3_0  (ctx : sequential_generatedRoot_76Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_76.w_1_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  if (sequential_generatedRoot_76.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_76.w_2_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  0

def sequential_generatedRoot_76.w_4_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_76.w_5_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_76.w_6_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  1

def sequential_generatedRoot_76.w_7_0  (ctx : sequential_generatedRoot_76Ctx) : Bool :=
  decide ((sequential_generatedRoot_76.w_5_0 ctx) = (sequential_generatedRoot_76.w_6_0 ctx))

def sequential_generatedRoot_76.w_8_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  if (sequential_generatedRoot_76.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_76.w_9_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  ((sequential_generatedRoot_76.w_1_0 ctx) + (sequential_generatedRoot_76.w_8_0 ctx))

def sequential_generatedRoot_76.w_10_0  (ctx : sequential_generatedRoot_76Ctx) : Int :=
  2

def sequential_generatedRoot_76.w_11_0  (ctx : sequential_generatedRoot_76Ctx) : Bool :=
  decide ((sequential_generatedRoot_76.w_9_0 ctx) = (sequential_generatedRoot_76.w_10_0 ctx))

def sequential_generatedRoot_76.outs  (ctx : sequential_generatedRoot_76Ctx) : Bool :=
  (sequential_generatedRoot_76.w_11_0 ctx)

structure sequential_generatedRoot_76.Facts_0  (ctx : sequential_generatedRoot_76Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 params i_0 i (current, ((sequential_generatedRoot_76.w_3_0 ctx), ((sequential_generatedRoot_76.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_76.outs ctx

theorem sequential_generatedRoot_76.facts_0  {ctx : sequential_generatedRoot_76Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_76.constraints_0 params i_0 (sequential_generatedRoot_76.w_3_0 ctx) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_4_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx) (sequential_generatedRoot_76.w_11_0 ctx) outputs) :
    sequential_generatedRoot_76.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_76.Facts  (ctx : sequential_generatedRoot_76Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_76.Facts_0 ctx outputs

theorem sequential_generatedRoot_76.facts  (ctx : sequential_generatedRoot_76Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_76.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_76.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_76.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_78.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_78.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_78Ctx" => (sequential_generatedRoot_78.Ctx params i_0)

def sequential_generatedRoot_78.w_0_0  (ctx : sequential_generatedRoot_78Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_78.w_3_0  (ctx : sequential_generatedRoot_78Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_78.w_1_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  if (sequential_generatedRoot_78.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_78.w_2_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  0

def sequential_generatedRoot_78.w_4_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_78.w_5_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_78.w_6_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  1

def sequential_generatedRoot_78.w_7_0  (ctx : sequential_generatedRoot_78Ctx) : Bool :=
  decide ((sequential_generatedRoot_78.w_5_0 ctx) = (sequential_generatedRoot_78.w_6_0 ctx))

def sequential_generatedRoot_78.w_8_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  if (sequential_generatedRoot_78.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_78.w_9_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  ((sequential_generatedRoot_78.w_1_0 ctx) + (sequential_generatedRoot_78.w_8_0 ctx))

def sequential_generatedRoot_78.w_10_0  (ctx : sequential_generatedRoot_78Ctx) : Int :=
  2

def sequential_generatedRoot_78.w_11_0  (ctx : sequential_generatedRoot_78Ctx) : Bool :=
  decide ((sequential_generatedRoot_78.w_9_0 ctx) = (sequential_generatedRoot_78.w_10_0 ctx))

def sequential_generatedRoot_78.outs  (ctx : sequential_generatedRoot_78Ctx) : Bool :=
  (sequential_generatedRoot_78.w_11_0 ctx)

structure sequential_generatedRoot_78.Facts_0  (ctx : sequential_generatedRoot_78Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 params i_0 i (current, ((sequential_generatedRoot_78.w_3_0 ctx), ((sequential_generatedRoot_78.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_78.outs ctx

theorem sequential_generatedRoot_78.facts_0  {ctx : sequential_generatedRoot_78Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_78.constraints_0 params i_0 (sequential_generatedRoot_78.w_3_0 ctx) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_4_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx) (sequential_generatedRoot_78.w_11_0 ctx) outputs) :
    sequential_generatedRoot_78.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_78.Facts  (ctx : sequential_generatedRoot_78Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_78.Facts_0 ctx outputs

theorem sequential_generatedRoot_78.facts  (ctx : sequential_generatedRoot_78Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_78.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_78.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_78.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_80.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_80.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_80Ctx" => (sequential_generatedRoot_80.Ctx params i_0)

def sequential_generatedRoot_80.w_0_0  (ctx : sequential_generatedRoot_80Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_80.w_3_0  (ctx : sequential_generatedRoot_80Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_80.w_1_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  if (sequential_generatedRoot_80.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_80.w_2_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  0

def sequential_generatedRoot_80.w_4_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_80.w_5_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_80.w_6_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  1

def sequential_generatedRoot_80.w_7_0  (ctx : sequential_generatedRoot_80Ctx) : Bool :=
  decide ((sequential_generatedRoot_80.w_5_0 ctx) = (sequential_generatedRoot_80.w_6_0 ctx))

def sequential_generatedRoot_80.w_8_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  if (sequential_generatedRoot_80.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_80.w_9_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  ((sequential_generatedRoot_80.w_1_0 ctx) + (sequential_generatedRoot_80.w_8_0 ctx))

def sequential_generatedRoot_80.w_10_0  (ctx : sequential_generatedRoot_80Ctx) : Int :=
  2

def sequential_generatedRoot_80.w_11_0  (ctx : sequential_generatedRoot_80Ctx) : Bool :=
  decide ((sequential_generatedRoot_80.w_9_0 ctx) = (sequential_generatedRoot_80.w_10_0 ctx))

def sequential_generatedRoot_80.outs  (ctx : sequential_generatedRoot_80Ctx) : Bool :=
  (sequential_generatedRoot_80.w_11_0 ctx)

structure sequential_generatedRoot_80.Facts_0  (ctx : sequential_generatedRoot_80Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 params i_0 i (current, ((sequential_generatedRoot_80.w_3_0 ctx), ((sequential_generatedRoot_80.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_80.outs ctx

theorem sequential_generatedRoot_80.facts_0  {ctx : sequential_generatedRoot_80Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_80.constraints_0 params i_0 (sequential_generatedRoot_80.w_3_0 ctx) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_4_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx) (sequential_generatedRoot_80.w_11_0 ctx) outputs) :
    sequential_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_80.Facts  (ctx : sequential_generatedRoot_80Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_80.Facts_0 ctx outputs

theorem sequential_generatedRoot_80.facts  (ctx : sequential_generatedRoot_80Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_80.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_80.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_82.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_82.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_82Ctx" => (sequential_generatedRoot_82.Ctx params i_0)

def sequential_generatedRoot_82.w_0_0  (ctx : sequential_generatedRoot_82Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_82.w_3_0  (ctx : sequential_generatedRoot_82Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_82.w_1_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  if (sequential_generatedRoot_82.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_82.w_2_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  0

def sequential_generatedRoot_82.w_4_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_82.w_5_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_82.w_6_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  1

def sequential_generatedRoot_82.w_7_0  (ctx : sequential_generatedRoot_82Ctx) : Bool :=
  decide ((sequential_generatedRoot_82.w_5_0 ctx) = (sequential_generatedRoot_82.w_6_0 ctx))

def sequential_generatedRoot_82.w_8_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  if (sequential_generatedRoot_82.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_82.w_9_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  ((sequential_generatedRoot_82.w_1_0 ctx) + (sequential_generatedRoot_82.w_8_0 ctx))

def sequential_generatedRoot_82.w_10_0  (ctx : sequential_generatedRoot_82Ctx) : Int :=
  2

def sequential_generatedRoot_82.w_11_0  (ctx : sequential_generatedRoot_82Ctx) : Bool :=
  decide ((sequential_generatedRoot_82.w_9_0 ctx) = (sequential_generatedRoot_82.w_10_0 ctx))

def sequential_generatedRoot_82.outs  (ctx : sequential_generatedRoot_82Ctx) : Bool :=
  (sequential_generatedRoot_82.w_11_0 ctx)

structure sequential_generatedRoot_82.Facts_0  (ctx : sequential_generatedRoot_82Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 params i_0 i (current, ((sequential_generatedRoot_82.w_3_0 ctx), ((sequential_generatedRoot_82.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_82.outs ctx

theorem sequential_generatedRoot_82.facts_0  {ctx : sequential_generatedRoot_82Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_82.constraints_0 params i_0 (sequential_generatedRoot_82.w_3_0 ctx) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_4_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx) (sequential_generatedRoot_82.w_11_0 ctx) outputs) :
    sequential_generatedRoot_82.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_82.Facts  (ctx : sequential_generatedRoot_82Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_82.Facts_0 ctx outputs

theorem sequential_generatedRoot_82.facts  (ctx : sequential_generatedRoot_82Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_82.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_82.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_82.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_84.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_84.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_84Ctx" => (sequential_generatedRoot_84.Ctx params i_0)

def sequential_generatedRoot_84.w_0_0  (ctx : sequential_generatedRoot_84Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_84.w_3_0  (ctx : sequential_generatedRoot_84Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_84.w_1_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  if (sequential_generatedRoot_84.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_84.w_2_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  0

def sequential_generatedRoot_84.w_4_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_84.w_5_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_84.w_6_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  1

def sequential_generatedRoot_84.w_7_0  (ctx : sequential_generatedRoot_84Ctx) : Bool :=
  decide ((sequential_generatedRoot_84.w_5_0 ctx) = (sequential_generatedRoot_84.w_6_0 ctx))

def sequential_generatedRoot_84.w_8_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  if (sequential_generatedRoot_84.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_84.w_9_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  ((sequential_generatedRoot_84.w_1_0 ctx) + (sequential_generatedRoot_84.w_8_0 ctx))

def sequential_generatedRoot_84.w_10_0  (ctx : sequential_generatedRoot_84Ctx) : Int :=
  2

def sequential_generatedRoot_84.w_11_0  (ctx : sequential_generatedRoot_84Ctx) : Bool :=
  decide ((sequential_generatedRoot_84.w_9_0 ctx) = (sequential_generatedRoot_84.w_10_0 ctx))

def sequential_generatedRoot_84.outs  (ctx : sequential_generatedRoot_84Ctx) : Bool :=
  (sequential_generatedRoot_84.w_11_0 ctx)

structure sequential_generatedRoot_84.Facts_0  (ctx : sequential_generatedRoot_84Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 params i_0 i (current, ((sequential_generatedRoot_84.w_3_0 ctx), ((sequential_generatedRoot_84.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_84.outs ctx

theorem sequential_generatedRoot_84.facts_0  {ctx : sequential_generatedRoot_84Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_84.constraints_0 params i_0 (sequential_generatedRoot_84.w_3_0 ctx) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_4_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx) (sequential_generatedRoot_84.w_11_0 ctx) outputs) :
    sequential_generatedRoot_84.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_84.Facts  (ctx : sequential_generatedRoot_84Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_84.Facts_0 ctx outputs

theorem sequential_generatedRoot_84.facts  (ctx : sequential_generatedRoot_84Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_84.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_84.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_84.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_86.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_86.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_86Ctx" => (sequential_generatedRoot_86.Ctx params i_0)

def sequential_generatedRoot_86.w_0_0  (ctx : sequential_generatedRoot_86Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_86.w_3_0  (ctx : sequential_generatedRoot_86Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_86.w_1_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  if (sequential_generatedRoot_86.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_86.w_2_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  0

def sequential_generatedRoot_86.w_4_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_86.w_5_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_86.w_6_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  1

def sequential_generatedRoot_86.w_7_0  (ctx : sequential_generatedRoot_86Ctx) : Bool :=
  decide ((sequential_generatedRoot_86.w_5_0 ctx) = (sequential_generatedRoot_86.w_6_0 ctx))

def sequential_generatedRoot_86.w_8_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  if (sequential_generatedRoot_86.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_86.w_9_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  ((sequential_generatedRoot_86.w_1_0 ctx) + (sequential_generatedRoot_86.w_8_0 ctx))

def sequential_generatedRoot_86.w_10_0  (ctx : sequential_generatedRoot_86Ctx) : Int :=
  2

def sequential_generatedRoot_86.w_11_0  (ctx : sequential_generatedRoot_86Ctx) : Bool :=
  decide ((sequential_generatedRoot_86.w_9_0 ctx) = (sequential_generatedRoot_86.w_10_0 ctx))

def sequential_generatedRoot_86.outs  (ctx : sequential_generatedRoot_86Ctx) : Bool :=
  (sequential_generatedRoot_86.w_11_0 ctx)

structure sequential_generatedRoot_86.Facts_0  (ctx : sequential_generatedRoot_86Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 params i_0 i (current, ((sequential_generatedRoot_86.w_3_0 ctx), ((sequential_generatedRoot_86.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_86.outs ctx

theorem sequential_generatedRoot_86.facts_0  {ctx : sequential_generatedRoot_86Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_86.constraints_0 params i_0 (sequential_generatedRoot_86.w_3_0 ctx) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_4_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx) (sequential_generatedRoot_86.w_11_0 ctx) outputs) :
    sequential_generatedRoot_86.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_86.Facts  (ctx : sequential_generatedRoot_86Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_86.Facts_0 ctx outputs

theorem sequential_generatedRoot_86.facts  (ctx : sequential_generatedRoot_86Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_86.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_86.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_86.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_88.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_88.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_88Ctx" => (sequential_generatedRoot_88.Ctx params i_0)

def sequential_generatedRoot_88.w_0_0  (ctx : sequential_generatedRoot_88Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_88.w_3_0  (ctx : sequential_generatedRoot_88Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_88.w_1_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  if (sequential_generatedRoot_88.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_88.w_2_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  0

def sequential_generatedRoot_88.w_4_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_88.w_5_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_88.w_6_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  1

def sequential_generatedRoot_88.w_7_0  (ctx : sequential_generatedRoot_88Ctx) : Bool :=
  decide ((sequential_generatedRoot_88.w_5_0 ctx) = (sequential_generatedRoot_88.w_6_0 ctx))

def sequential_generatedRoot_88.w_8_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  if (sequential_generatedRoot_88.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_88.w_9_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  ((sequential_generatedRoot_88.w_1_0 ctx) + (sequential_generatedRoot_88.w_8_0 ctx))

def sequential_generatedRoot_88.w_10_0  (ctx : sequential_generatedRoot_88Ctx) : Int :=
  2

def sequential_generatedRoot_88.w_11_0  (ctx : sequential_generatedRoot_88Ctx) : Bool :=
  decide ((sequential_generatedRoot_88.w_9_0 ctx) = (sequential_generatedRoot_88.w_10_0 ctx))

def sequential_generatedRoot_88.outs  (ctx : sequential_generatedRoot_88Ctx) : Bool :=
  (sequential_generatedRoot_88.w_11_0 ctx)

structure sequential_generatedRoot_88.Facts_0  (ctx : sequential_generatedRoot_88Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 params i_0 i (current, ((sequential_generatedRoot_88.w_3_0 ctx), ((sequential_generatedRoot_88.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_88.outs ctx

theorem sequential_generatedRoot_88.facts_0  {ctx : sequential_generatedRoot_88Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_88.constraints_0 params i_0 (sequential_generatedRoot_88.w_3_0 ctx) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_4_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx) (sequential_generatedRoot_88.w_11_0 ctx) outputs) :
    sequential_generatedRoot_88.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_88.Facts  (ctx : sequential_generatedRoot_88Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_88.Facts_0 ctx outputs

theorem sequential_generatedRoot_88.facts  (ctx : sequential_generatedRoot_88Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_88.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_88.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_88.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_90.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_90.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_90Ctx" => (sequential_generatedRoot_90.Ctx params i_0)

def sequential_generatedRoot_90.w_0_0  (ctx : sequential_generatedRoot_90Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_90.w_3_0  (ctx : sequential_generatedRoot_90Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_90.w_1_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  if (sequential_generatedRoot_90.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_90.w_2_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  0

def sequential_generatedRoot_90.w_4_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_90.w_5_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_90.w_6_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  1

def sequential_generatedRoot_90.w_7_0  (ctx : sequential_generatedRoot_90Ctx) : Bool :=
  decide ((sequential_generatedRoot_90.w_5_0 ctx) = (sequential_generatedRoot_90.w_6_0 ctx))

def sequential_generatedRoot_90.w_8_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  if (sequential_generatedRoot_90.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_90.w_9_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  ((sequential_generatedRoot_90.w_1_0 ctx) + (sequential_generatedRoot_90.w_8_0 ctx))

def sequential_generatedRoot_90.w_10_0  (ctx : sequential_generatedRoot_90Ctx) : Int :=
  2

def sequential_generatedRoot_90.w_11_0  (ctx : sequential_generatedRoot_90Ctx) : Bool :=
  decide ((sequential_generatedRoot_90.w_9_0 ctx) = (sequential_generatedRoot_90.w_10_0 ctx))

def sequential_generatedRoot_90.outs  (ctx : sequential_generatedRoot_90Ctx) : Bool :=
  (sequential_generatedRoot_90.w_11_0 ctx)

structure sequential_generatedRoot_90.Facts_0  (ctx : sequential_generatedRoot_90Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_90_5 params i_0 i (current, ((sequential_generatedRoot_90.w_3_0 ctx), ((sequential_generatedRoot_90.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_90.w_2_0 ctx) (sequential_generatedRoot_90.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_90.outs ctx

theorem sequential_generatedRoot_90.facts_0  {ctx : sequential_generatedRoot_90Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_90.constraints_0 params i_0 (sequential_generatedRoot_90.w_3_0 ctx) (sequential_generatedRoot_90.w_2_0 ctx) (sequential_generatedRoot_90.w_4_0 ctx) (sequential_generatedRoot_90.w_5_0 ctx) (sequential_generatedRoot_90.w_11_0 ctx) outputs) :
    sequential_generatedRoot_90.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_90.Facts  (ctx : sequential_generatedRoot_90Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_90.Facts_0 ctx outputs

theorem sequential_generatedRoot_90.facts  (ctx : sequential_generatedRoot_90Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_90.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_90.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_90.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_92.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_92.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_92Ctx" => (sequential_generatedRoot_92.Ctx params i_0)

def sequential_generatedRoot_92.w_0_0  (ctx : sequential_generatedRoot_92Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_92.w_3_0  (ctx : sequential_generatedRoot_92Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_92.w_1_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  if (sequential_generatedRoot_92.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_92.w_2_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  0

def sequential_generatedRoot_92.w_4_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_92.w_5_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_92.w_6_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  1

def sequential_generatedRoot_92.w_7_0  (ctx : sequential_generatedRoot_92Ctx) : Bool :=
  decide ((sequential_generatedRoot_92.w_5_0 ctx) = (sequential_generatedRoot_92.w_6_0 ctx))

def sequential_generatedRoot_92.w_8_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  if (sequential_generatedRoot_92.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_92.w_9_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  ((sequential_generatedRoot_92.w_1_0 ctx) + (sequential_generatedRoot_92.w_8_0 ctx))

def sequential_generatedRoot_92.w_10_0  (ctx : sequential_generatedRoot_92Ctx) : Int :=
  2

def sequential_generatedRoot_92.w_11_0  (ctx : sequential_generatedRoot_92Ctx) : Bool :=
  decide ((sequential_generatedRoot_92.w_9_0 ctx) = (sequential_generatedRoot_92.w_10_0 ctx))

def sequential_generatedRoot_92.outs  (ctx : sequential_generatedRoot_92Ctx) : Bool :=
  (sequential_generatedRoot_92.w_11_0 ctx)

structure sequential_generatedRoot_92.Facts_0  (ctx : sequential_generatedRoot_92Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_92_5 params i_0 i (current, ((sequential_generatedRoot_92.w_3_0 ctx), ((sequential_generatedRoot_92.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_92.w_2_0 ctx) (sequential_generatedRoot_92.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_92.outs ctx

theorem sequential_generatedRoot_92.facts_0  {ctx : sequential_generatedRoot_92Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_92.constraints_0 params i_0 (sequential_generatedRoot_92.w_3_0 ctx) (sequential_generatedRoot_92.w_2_0 ctx) (sequential_generatedRoot_92.w_4_0 ctx) (sequential_generatedRoot_92.w_5_0 ctx) (sequential_generatedRoot_92.w_11_0 ctx) outputs) :
    sequential_generatedRoot_92.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_92.Facts  (ctx : sequential_generatedRoot_92Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_92.Facts_0 ctx outputs

theorem sequential_generatedRoot_92.facts  (ctx : sequential_generatedRoot_92Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_92.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_92.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_92.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_94.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_94.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_94Ctx" => (sequential_generatedRoot_94.Ctx params i_0)

def sequential_generatedRoot_94.w_0_0  (ctx : sequential_generatedRoot_94Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_94.w_3_0  (ctx : sequential_generatedRoot_94Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_94.w_1_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  if (sequential_generatedRoot_94.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_94.w_2_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  0

def sequential_generatedRoot_94.w_4_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_94.w_5_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_94.w_6_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  1

def sequential_generatedRoot_94.w_7_0  (ctx : sequential_generatedRoot_94Ctx) : Bool :=
  decide ((sequential_generatedRoot_94.w_5_0 ctx) = (sequential_generatedRoot_94.w_6_0 ctx))

def sequential_generatedRoot_94.w_8_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  if (sequential_generatedRoot_94.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_94.w_9_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  ((sequential_generatedRoot_94.w_1_0 ctx) + (sequential_generatedRoot_94.w_8_0 ctx))

def sequential_generatedRoot_94.w_10_0  (ctx : sequential_generatedRoot_94Ctx) : Int :=
  2

def sequential_generatedRoot_94.w_11_0  (ctx : sequential_generatedRoot_94Ctx) : Bool :=
  decide ((sequential_generatedRoot_94.w_9_0 ctx) = (sequential_generatedRoot_94.w_10_0 ctx))

def sequential_generatedRoot_94.outs  (ctx : sequential_generatedRoot_94Ctx) : Bool :=
  (sequential_generatedRoot_94.w_11_0 ctx)

structure sequential_generatedRoot_94.Facts_0  (ctx : sequential_generatedRoot_94Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_94_5 params i_0 i (current, ((sequential_generatedRoot_94.w_3_0 ctx), ((sequential_generatedRoot_94.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_94.w_2_0 ctx) (sequential_generatedRoot_94.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_94.outs ctx

theorem sequential_generatedRoot_94.facts_0  {ctx : sequential_generatedRoot_94Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_94.constraints_0 params i_0 (sequential_generatedRoot_94.w_3_0 ctx) (sequential_generatedRoot_94.w_2_0 ctx) (sequential_generatedRoot_94.w_4_0 ctx) (sequential_generatedRoot_94.w_5_0 ctx) (sequential_generatedRoot_94.w_11_0 ctx) outputs) :
    sequential_generatedRoot_94.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_94.Facts  (ctx : sequential_generatedRoot_94Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_94.Facts_0 ctx outputs

theorem sequential_generatedRoot_94.facts  (ctx : sequential_generatedRoot_94Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_94.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_94.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_94.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_96.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_96.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_96Ctx" => (sequential_generatedRoot_96.Ctx params i_0)

def sequential_generatedRoot_96.w_0_0  (ctx : sequential_generatedRoot_96Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_96.w_3_0  (ctx : sequential_generatedRoot_96Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_96.w_1_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  if (sequential_generatedRoot_96.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_96.w_2_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  0

def sequential_generatedRoot_96.w_4_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_96.w_5_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_96.w_6_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  1

def sequential_generatedRoot_96.w_7_0  (ctx : sequential_generatedRoot_96Ctx) : Bool :=
  decide ((sequential_generatedRoot_96.w_5_0 ctx) = (sequential_generatedRoot_96.w_6_0 ctx))

def sequential_generatedRoot_96.w_8_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  if (sequential_generatedRoot_96.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_96.w_9_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  ((sequential_generatedRoot_96.w_1_0 ctx) + (sequential_generatedRoot_96.w_8_0 ctx))

def sequential_generatedRoot_96.w_10_0  (ctx : sequential_generatedRoot_96Ctx) : Int :=
  2

def sequential_generatedRoot_96.w_11_0  (ctx : sequential_generatedRoot_96Ctx) : Bool :=
  decide ((sequential_generatedRoot_96.w_9_0 ctx) = (sequential_generatedRoot_96.w_10_0 ctx))

def sequential_generatedRoot_96.outs  (ctx : sequential_generatedRoot_96Ctx) : Bool :=
  (sequential_generatedRoot_96.w_11_0 ctx)

structure sequential_generatedRoot_96.Facts_0  (ctx : sequential_generatedRoot_96Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_96_5 params i_0 i (current, ((sequential_generatedRoot_96.w_3_0 ctx), ((sequential_generatedRoot_96.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_96.w_2_0 ctx) (sequential_generatedRoot_96.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_96.outs ctx

theorem sequential_generatedRoot_96.facts_0  {ctx : sequential_generatedRoot_96Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_96.constraints_0 params i_0 (sequential_generatedRoot_96.w_3_0 ctx) (sequential_generatedRoot_96.w_2_0 ctx) (sequential_generatedRoot_96.w_4_0 ctx) (sequential_generatedRoot_96.w_5_0 ctx) (sequential_generatedRoot_96.w_11_0 ctx) outputs) :
    sequential_generatedRoot_96.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_96.Facts  (ctx : sequential_generatedRoot_96Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_96.Facts_0 ctx outputs

theorem sequential_generatedRoot_96.facts  (ctx : sequential_generatedRoot_96Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_96.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_96.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_96.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_98.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_98.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_98Ctx" => (sequential_generatedRoot_98.Ctx params i_0)

def sequential_generatedRoot_98.w_0_0  (ctx : sequential_generatedRoot_98Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_98.w_3_0  (ctx : sequential_generatedRoot_98Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_98.w_1_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  if (sequential_generatedRoot_98.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_98.w_2_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  0

def sequential_generatedRoot_98.w_4_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_98.w_5_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_98.w_6_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  1

def sequential_generatedRoot_98.w_7_0  (ctx : sequential_generatedRoot_98Ctx) : Bool :=
  decide ((sequential_generatedRoot_98.w_5_0 ctx) = (sequential_generatedRoot_98.w_6_0 ctx))

def sequential_generatedRoot_98.w_8_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  if (sequential_generatedRoot_98.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_98.w_9_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  ((sequential_generatedRoot_98.w_1_0 ctx) + (sequential_generatedRoot_98.w_8_0 ctx))

def sequential_generatedRoot_98.w_10_0  (ctx : sequential_generatedRoot_98Ctx) : Int :=
  2

def sequential_generatedRoot_98.w_11_0  (ctx : sequential_generatedRoot_98Ctx) : Bool :=
  decide ((sequential_generatedRoot_98.w_9_0 ctx) = (sequential_generatedRoot_98.w_10_0 ctx))

def sequential_generatedRoot_98.outs  (ctx : sequential_generatedRoot_98Ctx) : Bool :=
  (sequential_generatedRoot_98.w_11_0 ctx)

structure sequential_generatedRoot_98.Facts_0  (ctx : sequential_generatedRoot_98Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_98_5 params i_0 i (current, ((sequential_generatedRoot_98.w_3_0 ctx), ((sequential_generatedRoot_98.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_98.w_2_0 ctx) (sequential_generatedRoot_98.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_98.outs ctx

theorem sequential_generatedRoot_98.facts_0  {ctx : sequential_generatedRoot_98Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_98.constraints_0 params i_0 (sequential_generatedRoot_98.w_3_0 ctx) (sequential_generatedRoot_98.w_2_0 ctx) (sequential_generatedRoot_98.w_4_0 ctx) (sequential_generatedRoot_98.w_5_0 ctx) (sequential_generatedRoot_98.w_11_0 ctx) outputs) :
    sequential_generatedRoot_98.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_98.Facts  (ctx : sequential_generatedRoot_98Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_98.Facts_0 ctx outputs

theorem sequential_generatedRoot_98.facts  (ctx : sequential_generatedRoot_98Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_98.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_98.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_98.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_100.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_100.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_100Ctx" => (sequential_generatedRoot_100.Ctx params i_0)

def sequential_generatedRoot_100.w_0_0  (ctx : sequential_generatedRoot_100Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_100.w_3_0  (ctx : sequential_generatedRoot_100Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_100.w_1_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  if (sequential_generatedRoot_100.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_100.w_2_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  0

def sequential_generatedRoot_100.w_4_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_100.w_5_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_100.w_6_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  1

def sequential_generatedRoot_100.w_7_0  (ctx : sequential_generatedRoot_100Ctx) : Bool :=
  decide ((sequential_generatedRoot_100.w_5_0 ctx) = (sequential_generatedRoot_100.w_6_0 ctx))

def sequential_generatedRoot_100.w_8_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  if (sequential_generatedRoot_100.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_100.w_9_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  ((sequential_generatedRoot_100.w_1_0 ctx) + (sequential_generatedRoot_100.w_8_0 ctx))

def sequential_generatedRoot_100.w_10_0  (ctx : sequential_generatedRoot_100Ctx) : Int :=
  2

def sequential_generatedRoot_100.w_11_0  (ctx : sequential_generatedRoot_100Ctx) : Bool :=
  decide ((sequential_generatedRoot_100.w_9_0 ctx) = (sequential_generatedRoot_100.w_10_0 ctx))

def sequential_generatedRoot_100.outs  (ctx : sequential_generatedRoot_100Ctx) : Bool :=
  (sequential_generatedRoot_100.w_11_0 ctx)

structure sequential_generatedRoot_100.Facts_0  (ctx : sequential_generatedRoot_100Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_155 params i_0 i (current, ((sequential_generatedRoot_100.w_3_0 ctx), ((sequential_generatedRoot_100.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_100.w_2_0 ctx) (sequential_generatedRoot_100.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_100.outs ctx

theorem sequential_generatedRoot_100.facts_0  {ctx : sequential_generatedRoot_100Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_100.constraints_0 params i_0 (sequential_generatedRoot_100.w_3_0 ctx) (sequential_generatedRoot_100.w_2_0 ctx) (sequential_generatedRoot_100.w_4_0 ctx) (sequential_generatedRoot_100.w_5_0 ctx) (sequential_generatedRoot_100.w_11_0 ctx) outputs) :
    sequential_generatedRoot_100.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_100.Facts  (ctx : sequential_generatedRoot_100Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_100.Facts_0 ctx outputs

theorem sequential_generatedRoot_100.facts  (ctx : sequential_generatedRoot_100Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_100.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_100.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_100.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_102.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_102.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_102Ctx" => (sequential_generatedRoot_102.Ctx params i_0)

def sequential_generatedRoot_102.w_0_0  (ctx : sequential_generatedRoot_102Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_102.w_3_0  (ctx : sequential_generatedRoot_102Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_102.w_1_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  if (sequential_generatedRoot_102.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_102.w_2_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  0

def sequential_generatedRoot_102.w_4_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_102.w_5_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_102.w_6_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  1

def sequential_generatedRoot_102.w_7_0  (ctx : sequential_generatedRoot_102Ctx) : Bool :=
  decide ((sequential_generatedRoot_102.w_5_0 ctx) = (sequential_generatedRoot_102.w_6_0 ctx))

def sequential_generatedRoot_102.w_8_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  if (sequential_generatedRoot_102.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_102.w_9_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  ((sequential_generatedRoot_102.w_1_0 ctx) + (sequential_generatedRoot_102.w_8_0 ctx))

def sequential_generatedRoot_102.w_10_0  (ctx : sequential_generatedRoot_102Ctx) : Int :=
  2

def sequential_generatedRoot_102.w_11_0  (ctx : sequential_generatedRoot_102Ctx) : Bool :=
  decide ((sequential_generatedRoot_102.w_9_0 ctx) = (sequential_generatedRoot_102.w_10_0 ctx))

def sequential_generatedRoot_102.outs  (ctx : sequential_generatedRoot_102Ctx) : Bool :=
  (sequential_generatedRoot_102.w_11_0 ctx)

structure sequential_generatedRoot_102.Facts_0  (ctx : sequential_generatedRoot_102Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_156 params i_0 i (current, ((sequential_generatedRoot_102.w_3_0 ctx), ((sequential_generatedRoot_102.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_102.w_2_0 ctx) (sequential_generatedRoot_102.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_102.outs ctx

theorem sequential_generatedRoot_102.facts_0  {ctx : sequential_generatedRoot_102Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_102.constraints_0 params i_0 (sequential_generatedRoot_102.w_3_0 ctx) (sequential_generatedRoot_102.w_2_0 ctx) (sequential_generatedRoot_102.w_4_0 ctx) (sequential_generatedRoot_102.w_5_0 ctx) (sequential_generatedRoot_102.w_11_0 ctx) outputs) :
    sequential_generatedRoot_102.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_102.Facts  (ctx : sequential_generatedRoot_102Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_102.Facts_0 ctx outputs

theorem sequential_generatedRoot_102.facts  (ctx : sequential_generatedRoot_102Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_102.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_102.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_102.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_104.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_104.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_104Ctx" => (sequential_generatedRoot_104.Ctx params i_0)

def sequential_generatedRoot_104.w_0_0  (ctx : sequential_generatedRoot_104Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_104.w_3_0  (ctx : sequential_generatedRoot_104Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_104.w_1_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  if (sequential_generatedRoot_104.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_104.w_2_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  0

def sequential_generatedRoot_104.w_4_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_104.w_5_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_104.w_6_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  1

def sequential_generatedRoot_104.w_7_0  (ctx : sequential_generatedRoot_104Ctx) : Bool :=
  decide ((sequential_generatedRoot_104.w_5_0 ctx) = (sequential_generatedRoot_104.w_6_0 ctx))

def sequential_generatedRoot_104.w_8_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  if (sequential_generatedRoot_104.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_104.w_9_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  ((sequential_generatedRoot_104.w_1_0 ctx) + (sequential_generatedRoot_104.w_8_0 ctx))

def sequential_generatedRoot_104.w_10_0  (ctx : sequential_generatedRoot_104Ctx) : Int :=
  2

def sequential_generatedRoot_104.w_11_0  (ctx : sequential_generatedRoot_104Ctx) : Bool :=
  decide ((sequential_generatedRoot_104.w_9_0 ctx) = (sequential_generatedRoot_104.w_10_0 ctx))

def sequential_generatedRoot_104.outs  (ctx : sequential_generatedRoot_104Ctx) : Bool :=
  (sequential_generatedRoot_104.w_11_0 ctx)

structure sequential_generatedRoot_104.Facts_0  (ctx : sequential_generatedRoot_104Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_157 params i_0 i (current, ((sequential_generatedRoot_104.w_3_0 ctx), ((sequential_generatedRoot_104.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_104.w_2_0 ctx) (sequential_generatedRoot_104.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_104.outs ctx

theorem sequential_generatedRoot_104.facts_0  {ctx : sequential_generatedRoot_104Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_104.constraints_0 params i_0 (sequential_generatedRoot_104.w_3_0 ctx) (sequential_generatedRoot_104.w_2_0 ctx) (sequential_generatedRoot_104.w_4_0 ctx) (sequential_generatedRoot_104.w_5_0 ctx) (sequential_generatedRoot_104.w_11_0 ctx) outputs) :
    sequential_generatedRoot_104.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_104.Facts  (ctx : sequential_generatedRoot_104Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_104.Facts_0 ctx outputs

theorem sequential_generatedRoot_104.facts  (ctx : sequential_generatedRoot_104Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_104.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_104.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_104.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_106.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_106.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_106Ctx" => (sequential_generatedRoot_106.Ctx params i_0)

def sequential_generatedRoot_106.w_0_0  (ctx : sequential_generatedRoot_106Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_106.w_3_0  (ctx : sequential_generatedRoot_106Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_106.w_1_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  if (sequential_generatedRoot_106.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_106.w_2_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  0

def sequential_generatedRoot_106.w_4_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_106.w_5_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_106.w_6_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  1

def sequential_generatedRoot_106.w_7_0  (ctx : sequential_generatedRoot_106Ctx) : Bool :=
  decide ((sequential_generatedRoot_106.w_5_0 ctx) = (sequential_generatedRoot_106.w_6_0 ctx))

def sequential_generatedRoot_106.w_8_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  if (sequential_generatedRoot_106.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_106.w_9_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  ((sequential_generatedRoot_106.w_1_0 ctx) + (sequential_generatedRoot_106.w_8_0 ctx))

def sequential_generatedRoot_106.w_10_0  (ctx : sequential_generatedRoot_106Ctx) : Int :=
  2

def sequential_generatedRoot_106.w_11_0  (ctx : sequential_generatedRoot_106Ctx) : Bool :=
  decide ((sequential_generatedRoot_106.w_9_0 ctx) = (sequential_generatedRoot_106.w_10_0 ctx))

def sequential_generatedRoot_106.outs  (ctx : sequential_generatedRoot_106Ctx) : Bool :=
  (sequential_generatedRoot_106.w_11_0 ctx)

structure sequential_generatedRoot_106.Facts_0  (ctx : sequential_generatedRoot_106Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_158 params i_0 i (current, ((sequential_generatedRoot_106.w_3_0 ctx), ((sequential_generatedRoot_106.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_106.w_2_0 ctx) (sequential_generatedRoot_106.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_106.outs ctx

theorem sequential_generatedRoot_106.facts_0  {ctx : sequential_generatedRoot_106Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_106.constraints_0 params i_0 (sequential_generatedRoot_106.w_3_0 ctx) (sequential_generatedRoot_106.w_2_0 ctx) (sequential_generatedRoot_106.w_4_0 ctx) (sequential_generatedRoot_106.w_5_0 ctx) (sequential_generatedRoot_106.w_11_0 ctx) outputs) :
    sequential_generatedRoot_106.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_106.Facts  (ctx : sequential_generatedRoot_106Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_106.Facts_0 ctx outputs

theorem sequential_generatedRoot_106.facts  (ctx : sequential_generatedRoot_106Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_106.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_106.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_106.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_108.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_108.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_108Ctx" => (sequential_generatedRoot_108.Ctx params i_0)

def sequential_generatedRoot_108.w_0_0  (ctx : sequential_generatedRoot_108Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_108.w_3_0  (ctx : sequential_generatedRoot_108Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_108.w_1_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  if (sequential_generatedRoot_108.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_108.w_2_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  0

def sequential_generatedRoot_108.w_4_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_108.w_5_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_108.w_6_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  1

def sequential_generatedRoot_108.w_7_0  (ctx : sequential_generatedRoot_108Ctx) : Bool :=
  decide ((sequential_generatedRoot_108.w_5_0 ctx) = (sequential_generatedRoot_108.w_6_0 ctx))

def sequential_generatedRoot_108.w_8_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  if (sequential_generatedRoot_108.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_108.w_9_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  ((sequential_generatedRoot_108.w_1_0 ctx) + (sequential_generatedRoot_108.w_8_0 ctx))

def sequential_generatedRoot_108.w_10_0  (ctx : sequential_generatedRoot_108Ctx) : Int :=
  2

def sequential_generatedRoot_108.w_11_0  (ctx : sequential_generatedRoot_108Ctx) : Bool :=
  decide ((sequential_generatedRoot_108.w_9_0 ctx) = (sequential_generatedRoot_108.w_10_0 ctx))

def sequential_generatedRoot_108.outs  (ctx : sequential_generatedRoot_108Ctx) : Bool :=
  (sequential_generatedRoot_108.w_11_0 ctx)

structure sequential_generatedRoot_108.Facts_0  (ctx : sequential_generatedRoot_108Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_159 params i_0 i (current, ((sequential_generatedRoot_108.w_3_0 ctx), ((sequential_generatedRoot_108.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_108.w_2_0 ctx) (sequential_generatedRoot_108.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_108.outs ctx

theorem sequential_generatedRoot_108.facts_0  {ctx : sequential_generatedRoot_108Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_108.constraints_0 params i_0 (sequential_generatedRoot_108.w_3_0 ctx) (sequential_generatedRoot_108.w_2_0 ctx) (sequential_generatedRoot_108.w_4_0 ctx) (sequential_generatedRoot_108.w_5_0 ctx) (sequential_generatedRoot_108.w_11_0 ctx) outputs) :
    sequential_generatedRoot_108.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_108.Facts  (ctx : sequential_generatedRoot_108Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_108.Facts_0 ctx outputs

theorem sequential_generatedRoot_108.facts  (ctx : sequential_generatedRoot_108Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_108.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_108.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_108.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_110.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_110.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_110Ctx" => (sequential_generatedRoot_110.Ctx params i_0)

def sequential_generatedRoot_110.w_0_0  (ctx : sequential_generatedRoot_110Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_110.w_3_0  (ctx : sequential_generatedRoot_110Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_110.w_1_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  if (sequential_generatedRoot_110.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_110.w_2_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  0

def sequential_generatedRoot_110.w_4_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_110.w_5_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_110.w_6_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  1

def sequential_generatedRoot_110.w_7_0  (ctx : sequential_generatedRoot_110Ctx) : Bool :=
  decide ((sequential_generatedRoot_110.w_5_0 ctx) = (sequential_generatedRoot_110.w_6_0 ctx))

def sequential_generatedRoot_110.w_8_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  if (sequential_generatedRoot_110.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_110.w_9_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  ((sequential_generatedRoot_110.w_1_0 ctx) + (sequential_generatedRoot_110.w_8_0 ctx))

def sequential_generatedRoot_110.w_10_0  (ctx : sequential_generatedRoot_110Ctx) : Int :=
  2

def sequential_generatedRoot_110.w_11_0  (ctx : sequential_generatedRoot_110Ctx) : Bool :=
  decide ((sequential_generatedRoot_110.w_9_0 ctx) = (sequential_generatedRoot_110.w_10_0 ctx))

def sequential_generatedRoot_110.outs  (ctx : sequential_generatedRoot_110Ctx) : Bool :=
  (sequential_generatedRoot_110.w_11_0 ctx)

structure sequential_generatedRoot_110.Facts_0  (ctx : sequential_generatedRoot_110Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_160 params i_0 i (current, ((sequential_generatedRoot_110.w_3_0 ctx), ((sequential_generatedRoot_110.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_110.w_2_0 ctx) (sequential_generatedRoot_110.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_110.outs ctx

theorem sequential_generatedRoot_110.facts_0  {ctx : sequential_generatedRoot_110Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_110.constraints_0 params i_0 (sequential_generatedRoot_110.w_3_0 ctx) (sequential_generatedRoot_110.w_2_0 ctx) (sequential_generatedRoot_110.w_4_0 ctx) (sequential_generatedRoot_110.w_5_0 ctx) (sequential_generatedRoot_110.w_11_0 ctx) outputs) :
    sequential_generatedRoot_110.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_110.Facts  (ctx : sequential_generatedRoot_110Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_110.Facts_0 ctx outputs

theorem sequential_generatedRoot_110.facts  (ctx : sequential_generatedRoot_110Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_110.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_110.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_110.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_112.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_112.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_112Ctx" => (sequential_generatedRoot_112.Ctx params i_0)

def sequential_generatedRoot_112.w_0_0  (ctx : sequential_generatedRoot_112Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_112.w_3_0  (ctx : sequential_generatedRoot_112Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_112.w_1_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  if (sequential_generatedRoot_112.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_112.w_2_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  0

def sequential_generatedRoot_112.w_4_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_112.w_5_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_112.w_6_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  1

def sequential_generatedRoot_112.w_7_0  (ctx : sequential_generatedRoot_112Ctx) : Bool :=
  decide ((sequential_generatedRoot_112.w_5_0 ctx) = (sequential_generatedRoot_112.w_6_0 ctx))

def sequential_generatedRoot_112.w_8_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  if (sequential_generatedRoot_112.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_112.w_9_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  ((sequential_generatedRoot_112.w_1_0 ctx) + (sequential_generatedRoot_112.w_8_0 ctx))

def sequential_generatedRoot_112.w_10_0  (ctx : sequential_generatedRoot_112Ctx) : Int :=
  2

def sequential_generatedRoot_112.w_11_0  (ctx : sequential_generatedRoot_112Ctx) : Bool :=
  decide ((sequential_generatedRoot_112.w_9_0 ctx) = (sequential_generatedRoot_112.w_10_0 ctx))

def sequential_generatedRoot_112.outs  (ctx : sequential_generatedRoot_112Ctx) : Bool :=
  (sequential_generatedRoot_112.w_11_0 ctx)

structure sequential_generatedRoot_112.Facts_0  (ctx : sequential_generatedRoot_112Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_161 params i_0 i (current, ((sequential_generatedRoot_112.w_3_0 ctx), ((sequential_generatedRoot_112.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_112.w_2_0 ctx) (sequential_generatedRoot_112.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_112.outs ctx

theorem sequential_generatedRoot_112.facts_0  {ctx : sequential_generatedRoot_112Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_112.constraints_0 params i_0 (sequential_generatedRoot_112.w_3_0 ctx) (sequential_generatedRoot_112.w_2_0 ctx) (sequential_generatedRoot_112.w_4_0 ctx) (sequential_generatedRoot_112.w_5_0 ctx) (sequential_generatedRoot_112.w_11_0 ctx) outputs) :
    sequential_generatedRoot_112.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_112.Facts  (ctx : sequential_generatedRoot_112Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_112.Facts_0 ctx outputs

theorem sequential_generatedRoot_112.facts  (ctx : sequential_generatedRoot_112Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_112.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_112.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_112.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_114.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_114.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_114Ctx" => (sequential_generatedRoot_114.Ctx params i_0)

def sequential_generatedRoot_114.w_0_0  (ctx : sequential_generatedRoot_114Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_114.w_3_0  (ctx : sequential_generatedRoot_114Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_114.w_1_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  if (sequential_generatedRoot_114.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_114.w_2_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  0

def sequential_generatedRoot_114.w_4_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_114.w_5_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_114.w_6_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  1

def sequential_generatedRoot_114.w_7_0  (ctx : sequential_generatedRoot_114Ctx) : Bool :=
  decide ((sequential_generatedRoot_114.w_5_0 ctx) = (sequential_generatedRoot_114.w_6_0 ctx))

def sequential_generatedRoot_114.w_8_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  if (sequential_generatedRoot_114.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_114.w_9_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  ((sequential_generatedRoot_114.w_1_0 ctx) + (sequential_generatedRoot_114.w_8_0 ctx))

def sequential_generatedRoot_114.w_10_0  (ctx : sequential_generatedRoot_114Ctx) : Int :=
  2

def sequential_generatedRoot_114.w_11_0  (ctx : sequential_generatedRoot_114Ctx) : Bool :=
  decide ((sequential_generatedRoot_114.w_9_0 ctx) = (sequential_generatedRoot_114.w_10_0 ctx))

def sequential_generatedRoot_114.outs  (ctx : sequential_generatedRoot_114Ctx) : Bool :=
  (sequential_generatedRoot_114.w_11_0 ctx)

structure sequential_generatedRoot_114.Facts_0  (ctx : sequential_generatedRoot_114Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_162 params i_0 i (current, ((sequential_generatedRoot_114.w_3_0 ctx), ((sequential_generatedRoot_114.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_114.w_2_0 ctx) (sequential_generatedRoot_114.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_114.outs ctx

theorem sequential_generatedRoot_114.facts_0  {ctx : sequential_generatedRoot_114Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_114.constraints_0 params i_0 (sequential_generatedRoot_114.w_3_0 ctx) (sequential_generatedRoot_114.w_2_0 ctx) (sequential_generatedRoot_114.w_4_0 ctx) (sequential_generatedRoot_114.w_5_0 ctx) (sequential_generatedRoot_114.w_11_0 ctx) outputs) :
    sequential_generatedRoot_114.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_114.Facts  (ctx : sequential_generatedRoot_114Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_114.Facts_0 ctx outputs

theorem sequential_generatedRoot_114.facts  (ctx : sequential_generatedRoot_114Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_114.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_114.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_114.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_116.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_116.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_116Ctx" => (sequential_generatedRoot_116.Ctx params i_0)

def sequential_generatedRoot_116.w_0_0  (ctx : sequential_generatedRoot_116Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_116.w_3_0  (ctx : sequential_generatedRoot_116Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_116.w_1_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  if (sequential_generatedRoot_116.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_116.w_2_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  0

def sequential_generatedRoot_116.w_4_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_116.w_5_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_116.w_6_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  1

def sequential_generatedRoot_116.w_7_0  (ctx : sequential_generatedRoot_116Ctx) : Bool :=
  decide ((sequential_generatedRoot_116.w_5_0 ctx) = (sequential_generatedRoot_116.w_6_0 ctx))

def sequential_generatedRoot_116.w_8_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  if (sequential_generatedRoot_116.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_116.w_9_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  ((sequential_generatedRoot_116.w_1_0 ctx) + (sequential_generatedRoot_116.w_8_0 ctx))

def sequential_generatedRoot_116.w_10_0  (ctx : sequential_generatedRoot_116Ctx) : Int :=
  2

def sequential_generatedRoot_116.w_11_0  (ctx : sequential_generatedRoot_116Ctx) : Bool :=
  decide ((sequential_generatedRoot_116.w_9_0 ctx) = (sequential_generatedRoot_116.w_10_0 ctx))

def sequential_generatedRoot_116.outs  (ctx : sequential_generatedRoot_116Ctx) : Bool :=
  (sequential_generatedRoot_116.w_11_0 ctx)

structure sequential_generatedRoot_116.Facts_0  (ctx : sequential_generatedRoot_116Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_163 params i_0 i (current, ((sequential_generatedRoot_116.w_3_0 ctx), ((sequential_generatedRoot_116.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_116.w_2_0 ctx) (sequential_generatedRoot_116.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_116.outs ctx

theorem sequential_generatedRoot_116.facts_0  {ctx : sequential_generatedRoot_116Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_116.constraints_0 params i_0 (sequential_generatedRoot_116.w_3_0 ctx) (sequential_generatedRoot_116.w_2_0 ctx) (sequential_generatedRoot_116.w_4_0 ctx) (sequential_generatedRoot_116.w_5_0 ctx) (sequential_generatedRoot_116.w_11_0 ctx) outputs) :
    sequential_generatedRoot_116.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_116.Facts  (ctx : sequential_generatedRoot_116Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_116.Facts_0 ctx outputs

theorem sequential_generatedRoot_116.facts  (ctx : sequential_generatedRoot_116Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_116.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_116.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_116.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_118.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_118.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_118Ctx" => (sequential_generatedRoot_118.Ctx params i_0)

def sequential_generatedRoot_118.w_0_0  (ctx : sequential_generatedRoot_118Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_118.w_3_0  (ctx : sequential_generatedRoot_118Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_118.w_1_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  if (sequential_generatedRoot_118.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_118.w_2_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  0

def sequential_generatedRoot_118.w_4_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_118.w_5_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_118.w_6_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  1

def sequential_generatedRoot_118.w_7_0  (ctx : sequential_generatedRoot_118Ctx) : Bool :=
  decide ((sequential_generatedRoot_118.w_5_0 ctx) = (sequential_generatedRoot_118.w_6_0 ctx))

def sequential_generatedRoot_118.w_8_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  if (sequential_generatedRoot_118.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_118.w_9_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  ((sequential_generatedRoot_118.w_1_0 ctx) + (sequential_generatedRoot_118.w_8_0 ctx))

def sequential_generatedRoot_118.w_10_0  (ctx : sequential_generatedRoot_118Ctx) : Int :=
  2

def sequential_generatedRoot_118.w_11_0  (ctx : sequential_generatedRoot_118Ctx) : Bool :=
  decide ((sequential_generatedRoot_118.w_9_0 ctx) = (sequential_generatedRoot_118.w_10_0 ctx))

def sequential_generatedRoot_118.outs  (ctx : sequential_generatedRoot_118Ctx) : Bool :=
  (sequential_generatedRoot_118.w_11_0 ctx)

structure sequential_generatedRoot_118.Facts_0  (ctx : sequential_generatedRoot_118Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_164 params i_0 i (current, ((sequential_generatedRoot_118.w_3_0 ctx), ((sequential_generatedRoot_118.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_118.w_2_0 ctx) (sequential_generatedRoot_118.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_118.outs ctx

theorem sequential_generatedRoot_118.facts_0  {ctx : sequential_generatedRoot_118Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_118.constraints_0 params i_0 (sequential_generatedRoot_118.w_3_0 ctx) (sequential_generatedRoot_118.w_2_0 ctx) (sequential_generatedRoot_118.w_4_0 ctx) (sequential_generatedRoot_118.w_5_0 ctx) (sequential_generatedRoot_118.w_11_0 ctx) outputs) :
    sequential_generatedRoot_118.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_118.Facts  (ctx : sequential_generatedRoot_118Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_118.Facts_0 ctx outputs

theorem sequential_generatedRoot_118.facts  (ctx : sequential_generatedRoot_118Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_118.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_118.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_118.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_120.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_120.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_120Ctx" => (sequential_generatedRoot_120.Ctx params i_0)

def sequential_generatedRoot_120.w_0_0  (ctx : sequential_generatedRoot_120Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_120.w_3_0  (ctx : sequential_generatedRoot_120Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_120.w_1_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  if (sequential_generatedRoot_120.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_120.w_2_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  0

def sequential_generatedRoot_120.w_4_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_120.w_5_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_120.w_6_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  1

def sequential_generatedRoot_120.w_7_0  (ctx : sequential_generatedRoot_120Ctx) : Bool :=
  decide ((sequential_generatedRoot_120.w_5_0 ctx) = (sequential_generatedRoot_120.w_6_0 ctx))

def sequential_generatedRoot_120.w_8_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  if (sequential_generatedRoot_120.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_120.w_9_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  ((sequential_generatedRoot_120.w_1_0 ctx) + (sequential_generatedRoot_120.w_8_0 ctx))

def sequential_generatedRoot_120.w_10_0  (ctx : sequential_generatedRoot_120Ctx) : Int :=
  2

def sequential_generatedRoot_120.w_11_0  (ctx : sequential_generatedRoot_120Ctx) : Bool :=
  decide ((sequential_generatedRoot_120.w_9_0 ctx) = (sequential_generatedRoot_120.w_10_0 ctx))

def sequential_generatedRoot_120.outs  (ctx : sequential_generatedRoot_120Ctx) : Bool :=
  (sequential_generatedRoot_120.w_11_0 ctx)

structure sequential_generatedRoot_120.Facts_0  (ctx : sequential_generatedRoot_120Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_165 params i_0 i (current, ((sequential_generatedRoot_120.w_3_0 ctx), ((sequential_generatedRoot_120.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_120.w_2_0 ctx) (sequential_generatedRoot_120.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_120.outs ctx

theorem sequential_generatedRoot_120.facts_0  {ctx : sequential_generatedRoot_120Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_120.constraints_0 params i_0 (sequential_generatedRoot_120.w_3_0 ctx) (sequential_generatedRoot_120.w_2_0 ctx) (sequential_generatedRoot_120.w_4_0 ctx) (sequential_generatedRoot_120.w_5_0 ctx) (sequential_generatedRoot_120.w_11_0 ctx) outputs) :
    sequential_generatedRoot_120.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_120.Facts  (ctx : sequential_generatedRoot_120Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_120.Facts_0 ctx outputs

theorem sequential_generatedRoot_120.facts  (ctx : sequential_generatedRoot_120Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_120.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_120.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_120.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_122.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_122.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_122Ctx" => (sequential_generatedRoot_122.Ctx params i_0)

def sequential_generatedRoot_122.w_0_0  (ctx : sequential_generatedRoot_122Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_122.w_3_0  (ctx : sequential_generatedRoot_122Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_122.w_1_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  if (sequential_generatedRoot_122.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_122.w_2_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  0

def sequential_generatedRoot_122.w_4_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_122.w_5_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_122.w_6_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  1

def sequential_generatedRoot_122.w_7_0  (ctx : sequential_generatedRoot_122Ctx) : Bool :=
  decide ((sequential_generatedRoot_122.w_5_0 ctx) = (sequential_generatedRoot_122.w_6_0 ctx))

def sequential_generatedRoot_122.w_8_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  if (sequential_generatedRoot_122.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_122.w_9_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  ((sequential_generatedRoot_122.w_1_0 ctx) + (sequential_generatedRoot_122.w_8_0 ctx))

def sequential_generatedRoot_122.w_10_0  (ctx : sequential_generatedRoot_122Ctx) : Int :=
  2

def sequential_generatedRoot_122.w_11_0  (ctx : sequential_generatedRoot_122Ctx) : Bool :=
  decide ((sequential_generatedRoot_122.w_9_0 ctx) = (sequential_generatedRoot_122.w_10_0 ctx))

def sequential_generatedRoot_122.outs  (ctx : sequential_generatedRoot_122Ctx) : Bool :=
  (sequential_generatedRoot_122.w_11_0 ctx)

structure sequential_generatedRoot_122.Facts_0  (ctx : sequential_generatedRoot_122Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_166 params i_0 i (current, ((sequential_generatedRoot_122.w_3_0 ctx), ((sequential_generatedRoot_122.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_122.w_2_0 ctx) (sequential_generatedRoot_122.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_122.outs ctx

theorem sequential_generatedRoot_122.facts_0  {ctx : sequential_generatedRoot_122Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_122.constraints_0 params i_0 (sequential_generatedRoot_122.w_3_0 ctx) (sequential_generatedRoot_122.w_2_0 ctx) (sequential_generatedRoot_122.w_4_0 ctx) (sequential_generatedRoot_122.w_5_0 ctx) (sequential_generatedRoot_122.w_11_0 ctx) outputs) :
    sequential_generatedRoot_122.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_122.Facts  (ctx : sequential_generatedRoot_122Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_122.Facts_0 ctx outputs

theorem sequential_generatedRoot_122.facts  (ctx : sequential_generatedRoot_122Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_122.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_122.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_122.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_124.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_124.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_124Ctx" => (sequential_generatedRoot_124.Ctx params i_0)

def sequential_generatedRoot_124.w_0_0  (ctx : sequential_generatedRoot_124Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_124.w_3_0  (ctx : sequential_generatedRoot_124Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_124.w_1_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  if (sequential_generatedRoot_124.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_124.w_2_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  0

def sequential_generatedRoot_124.w_4_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_124.w_5_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_124.w_6_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  1

def sequential_generatedRoot_124.w_7_0  (ctx : sequential_generatedRoot_124Ctx) : Bool :=
  decide ((sequential_generatedRoot_124.w_5_0 ctx) = (sequential_generatedRoot_124.w_6_0 ctx))

def sequential_generatedRoot_124.w_8_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  if (sequential_generatedRoot_124.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_124.w_9_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  ((sequential_generatedRoot_124.w_1_0 ctx) + (sequential_generatedRoot_124.w_8_0 ctx))

def sequential_generatedRoot_124.w_10_0  (ctx : sequential_generatedRoot_124Ctx) : Int :=
  2

def sequential_generatedRoot_124.w_11_0  (ctx : sequential_generatedRoot_124Ctx) : Bool :=
  decide ((sequential_generatedRoot_124.w_9_0 ctx) = (sequential_generatedRoot_124.w_10_0 ctx))

def sequential_generatedRoot_124.outs  (ctx : sequential_generatedRoot_124Ctx) : Bool :=
  (sequential_generatedRoot_124.w_11_0 ctx)

structure sequential_generatedRoot_124.Facts_0  (ctx : sequential_generatedRoot_124Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_167 params i_0 i (current, ((sequential_generatedRoot_124.w_3_0 ctx), ((sequential_generatedRoot_124.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_124.w_2_0 ctx) (sequential_generatedRoot_124.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_124.outs ctx

theorem sequential_generatedRoot_124.facts_0  {ctx : sequential_generatedRoot_124Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_124.constraints_0 params i_0 (sequential_generatedRoot_124.w_3_0 ctx) (sequential_generatedRoot_124.w_2_0 ctx) (sequential_generatedRoot_124.w_4_0 ctx) (sequential_generatedRoot_124.w_5_0 ctx) (sequential_generatedRoot_124.w_11_0 ctx) outputs) :
    sequential_generatedRoot_124.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_124.Facts  (ctx : sequential_generatedRoot_124Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_124.Facts_0 ctx outputs

theorem sequential_generatedRoot_124.facts  (ctx : sequential_generatedRoot_124Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_124.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_124.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_124.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_126.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_126.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_126Ctx" => (sequential_generatedRoot_126.Ctx params i_0)

def sequential_generatedRoot_126.w_0_0  (ctx : sequential_generatedRoot_126Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_126.w_3_0  (ctx : sequential_generatedRoot_126Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_126.w_1_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  if (sequential_generatedRoot_126.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_126.w_2_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  0

def sequential_generatedRoot_126.w_4_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_126.w_5_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_126.w_6_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  1

def sequential_generatedRoot_126.w_7_0  (ctx : sequential_generatedRoot_126Ctx) : Bool :=
  decide ((sequential_generatedRoot_126.w_5_0 ctx) = (sequential_generatedRoot_126.w_6_0 ctx))

def sequential_generatedRoot_126.w_8_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  if (sequential_generatedRoot_126.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_126.w_9_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  ((sequential_generatedRoot_126.w_1_0 ctx) + (sequential_generatedRoot_126.w_8_0 ctx))

def sequential_generatedRoot_126.w_10_0  (ctx : sequential_generatedRoot_126Ctx) : Int :=
  2

def sequential_generatedRoot_126.w_11_0  (ctx : sequential_generatedRoot_126Ctx) : Bool :=
  decide ((sequential_generatedRoot_126.w_9_0 ctx) = (sequential_generatedRoot_126.w_10_0 ctx))

def sequential_generatedRoot_126.outs  (ctx : sequential_generatedRoot_126Ctx) : Bool :=
  (sequential_generatedRoot_126.w_11_0 ctx)

structure sequential_generatedRoot_126.Facts_0  (ctx : sequential_generatedRoot_126Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_168 params i_0 i (current, ((sequential_generatedRoot_126.w_3_0 ctx), ((sequential_generatedRoot_126.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_126.w_2_0 ctx) (sequential_generatedRoot_126.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_126.outs ctx

theorem sequential_generatedRoot_126.facts_0  {ctx : sequential_generatedRoot_126Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_126.constraints_0 params i_0 (sequential_generatedRoot_126.w_3_0 ctx) (sequential_generatedRoot_126.w_2_0 ctx) (sequential_generatedRoot_126.w_4_0 ctx) (sequential_generatedRoot_126.w_5_0 ctx) (sequential_generatedRoot_126.w_11_0 ctx) outputs) :
    sequential_generatedRoot_126.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_126.Facts  (ctx : sequential_generatedRoot_126Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_126.Facts_0 ctx outputs

theorem sequential_generatedRoot_126.facts  (ctx : sequential_generatedRoot_126Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_126.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_126.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_126.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_128.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_128.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_128Ctx" => (sequential_generatedRoot_128.Ctx params i_0)

def sequential_generatedRoot_128.w_0_0  (ctx : sequential_generatedRoot_128Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_128.w_3_0  (ctx : sequential_generatedRoot_128Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_128.w_1_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  if (sequential_generatedRoot_128.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_128.w_2_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  0

def sequential_generatedRoot_128.w_4_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_128.w_5_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_128.w_6_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  1

def sequential_generatedRoot_128.w_7_0  (ctx : sequential_generatedRoot_128Ctx) : Bool :=
  decide ((sequential_generatedRoot_128.w_5_0 ctx) = (sequential_generatedRoot_128.w_6_0 ctx))

def sequential_generatedRoot_128.w_8_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  if (sequential_generatedRoot_128.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_128.w_9_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  ((sequential_generatedRoot_128.w_1_0 ctx) + (sequential_generatedRoot_128.w_8_0 ctx))

def sequential_generatedRoot_128.w_10_0  (ctx : sequential_generatedRoot_128Ctx) : Int :=
  2

def sequential_generatedRoot_128.w_11_0  (ctx : sequential_generatedRoot_128Ctx) : Bool :=
  decide ((sequential_generatedRoot_128.w_9_0 ctx) = (sequential_generatedRoot_128.w_10_0 ctx))

def sequential_generatedRoot_128.outs  (ctx : sequential_generatedRoot_128Ctx) : Bool :=
  (sequential_generatedRoot_128.w_11_0 ctx)

structure sequential_generatedRoot_128.Facts_0  (ctx : sequential_generatedRoot_128Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_169 params i_0 i (current, ((sequential_generatedRoot_128.w_3_0 ctx), ((sequential_generatedRoot_128.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_128.w_2_0 ctx) (sequential_generatedRoot_128.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_128.outs ctx

theorem sequential_generatedRoot_128.facts_0  {ctx : sequential_generatedRoot_128Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_128.constraints_0 params i_0 (sequential_generatedRoot_128.w_3_0 ctx) (sequential_generatedRoot_128.w_2_0 ctx) (sequential_generatedRoot_128.w_4_0 ctx) (sequential_generatedRoot_128.w_5_0 ctx) (sequential_generatedRoot_128.w_11_0 ctx) outputs) :
    sequential_generatedRoot_128.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_128.Facts  (ctx : sequential_generatedRoot_128Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_128.Facts_0 ctx outputs

theorem sequential_generatedRoot_128.facts  (ctx : sequential_generatedRoot_128Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_128.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_128.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_128.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_130.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_130.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_130Ctx" => (sequential_generatedRoot_130.Ctx params i_0)

def sequential_generatedRoot_130.w_0_0  (ctx : sequential_generatedRoot_130Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_130.w_3_0  (ctx : sequential_generatedRoot_130Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_130.w_1_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  if (sequential_generatedRoot_130.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_130.w_2_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  0

def sequential_generatedRoot_130.w_4_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_130.w_5_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_130.w_6_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  1

def sequential_generatedRoot_130.w_7_0  (ctx : sequential_generatedRoot_130Ctx) : Bool :=
  decide ((sequential_generatedRoot_130.w_5_0 ctx) = (sequential_generatedRoot_130.w_6_0 ctx))

def sequential_generatedRoot_130.w_8_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  if (sequential_generatedRoot_130.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_130.w_9_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  ((sequential_generatedRoot_130.w_1_0 ctx) + (sequential_generatedRoot_130.w_8_0 ctx))

def sequential_generatedRoot_130.w_10_0  (ctx : sequential_generatedRoot_130Ctx) : Int :=
  2

def sequential_generatedRoot_130.w_11_0  (ctx : sequential_generatedRoot_130Ctx) : Bool :=
  decide ((sequential_generatedRoot_130.w_9_0 ctx) = (sequential_generatedRoot_130.w_10_0 ctx))

def sequential_generatedRoot_130.outs  (ctx : sequential_generatedRoot_130Ctx) : Bool :=
  (sequential_generatedRoot_130.w_11_0 ctx)

structure sequential_generatedRoot_130.Facts_0  (ctx : sequential_generatedRoot_130Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_170 params i_0 i (current, ((sequential_generatedRoot_130.w_3_0 ctx), ((sequential_generatedRoot_130.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_130.w_2_0 ctx) (sequential_generatedRoot_130.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_130.outs ctx

theorem sequential_generatedRoot_130.facts_0  {ctx : sequential_generatedRoot_130Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_130.constraints_0 params i_0 (sequential_generatedRoot_130.w_3_0 ctx) (sequential_generatedRoot_130.w_2_0 ctx) (sequential_generatedRoot_130.w_4_0 ctx) (sequential_generatedRoot_130.w_5_0 ctx) (sequential_generatedRoot_130.w_11_0 ctx) outputs) :
    sequential_generatedRoot_130.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_130.Facts  (ctx : sequential_generatedRoot_130Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_130.Facts_0 ctx outputs

theorem sequential_generatedRoot_130.facts  (ctx : sequential_generatedRoot_130Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_130.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_130.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_130.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_132.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_132.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_132Ctx" => (sequential_generatedRoot_132.Ctx params i_0)

def sequential_generatedRoot_132.w_0_0  (ctx : sequential_generatedRoot_132Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_132.w_3_0  (ctx : sequential_generatedRoot_132Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_132.w_1_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  if (sequential_generatedRoot_132.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_132.w_2_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  0

def sequential_generatedRoot_132.w_4_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_132.w_5_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_132.w_6_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  1

def sequential_generatedRoot_132.w_7_0  (ctx : sequential_generatedRoot_132Ctx) : Bool :=
  decide ((sequential_generatedRoot_132.w_5_0 ctx) = (sequential_generatedRoot_132.w_6_0 ctx))

def sequential_generatedRoot_132.w_8_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  if (sequential_generatedRoot_132.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_132.w_9_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  ((sequential_generatedRoot_132.w_1_0 ctx) + (sequential_generatedRoot_132.w_8_0 ctx))

def sequential_generatedRoot_132.w_10_0  (ctx : sequential_generatedRoot_132Ctx) : Int :=
  2

def sequential_generatedRoot_132.w_11_0  (ctx : sequential_generatedRoot_132Ctx) : Bool :=
  decide ((sequential_generatedRoot_132.w_9_0 ctx) = (sequential_generatedRoot_132.w_10_0 ctx))

def sequential_generatedRoot_132.outs  (ctx : sequential_generatedRoot_132Ctx) : Bool :=
  (sequential_generatedRoot_132.w_11_0 ctx)

structure sequential_generatedRoot_132.Facts_0  (ctx : sequential_generatedRoot_132Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_171 params i_0 i (current, ((sequential_generatedRoot_132.w_3_0 ctx), ((sequential_generatedRoot_132.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_132.w_2_0 ctx) (sequential_generatedRoot_132.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_132.outs ctx

theorem sequential_generatedRoot_132.facts_0  {ctx : sequential_generatedRoot_132Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_132.constraints_0 params i_0 (sequential_generatedRoot_132.w_3_0 ctx) (sequential_generatedRoot_132.w_2_0 ctx) (sequential_generatedRoot_132.w_4_0 ctx) (sequential_generatedRoot_132.w_5_0 ctx) (sequential_generatedRoot_132.w_11_0 ctx) outputs) :
    sequential_generatedRoot_132.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_132.Facts  (ctx : sequential_generatedRoot_132Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_132.Facts_0 ctx outputs

theorem sequential_generatedRoot_132.facts  (ctx : sequential_generatedRoot_132Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_132.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_132.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_132.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_134.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_134.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_134Ctx" => (sequential_generatedRoot_134.Ctx params i_0)

def sequential_generatedRoot_134.w_0_0  (ctx : sequential_generatedRoot_134Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_134.w_3_0  (ctx : sequential_generatedRoot_134Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_134.w_1_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  if (sequential_generatedRoot_134.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_134.w_2_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  0

def sequential_generatedRoot_134.w_4_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_134.w_5_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_134.w_6_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  1

def sequential_generatedRoot_134.w_7_0  (ctx : sequential_generatedRoot_134Ctx) : Bool :=
  decide ((sequential_generatedRoot_134.w_5_0 ctx) = (sequential_generatedRoot_134.w_6_0 ctx))

def sequential_generatedRoot_134.w_8_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  if (sequential_generatedRoot_134.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_134.w_9_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  ((sequential_generatedRoot_134.w_1_0 ctx) + (sequential_generatedRoot_134.w_8_0 ctx))

def sequential_generatedRoot_134.w_10_0  (ctx : sequential_generatedRoot_134Ctx) : Int :=
  2

def sequential_generatedRoot_134.w_11_0  (ctx : sequential_generatedRoot_134Ctx) : Bool :=
  decide ((sequential_generatedRoot_134.w_9_0 ctx) = (sequential_generatedRoot_134.w_10_0 ctx))

def sequential_generatedRoot_134.outs  (ctx : sequential_generatedRoot_134Ctx) : Bool :=
  (sequential_generatedRoot_134.w_11_0 ctx)

structure sequential_generatedRoot_134.Facts_0  (ctx : sequential_generatedRoot_134Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_172 params i_0 i (current, ((sequential_generatedRoot_134.w_3_0 ctx), ((sequential_generatedRoot_134.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_134.w_2_0 ctx) (sequential_generatedRoot_134.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_134.outs ctx

theorem sequential_generatedRoot_134.facts_0  {ctx : sequential_generatedRoot_134Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_134.constraints_0 params i_0 (sequential_generatedRoot_134.w_3_0 ctx) (sequential_generatedRoot_134.w_2_0 ctx) (sequential_generatedRoot_134.w_4_0 ctx) (sequential_generatedRoot_134.w_5_0 ctx) (sequential_generatedRoot_134.w_11_0 ctx) outputs) :
    sequential_generatedRoot_134.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_134.Facts  (ctx : sequential_generatedRoot_134Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_134.Facts_0 ctx outputs

theorem sequential_generatedRoot_134.facts  (ctx : sequential_generatedRoot_134Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_134.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_134.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_134.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_136.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_136Ctx" => (sequential_generatedRoot_136.Ctx params i_0)

def sequential_generatedRoot_136.w_0_0  (ctx : sequential_generatedRoot_136Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_136.w_3_0  (ctx : sequential_generatedRoot_136Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_136.w_1_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  if (sequential_generatedRoot_136.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_136.w_2_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  0

def sequential_generatedRoot_136.w_4_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_136.w_5_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_136.w_6_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  1

def sequential_generatedRoot_136.w_7_0  (ctx : sequential_generatedRoot_136Ctx) : Bool :=
  decide ((sequential_generatedRoot_136.w_5_0 ctx) = (sequential_generatedRoot_136.w_6_0 ctx))

def sequential_generatedRoot_136.w_8_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  if (sequential_generatedRoot_136.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_136.w_9_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  ((sequential_generatedRoot_136.w_1_0 ctx) + (sequential_generatedRoot_136.w_8_0 ctx))

def sequential_generatedRoot_136.w_10_0  (ctx : sequential_generatedRoot_136Ctx) : Int :=
  2

def sequential_generatedRoot_136.w_11_0  (ctx : sequential_generatedRoot_136Ctx) : Bool :=
  decide ((sequential_generatedRoot_136.w_9_0 ctx) = (sequential_generatedRoot_136.w_10_0 ctx))

def sequential_generatedRoot_136.outs  (ctx : sequential_generatedRoot_136Ctx) : Bool :=
  (sequential_generatedRoot_136.w_11_0 ctx)

structure sequential_generatedRoot_136.Facts_0  (ctx : sequential_generatedRoot_136Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_173 params i_0 i (current, ((sequential_generatedRoot_136.w_3_0 ctx), ((sequential_generatedRoot_136.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_136.w_2_0 ctx) (sequential_generatedRoot_136.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_136.outs ctx

theorem sequential_generatedRoot_136.facts_0  {ctx : sequential_generatedRoot_136Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_136.constraints_0 params i_0 (sequential_generatedRoot_136.w_3_0 ctx) (sequential_generatedRoot_136.w_2_0 ctx) (sequential_generatedRoot_136.w_4_0 ctx) (sequential_generatedRoot_136.w_5_0 ctx) (sequential_generatedRoot_136.w_11_0 ctx) outputs) :
    sequential_generatedRoot_136.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_136.Facts  (ctx : sequential_generatedRoot_136Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_136.Facts_0 ctx outputs

theorem sequential_generatedRoot_136.facts  (ctx : sequential_generatedRoot_136Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_136.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_136.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_136.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_138.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_138.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_138Ctx" => (sequential_generatedRoot_138.Ctx params i_0)

def sequential_generatedRoot_138.w_0_0  (ctx : sequential_generatedRoot_138Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_138.w_3_0  (ctx : sequential_generatedRoot_138Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_138.w_1_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  if (sequential_generatedRoot_138.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_138.w_2_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  0

def sequential_generatedRoot_138.w_4_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_138.w_5_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_138.w_6_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  1

def sequential_generatedRoot_138.w_7_0  (ctx : sequential_generatedRoot_138Ctx) : Bool :=
  decide ((sequential_generatedRoot_138.w_5_0 ctx) = (sequential_generatedRoot_138.w_6_0 ctx))

def sequential_generatedRoot_138.w_8_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  if (sequential_generatedRoot_138.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_138.w_9_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  ((sequential_generatedRoot_138.w_1_0 ctx) + (sequential_generatedRoot_138.w_8_0 ctx))

def sequential_generatedRoot_138.w_10_0  (ctx : sequential_generatedRoot_138Ctx) : Int :=
  2

def sequential_generatedRoot_138.w_11_0  (ctx : sequential_generatedRoot_138Ctx) : Bool :=
  decide ((sequential_generatedRoot_138.w_9_0 ctx) = (sequential_generatedRoot_138.w_10_0 ctx))

def sequential_generatedRoot_138.outs  (ctx : sequential_generatedRoot_138Ctx) : Bool :=
  (sequential_generatedRoot_138.w_11_0 ctx)

structure sequential_generatedRoot_138.Facts_0  (ctx : sequential_generatedRoot_138Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_174 params i_0 i (current, ((sequential_generatedRoot_138.w_3_0 ctx), ((sequential_generatedRoot_138.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_138.w_2_0 ctx) (sequential_generatedRoot_138.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_138.outs ctx

theorem sequential_generatedRoot_138.facts_0  {ctx : sequential_generatedRoot_138Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_138.constraints_0 params i_0 (sequential_generatedRoot_138.w_3_0 ctx) (sequential_generatedRoot_138.w_2_0 ctx) (sequential_generatedRoot_138.w_4_0 ctx) (sequential_generatedRoot_138.w_5_0 ctx) (sequential_generatedRoot_138.w_11_0 ctx) outputs) :
    sequential_generatedRoot_138.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_138.Facts  (ctx : sequential_generatedRoot_138Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_138.Facts_0 ctx outputs

theorem sequential_generatedRoot_138.facts  (ctx : sequential_generatedRoot_138Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_138.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_138.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_138.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_140.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 318898285953889075200 → Int) × Unit
  witness : sequential_generatedRoot_140.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_140Ctx" => (sequential_generatedRoot_140.Ctx params i_0)

def sequential_generatedRoot_140.w_0_0  (ctx : sequential_generatedRoot_140Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_140.w_3_0  (ctx : sequential_generatedRoot_140Ctx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_140.w_1_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  if (sequential_generatedRoot_140.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_140.w_2_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  0

def sequential_generatedRoot_140.w_4_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_140.w_5_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_140.w_6_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  1

def sequential_generatedRoot_140.w_7_0  (ctx : sequential_generatedRoot_140Ctx) : Bool :=
  decide ((sequential_generatedRoot_140.w_5_0 ctx) = (sequential_generatedRoot_140.w_6_0 ctx))

def sequential_generatedRoot_140.w_8_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  if (sequential_generatedRoot_140.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_140.w_9_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  ((sequential_generatedRoot_140.w_1_0 ctx) + (sequential_generatedRoot_140.w_8_0 ctx))

def sequential_generatedRoot_140.w_10_0  (ctx : sequential_generatedRoot_140Ctx) : Int :=
  2

def sequential_generatedRoot_140.w_11_0  (ctx : sequential_generatedRoot_140Ctx) : Bool :=
  decide ((sequential_generatedRoot_140.w_9_0 ctx) = (sequential_generatedRoot_140.w_10_0 ctx))

def sequential_generatedRoot_140.outs  (ctx : sequential_generatedRoot_140Ctx) : Bool :=
  (sequential_generatedRoot_140.w_11_0 ctx)

structure sequential_generatedRoot_140.Facts_0  (ctx : sequential_generatedRoot_140Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_175 params i_0 i (current, ((sequential_generatedRoot_140.w_3_0 ctx), ((sequential_generatedRoot_140.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_140.w_2_0 ctx) (sequential_generatedRoot_140.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_140.outs ctx

theorem sequential_generatedRoot_140.facts_0  {ctx : sequential_generatedRoot_140Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_140.constraints_0 params i_0 (sequential_generatedRoot_140.w_3_0 ctx) (sequential_generatedRoot_140.w_2_0 ctx) (sequential_generatedRoot_140.w_4_0 ctx) (sequential_generatedRoot_140.w_5_0 ctx) (sequential_generatedRoot_140.w_11_0 ctx) outputs) :
    sequential_generatedRoot_140.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_140.Facts  (ctx : sequential_generatedRoot_140Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_140.Facts_0 ctx outputs

theorem sequential_generatedRoot_140.facts  (ctx : sequential_generatedRoot_140Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_140.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_140.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_140.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_142.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_142.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_142Ctx" => (sequential_generatedRoot_142.Ctx params i_0)

def sequential_generatedRoot_142.w_0_0  (ctx : sequential_generatedRoot_142Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_142.w_3_0  (ctx : sequential_generatedRoot_142Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_142.w_1_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  if (sequential_generatedRoot_142.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_142.w_2_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  0

def sequential_generatedRoot_142.w_4_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_142.w_5_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_142.w_6_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  1

def sequential_generatedRoot_142.w_7_0  (ctx : sequential_generatedRoot_142Ctx) : Bool :=
  decide ((sequential_generatedRoot_142.w_5_0 ctx) = (sequential_generatedRoot_142.w_6_0 ctx))

def sequential_generatedRoot_142.w_8_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  if (sequential_generatedRoot_142.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_142.w_9_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  ((sequential_generatedRoot_142.w_1_0 ctx) + (sequential_generatedRoot_142.w_8_0 ctx))

def sequential_generatedRoot_142.w_10_0  (ctx : sequential_generatedRoot_142Ctx) : Int :=
  2

def sequential_generatedRoot_142.w_11_0  (ctx : sequential_generatedRoot_142Ctx) : Bool :=
  decide ((sequential_generatedRoot_142.w_9_0 ctx) = (sequential_generatedRoot_142.w_10_0 ctx))

def sequential_generatedRoot_142.outs  (ctx : sequential_generatedRoot_142Ctx) : Bool :=
  (sequential_generatedRoot_142.w_11_0 ctx)

structure sequential_generatedRoot_142.Facts_0  (ctx : sequential_generatedRoot_142Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_176 params i_0 i (current, ((sequential_generatedRoot_142.w_3_0 ctx), ((sequential_generatedRoot_142.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_142.w_2_0 ctx) (sequential_generatedRoot_142.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_142.outs ctx

theorem sequential_generatedRoot_142.facts_0  {ctx : sequential_generatedRoot_142Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_142.constraints_0 params i_0 (sequential_generatedRoot_142.w_3_0 ctx) (sequential_generatedRoot_142.w_2_0 ctx) (sequential_generatedRoot_142.w_4_0 ctx) (sequential_generatedRoot_142.w_5_0 ctx) (sequential_generatedRoot_142.w_11_0 ctx) outputs) :
    sequential_generatedRoot_142.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_142.Facts  (ctx : sequential_generatedRoot_142Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_142.Facts_0 ctx outputs

theorem sequential_generatedRoot_142.facts  (ctx : sequential_generatedRoot_142Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_142.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_142.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_142.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_144.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_144.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_144Ctx" => (sequential_generatedRoot_144.Ctx params i_0)

def sequential_generatedRoot_144.w_0_0  (ctx : sequential_generatedRoot_144Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_144.w_3_0  (ctx : sequential_generatedRoot_144Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_144.w_1_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  if (sequential_generatedRoot_144.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_144.w_2_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  0

def sequential_generatedRoot_144.w_4_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_144.w_5_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_144.w_6_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  1

def sequential_generatedRoot_144.w_7_0  (ctx : sequential_generatedRoot_144Ctx) : Bool :=
  decide ((sequential_generatedRoot_144.w_5_0 ctx) = (sequential_generatedRoot_144.w_6_0 ctx))

def sequential_generatedRoot_144.w_8_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  if (sequential_generatedRoot_144.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_144.w_9_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  ((sequential_generatedRoot_144.w_1_0 ctx) + (sequential_generatedRoot_144.w_8_0 ctx))

def sequential_generatedRoot_144.w_10_0  (ctx : sequential_generatedRoot_144Ctx) : Int :=
  2

def sequential_generatedRoot_144.w_11_0  (ctx : sequential_generatedRoot_144Ctx) : Bool :=
  decide ((sequential_generatedRoot_144.w_9_0 ctx) = (sequential_generatedRoot_144.w_10_0 ctx))

def sequential_generatedRoot_144.outs  (ctx : sequential_generatedRoot_144Ctx) : Bool :=
  (sequential_generatedRoot_144.w_11_0 ctx)

structure sequential_generatedRoot_144.Facts_0  (ctx : sequential_generatedRoot_144Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_177 params i_0 i (current, ((sequential_generatedRoot_144.w_3_0 ctx), ((sequential_generatedRoot_144.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_144.w_2_0 ctx) (sequential_generatedRoot_144.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_144.outs ctx

theorem sequential_generatedRoot_144.facts_0  {ctx : sequential_generatedRoot_144Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_144.constraints_0 params i_0 (sequential_generatedRoot_144.w_3_0 ctx) (sequential_generatedRoot_144.w_2_0 ctx) (sequential_generatedRoot_144.w_4_0 ctx) (sequential_generatedRoot_144.w_5_0 ctx) (sequential_generatedRoot_144.w_11_0 ctx) outputs) :
    sequential_generatedRoot_144.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_144.Facts  (ctx : sequential_generatedRoot_144Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_144.Facts_0 ctx outputs

theorem sequential_generatedRoot_144.facts  (ctx : sequential_generatedRoot_144Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_144.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_144.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_144.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_146.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_146.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_146Ctx" => (sequential_generatedRoot_146.Ctx params i_0)

def sequential_generatedRoot_146.w_0_0  (ctx : sequential_generatedRoot_146Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_146.w_3_0  (ctx : sequential_generatedRoot_146Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_146.w_1_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  if (sequential_generatedRoot_146.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_146.w_2_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  0

def sequential_generatedRoot_146.w_4_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_146.w_5_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_146.w_6_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  1

def sequential_generatedRoot_146.w_7_0  (ctx : sequential_generatedRoot_146Ctx) : Bool :=
  decide ((sequential_generatedRoot_146.w_5_0 ctx) = (sequential_generatedRoot_146.w_6_0 ctx))

def sequential_generatedRoot_146.w_8_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  if (sequential_generatedRoot_146.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_146.w_9_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  ((sequential_generatedRoot_146.w_1_0 ctx) + (sequential_generatedRoot_146.w_8_0 ctx))

def sequential_generatedRoot_146.w_10_0  (ctx : sequential_generatedRoot_146Ctx) : Int :=
  2

def sequential_generatedRoot_146.w_11_0  (ctx : sequential_generatedRoot_146Ctx) : Bool :=
  decide ((sequential_generatedRoot_146.w_9_0 ctx) = (sequential_generatedRoot_146.w_10_0 ctx))

def sequential_generatedRoot_146.outs  (ctx : sequential_generatedRoot_146Ctx) : Bool :=
  (sequential_generatedRoot_146.w_11_0 ctx)

structure sequential_generatedRoot_146.Facts_0  (ctx : sequential_generatedRoot_146Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_178 params i_0 i (current, ((sequential_generatedRoot_146.w_3_0 ctx), ((sequential_generatedRoot_146.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_146.w_2_0 ctx) (sequential_generatedRoot_146.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_146.outs ctx

theorem sequential_generatedRoot_146.facts_0  {ctx : sequential_generatedRoot_146Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_146.constraints_0 params i_0 (sequential_generatedRoot_146.w_3_0 ctx) (sequential_generatedRoot_146.w_2_0 ctx) (sequential_generatedRoot_146.w_4_0 ctx) (sequential_generatedRoot_146.w_5_0 ctx) (sequential_generatedRoot_146.w_11_0 ctx) outputs) :
    sequential_generatedRoot_146.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_146.Facts  (ctx : sequential_generatedRoot_146Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_146.Facts_0 ctx outputs

theorem sequential_generatedRoot_146.facts  (ctx : sequential_generatedRoot_146Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_146.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_146.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_146.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_148.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_148.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_148Ctx" => (sequential_generatedRoot_148.Ctx params i_0)

def sequential_generatedRoot_148.w_0_0  (ctx : sequential_generatedRoot_148Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_148.w_3_0  (ctx : sequential_generatedRoot_148Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_148.w_1_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  if (sequential_generatedRoot_148.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_148.w_2_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  0

def sequential_generatedRoot_148.w_4_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_148.w_5_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_148.w_6_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  1

def sequential_generatedRoot_148.w_7_0  (ctx : sequential_generatedRoot_148Ctx) : Bool :=
  decide ((sequential_generatedRoot_148.w_5_0 ctx) = (sequential_generatedRoot_148.w_6_0 ctx))

def sequential_generatedRoot_148.w_8_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  if (sequential_generatedRoot_148.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_148.w_9_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  ((sequential_generatedRoot_148.w_1_0 ctx) + (sequential_generatedRoot_148.w_8_0 ctx))

def sequential_generatedRoot_148.w_10_0  (ctx : sequential_generatedRoot_148Ctx) : Int :=
  2

def sequential_generatedRoot_148.w_11_0  (ctx : sequential_generatedRoot_148Ctx) : Bool :=
  decide ((sequential_generatedRoot_148.w_9_0 ctx) = (sequential_generatedRoot_148.w_10_0 ctx))

def sequential_generatedRoot_148.outs  (ctx : sequential_generatedRoot_148Ctx) : Bool :=
  (sequential_generatedRoot_148.w_11_0 ctx)

structure sequential_generatedRoot_148.Facts_0  (ctx : sequential_generatedRoot_148Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_179 params i_0 i (current, ((sequential_generatedRoot_148.w_3_0 ctx), ((sequential_generatedRoot_148.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_148.w_2_0 ctx) (sequential_generatedRoot_148.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_148.outs ctx

theorem sequential_generatedRoot_148.facts_0  {ctx : sequential_generatedRoot_148Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_148.constraints_0 params i_0 (sequential_generatedRoot_148.w_3_0 ctx) (sequential_generatedRoot_148.w_2_0 ctx) (sequential_generatedRoot_148.w_4_0 ctx) (sequential_generatedRoot_148.w_5_0 ctx) (sequential_generatedRoot_148.w_11_0 ctx) outputs) :
    sequential_generatedRoot_148.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_148.Facts  (ctx : sequential_generatedRoot_148Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_148.Facts_0 ctx outputs

theorem sequential_generatedRoot_148.facts  (ctx : sequential_generatedRoot_148Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_148.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_148.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_148.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_150.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_150.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_150Ctx" => (sequential_generatedRoot_150.Ctx params i_0)

def sequential_generatedRoot_150.w_0_0  (ctx : sequential_generatedRoot_150Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_150.w_3_0  (ctx : sequential_generatedRoot_150Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_150.w_1_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  if (sequential_generatedRoot_150.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_150.w_2_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  0

def sequential_generatedRoot_150.w_4_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_150.w_5_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_150.w_6_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  1

def sequential_generatedRoot_150.w_7_0  (ctx : sequential_generatedRoot_150Ctx) : Bool :=
  decide ((sequential_generatedRoot_150.w_5_0 ctx) = (sequential_generatedRoot_150.w_6_0 ctx))

def sequential_generatedRoot_150.w_8_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  if (sequential_generatedRoot_150.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_150.w_9_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  ((sequential_generatedRoot_150.w_1_0 ctx) + (sequential_generatedRoot_150.w_8_0 ctx))

def sequential_generatedRoot_150.w_10_0  (ctx : sequential_generatedRoot_150Ctx) : Int :=
  2

def sequential_generatedRoot_150.w_11_0  (ctx : sequential_generatedRoot_150Ctx) : Bool :=
  decide ((sequential_generatedRoot_150.w_9_0 ctx) = (sequential_generatedRoot_150.w_10_0 ctx))

def sequential_generatedRoot_150.outs  (ctx : sequential_generatedRoot_150Ctx) : Bool :=
  (sequential_generatedRoot_150.w_11_0 ctx)

structure sequential_generatedRoot_150.Facts_0  (ctx : sequential_generatedRoot_150Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_180 params i_0 i (current, ((sequential_generatedRoot_150.w_3_0 ctx), ((sequential_generatedRoot_150.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_150.w_2_0 ctx) (sequential_generatedRoot_150.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_150.outs ctx

theorem sequential_generatedRoot_150.facts_0  {ctx : sequential_generatedRoot_150Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_150.constraints_0 params i_0 (sequential_generatedRoot_150.w_3_0 ctx) (sequential_generatedRoot_150.w_2_0 ctx) (sequential_generatedRoot_150.w_4_0 ctx) (sequential_generatedRoot_150.w_5_0 ctx) (sequential_generatedRoot_150.w_11_0 ctx) outputs) :
    sequential_generatedRoot_150.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_150.Facts  (ctx : sequential_generatedRoot_150Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_150.Facts_0 ctx outputs

theorem sequential_generatedRoot_150.facts  (ctx : sequential_generatedRoot_150Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_150.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_150.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_150.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_152.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_152.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_152Ctx" => (sequential_generatedRoot_152.Ctx params i_0)

def sequential_generatedRoot_152.w_0_0  (ctx : sequential_generatedRoot_152Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_152.w_3_0  (ctx : sequential_generatedRoot_152Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_152.w_1_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  if (sequential_generatedRoot_152.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_152.w_2_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  0

def sequential_generatedRoot_152.w_4_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_152.w_5_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_152.w_6_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  1

def sequential_generatedRoot_152.w_7_0  (ctx : sequential_generatedRoot_152Ctx) : Bool :=
  decide ((sequential_generatedRoot_152.w_5_0 ctx) = (sequential_generatedRoot_152.w_6_0 ctx))

def sequential_generatedRoot_152.w_8_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  if (sequential_generatedRoot_152.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_152.w_9_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  ((sequential_generatedRoot_152.w_1_0 ctx) + (sequential_generatedRoot_152.w_8_0 ctx))

def sequential_generatedRoot_152.w_10_0  (ctx : sequential_generatedRoot_152Ctx) : Int :=
  2

def sequential_generatedRoot_152.w_11_0  (ctx : sequential_generatedRoot_152Ctx) : Bool :=
  decide ((sequential_generatedRoot_152.w_9_0 ctx) = (sequential_generatedRoot_152.w_10_0 ctx))

def sequential_generatedRoot_152.outs  (ctx : sequential_generatedRoot_152Ctx) : Bool :=
  (sequential_generatedRoot_152.w_11_0 ctx)

structure sequential_generatedRoot_152.Facts_0  (ctx : sequential_generatedRoot_152Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_181 params i_0 i (current, ((sequential_generatedRoot_152.w_3_0 ctx), ((sequential_generatedRoot_152.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_152.w_2_0 ctx) (sequential_generatedRoot_152.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_152.outs ctx

theorem sequential_generatedRoot_152.facts_0  {ctx : sequential_generatedRoot_152Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_152.constraints_0 params i_0 (sequential_generatedRoot_152.w_3_0 ctx) (sequential_generatedRoot_152.w_2_0 ctx) (sequential_generatedRoot_152.w_4_0 ctx) (sequential_generatedRoot_152.w_5_0 ctx) (sequential_generatedRoot_152.w_11_0 ctx) outputs) :
    sequential_generatedRoot_152.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_152.Facts  (ctx : sequential_generatedRoot_152Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_152.Facts_0 ctx outputs

theorem sequential_generatedRoot_152.facts  (ctx : sequential_generatedRoot_152Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_152.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_152.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_152.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_154.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_154.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_154Ctx" => (sequential_generatedRoot_154.Ctx params i_0)

def sequential_generatedRoot_154.w_0_0  (ctx : sequential_generatedRoot_154Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_154.w_3_0  (ctx : sequential_generatedRoot_154Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_154.w_1_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  if (sequential_generatedRoot_154.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_154.w_2_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  0

def sequential_generatedRoot_154.w_4_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_154.w_5_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_154.w_6_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  1

def sequential_generatedRoot_154.w_7_0  (ctx : sequential_generatedRoot_154Ctx) : Bool :=
  decide ((sequential_generatedRoot_154.w_5_0 ctx) = (sequential_generatedRoot_154.w_6_0 ctx))

def sequential_generatedRoot_154.w_8_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  if (sequential_generatedRoot_154.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_154.w_9_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  ((sequential_generatedRoot_154.w_1_0 ctx) + (sequential_generatedRoot_154.w_8_0 ctx))

def sequential_generatedRoot_154.w_10_0  (ctx : sequential_generatedRoot_154Ctx) : Int :=
  2

def sequential_generatedRoot_154.w_11_0  (ctx : sequential_generatedRoot_154Ctx) : Bool :=
  decide ((sequential_generatedRoot_154.w_9_0 ctx) = (sequential_generatedRoot_154.w_10_0 ctx))

def sequential_generatedRoot_154.outs  (ctx : sequential_generatedRoot_154Ctx) : Bool :=
  (sequential_generatedRoot_154.w_11_0 ctx)

structure sequential_generatedRoot_154.Facts_0  (ctx : sequential_generatedRoot_154Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_182 params i_0 i (current, ((sequential_generatedRoot_154.w_3_0 ctx), ((sequential_generatedRoot_154.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_154.w_2_0 ctx) (sequential_generatedRoot_154.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_154.outs ctx

theorem sequential_generatedRoot_154.facts_0  {ctx : sequential_generatedRoot_154Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_154.constraints_0 params i_0 (sequential_generatedRoot_154.w_3_0 ctx) (sequential_generatedRoot_154.w_2_0 ctx) (sequential_generatedRoot_154.w_4_0 ctx) (sequential_generatedRoot_154.w_5_0 ctx) (sequential_generatedRoot_154.w_11_0 ctx) outputs) :
    sequential_generatedRoot_154.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_154.Facts  (ctx : sequential_generatedRoot_154Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_154.Facts_0 ctx outputs

theorem sequential_generatedRoot_154.facts  (ctx : sequential_generatedRoot_154Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_154.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_154.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_154.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_156.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_156.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_156Ctx" => (sequential_generatedRoot_156.Ctx params i_0)

def sequential_generatedRoot_156.w_0_0  (ctx : sequential_generatedRoot_156Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_156.w_3_0  (ctx : sequential_generatedRoot_156Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_156.w_1_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  if (sequential_generatedRoot_156.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_156.w_2_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  0

def sequential_generatedRoot_156.w_4_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_156.w_5_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_156.w_6_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  1

def sequential_generatedRoot_156.w_7_0  (ctx : sequential_generatedRoot_156Ctx) : Bool :=
  decide ((sequential_generatedRoot_156.w_5_0 ctx) = (sequential_generatedRoot_156.w_6_0 ctx))

def sequential_generatedRoot_156.w_8_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  if (sequential_generatedRoot_156.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_156.w_9_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  ((sequential_generatedRoot_156.w_1_0 ctx) + (sequential_generatedRoot_156.w_8_0 ctx))

def sequential_generatedRoot_156.w_10_0  (ctx : sequential_generatedRoot_156Ctx) : Int :=
  2

def sequential_generatedRoot_156.w_11_0  (ctx : sequential_generatedRoot_156Ctx) : Bool :=
  decide ((sequential_generatedRoot_156.w_9_0 ctx) = (sequential_generatedRoot_156.w_10_0 ctx))

def sequential_generatedRoot_156.outs  (ctx : sequential_generatedRoot_156Ctx) : Bool :=
  (sequential_generatedRoot_156.w_11_0 ctx)

structure sequential_generatedRoot_156.Facts_0  (ctx : sequential_generatedRoot_156Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_183 params i_0 i (current, ((sequential_generatedRoot_156.w_3_0 ctx), ((sequential_generatedRoot_156.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_156.w_2_0 ctx) (sequential_generatedRoot_156.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_156.outs ctx

theorem sequential_generatedRoot_156.facts_0  {ctx : sequential_generatedRoot_156Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_156.constraints_0 params i_0 (sequential_generatedRoot_156.w_3_0 ctx) (sequential_generatedRoot_156.w_2_0 ctx) (sequential_generatedRoot_156.w_4_0 ctx) (sequential_generatedRoot_156.w_5_0 ctx) (sequential_generatedRoot_156.w_11_0 ctx) outputs) :
    sequential_generatedRoot_156.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_156.Facts  (ctx : sequential_generatedRoot_156Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_156.Facts_0 ctx outputs

theorem sequential_generatedRoot_156.facts  (ctx : sequential_generatedRoot_156Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_156.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_156.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_156.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_158.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_158.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_158Ctx" => (sequential_generatedRoot_158.Ctx params i_0)

def sequential_generatedRoot_158.w_0_0  (ctx : sequential_generatedRoot_158Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_158.w_3_0  (ctx : sequential_generatedRoot_158Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_158.w_1_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  if (sequential_generatedRoot_158.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_158.w_2_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  0

def sequential_generatedRoot_158.w_4_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_158.w_5_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_158.w_6_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  1

def sequential_generatedRoot_158.w_7_0  (ctx : sequential_generatedRoot_158Ctx) : Bool :=
  decide ((sequential_generatedRoot_158.w_5_0 ctx) = (sequential_generatedRoot_158.w_6_0 ctx))

def sequential_generatedRoot_158.w_8_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  if (sequential_generatedRoot_158.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_158.w_9_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  ((sequential_generatedRoot_158.w_1_0 ctx) + (sequential_generatedRoot_158.w_8_0 ctx))

def sequential_generatedRoot_158.w_10_0  (ctx : sequential_generatedRoot_158Ctx) : Int :=
  2

def sequential_generatedRoot_158.w_11_0  (ctx : sequential_generatedRoot_158Ctx) : Bool :=
  decide ((sequential_generatedRoot_158.w_9_0 ctx) = (sequential_generatedRoot_158.w_10_0 ctx))

def sequential_generatedRoot_158.outs  (ctx : sequential_generatedRoot_158Ctx) : Bool :=
  (sequential_generatedRoot_158.w_11_0 ctx)

structure sequential_generatedRoot_158.Facts_0  (ctx : sequential_generatedRoot_158Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_184 params i_0 i (current, ((sequential_generatedRoot_158.w_3_0 ctx), ((sequential_generatedRoot_158.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_158.w_2_0 ctx) (sequential_generatedRoot_158.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_158.outs ctx

theorem sequential_generatedRoot_158.facts_0  {ctx : sequential_generatedRoot_158Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_158.constraints_0 params i_0 (sequential_generatedRoot_158.w_3_0 ctx) (sequential_generatedRoot_158.w_2_0 ctx) (sequential_generatedRoot_158.w_4_0 ctx) (sequential_generatedRoot_158.w_5_0 ctx) (sequential_generatedRoot_158.w_11_0 ctx) outputs) :
    sequential_generatedRoot_158.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_158.Facts  (ctx : sequential_generatedRoot_158Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_158.Facts_0 ctx outputs

theorem sequential_generatedRoot_158.facts  (ctx : sequential_generatedRoot_158Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_158.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_158.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_158.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_160.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_160.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_160Ctx" => (sequential_generatedRoot_160.Ctx params i_0)

def sequential_generatedRoot_160.w_0_0  (ctx : sequential_generatedRoot_160Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_160.w_3_0  (ctx : sequential_generatedRoot_160Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_160.w_1_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  if (sequential_generatedRoot_160.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_160.w_2_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  0

def sequential_generatedRoot_160.w_4_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_160.w_5_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_160.w_6_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  1

def sequential_generatedRoot_160.w_7_0  (ctx : sequential_generatedRoot_160Ctx) : Bool :=
  decide ((sequential_generatedRoot_160.w_5_0 ctx) = (sequential_generatedRoot_160.w_6_0 ctx))

def sequential_generatedRoot_160.w_8_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  if (sequential_generatedRoot_160.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_160.w_9_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  ((sequential_generatedRoot_160.w_1_0 ctx) + (sequential_generatedRoot_160.w_8_0 ctx))

def sequential_generatedRoot_160.w_10_0  (ctx : sequential_generatedRoot_160Ctx) : Int :=
  2

def sequential_generatedRoot_160.w_11_0  (ctx : sequential_generatedRoot_160Ctx) : Bool :=
  decide ((sequential_generatedRoot_160.w_9_0 ctx) = (sequential_generatedRoot_160.w_10_0 ctx))

def sequential_generatedRoot_160.outs  (ctx : sequential_generatedRoot_160Ctx) : Bool :=
  (sequential_generatedRoot_160.w_11_0 ctx)

structure sequential_generatedRoot_160.Facts_0  (ctx : sequential_generatedRoot_160Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_185 params i_0 i (current, ((sequential_generatedRoot_160.w_3_0 ctx), ((sequential_generatedRoot_160.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_160.w_2_0 ctx) (sequential_generatedRoot_160.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_160.outs ctx

theorem sequential_generatedRoot_160.facts_0  {ctx : sequential_generatedRoot_160Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_160.constraints_0 params i_0 (sequential_generatedRoot_160.w_3_0 ctx) (sequential_generatedRoot_160.w_2_0 ctx) (sequential_generatedRoot_160.w_4_0 ctx) (sequential_generatedRoot_160.w_5_0 ctx) (sequential_generatedRoot_160.w_11_0 ctx) outputs) :
    sequential_generatedRoot_160.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_160.Facts  (ctx : sequential_generatedRoot_160Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_160.Facts_0 ctx outputs

theorem sequential_generatedRoot_160.facts  (ctx : sequential_generatedRoot_160Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_160.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_160.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_160.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_162.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_162.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_162Ctx" => (sequential_generatedRoot_162.Ctx params i_0)

def sequential_generatedRoot_162.w_0_0  (ctx : sequential_generatedRoot_162Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_162.w_3_0  (ctx : sequential_generatedRoot_162Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_162.w_1_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  if (sequential_generatedRoot_162.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_162.w_2_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  0

def sequential_generatedRoot_162.w_4_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_162.w_5_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_162.w_6_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  1

def sequential_generatedRoot_162.w_7_0  (ctx : sequential_generatedRoot_162Ctx) : Bool :=
  decide ((sequential_generatedRoot_162.w_5_0 ctx) = (sequential_generatedRoot_162.w_6_0 ctx))

def sequential_generatedRoot_162.w_8_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  if (sequential_generatedRoot_162.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_162.w_9_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  ((sequential_generatedRoot_162.w_1_0 ctx) + (sequential_generatedRoot_162.w_8_0 ctx))

def sequential_generatedRoot_162.w_10_0  (ctx : sequential_generatedRoot_162Ctx) : Int :=
  2

def sequential_generatedRoot_162.w_11_0  (ctx : sequential_generatedRoot_162Ctx) : Bool :=
  decide ((sequential_generatedRoot_162.w_9_0 ctx) = (sequential_generatedRoot_162.w_10_0 ctx))

def sequential_generatedRoot_162.outs  (ctx : sequential_generatedRoot_162Ctx) : Bool :=
  (sequential_generatedRoot_162.w_11_0 ctx)

structure sequential_generatedRoot_162.Facts_0  (ctx : sequential_generatedRoot_162Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_186 params i_0 i (current, ((sequential_generatedRoot_162.w_3_0 ctx), ((sequential_generatedRoot_162.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_162.w_2_0 ctx) (sequential_generatedRoot_162.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_162.outs ctx

theorem sequential_generatedRoot_162.facts_0  {ctx : sequential_generatedRoot_162Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_162.constraints_0 params i_0 (sequential_generatedRoot_162.w_3_0 ctx) (sequential_generatedRoot_162.w_2_0 ctx) (sequential_generatedRoot_162.w_4_0 ctx) (sequential_generatedRoot_162.w_5_0 ctx) (sequential_generatedRoot_162.w_11_0 ctx) outputs) :
    sequential_generatedRoot_162.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_162.Facts  (ctx : sequential_generatedRoot_162Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_162.Facts_0 ctx outputs

theorem sequential_generatedRoot_162.facts  (ctx : sequential_generatedRoot_162Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_162.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_162.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_162.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_164.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_164.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_164Ctx" => (sequential_generatedRoot_164.Ctx params i_0)

def sequential_generatedRoot_164.w_0_0  (ctx : sequential_generatedRoot_164Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_164.w_3_0  (ctx : sequential_generatedRoot_164Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_164.w_1_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  if (sequential_generatedRoot_164.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_164.w_2_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  0

def sequential_generatedRoot_164.w_4_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_164.w_5_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_164.w_6_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  1

def sequential_generatedRoot_164.w_7_0  (ctx : sequential_generatedRoot_164Ctx) : Bool :=
  decide ((sequential_generatedRoot_164.w_5_0 ctx) = (sequential_generatedRoot_164.w_6_0 ctx))

def sequential_generatedRoot_164.w_8_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  if (sequential_generatedRoot_164.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_164.w_9_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  ((sequential_generatedRoot_164.w_1_0 ctx) + (sequential_generatedRoot_164.w_8_0 ctx))

def sequential_generatedRoot_164.w_10_0  (ctx : sequential_generatedRoot_164Ctx) : Int :=
  2

def sequential_generatedRoot_164.w_11_0  (ctx : sequential_generatedRoot_164Ctx) : Bool :=
  decide ((sequential_generatedRoot_164.w_9_0 ctx) = (sequential_generatedRoot_164.w_10_0 ctx))

def sequential_generatedRoot_164.outs  (ctx : sequential_generatedRoot_164Ctx) : Bool :=
  (sequential_generatedRoot_164.w_11_0 ctx)

structure sequential_generatedRoot_164.Facts_0  (ctx : sequential_generatedRoot_164Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_187 params i_0 i (current, ((sequential_generatedRoot_164.w_3_0 ctx), ((sequential_generatedRoot_164.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_164.w_2_0 ctx) (sequential_generatedRoot_164.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_164.outs ctx

theorem sequential_generatedRoot_164.facts_0  {ctx : sequential_generatedRoot_164Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_164.constraints_0 params i_0 (sequential_generatedRoot_164.w_3_0 ctx) (sequential_generatedRoot_164.w_2_0 ctx) (sequential_generatedRoot_164.w_4_0 ctx) (sequential_generatedRoot_164.w_5_0 ctx) (sequential_generatedRoot_164.w_11_0 ctx) outputs) :
    sequential_generatedRoot_164.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_164.Facts  (ctx : sequential_generatedRoot_164Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_164.Facts_0 ctx outputs

theorem sequential_generatedRoot_164.facts  (ctx : sequential_generatedRoot_164Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_164.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_164.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_164.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_166.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_166.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_166Ctx" => (sequential_generatedRoot_166.Ctx params i_0)

def sequential_generatedRoot_166.w_0_0  (ctx : sequential_generatedRoot_166Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_166.w_3_0  (ctx : sequential_generatedRoot_166Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_166.w_1_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  if (sequential_generatedRoot_166.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_166.w_2_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  0

def sequential_generatedRoot_166.w_4_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_166.w_5_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_166.w_6_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  1

def sequential_generatedRoot_166.w_7_0  (ctx : sequential_generatedRoot_166Ctx) : Bool :=
  decide ((sequential_generatedRoot_166.w_5_0 ctx) = (sequential_generatedRoot_166.w_6_0 ctx))

def sequential_generatedRoot_166.w_8_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  if (sequential_generatedRoot_166.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_166.w_9_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  ((sequential_generatedRoot_166.w_1_0 ctx) + (sequential_generatedRoot_166.w_8_0 ctx))

def sequential_generatedRoot_166.w_10_0  (ctx : sequential_generatedRoot_166Ctx) : Int :=
  2

def sequential_generatedRoot_166.w_11_0  (ctx : sequential_generatedRoot_166Ctx) : Bool :=
  decide ((sequential_generatedRoot_166.w_9_0 ctx) = (sequential_generatedRoot_166.w_10_0 ctx))

def sequential_generatedRoot_166.outs  (ctx : sequential_generatedRoot_166Ctx) : Bool :=
  (sequential_generatedRoot_166.w_11_0 ctx)

structure sequential_generatedRoot_166.Facts_0  (ctx : sequential_generatedRoot_166Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_188 params i_0 i (current, ((sequential_generatedRoot_166.w_3_0 ctx), ((sequential_generatedRoot_166.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_166.w_2_0 ctx) (sequential_generatedRoot_166.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_166.outs ctx

theorem sequential_generatedRoot_166.facts_0  {ctx : sequential_generatedRoot_166Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_166.constraints_0 params i_0 (sequential_generatedRoot_166.w_3_0 ctx) (sequential_generatedRoot_166.w_2_0 ctx) (sequential_generatedRoot_166.w_4_0 ctx) (sequential_generatedRoot_166.w_5_0 ctx) (sequential_generatedRoot_166.w_11_0 ctx) outputs) :
    sequential_generatedRoot_166.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_166.Facts  (ctx : sequential_generatedRoot_166Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_166.Facts_0 ctx outputs

theorem sequential_generatedRoot_166.facts  (ctx : sequential_generatedRoot_166Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_166.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_166.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_166.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_168.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_168.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_168Ctx" => (sequential_generatedRoot_168.Ctx params i_0)

def sequential_generatedRoot_168.w_0_0  (ctx : sequential_generatedRoot_168Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_168.w_3_0  (ctx : sequential_generatedRoot_168Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_168.w_1_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  if (sequential_generatedRoot_168.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_168.w_2_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  0

def sequential_generatedRoot_168.w_4_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_168.w_5_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_168.w_6_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  1

def sequential_generatedRoot_168.w_7_0  (ctx : sequential_generatedRoot_168Ctx) : Bool :=
  decide ((sequential_generatedRoot_168.w_5_0 ctx) = (sequential_generatedRoot_168.w_6_0 ctx))

def sequential_generatedRoot_168.w_8_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  if (sequential_generatedRoot_168.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_168.w_9_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  ((sequential_generatedRoot_168.w_1_0 ctx) + (sequential_generatedRoot_168.w_8_0 ctx))

def sequential_generatedRoot_168.w_10_0  (ctx : sequential_generatedRoot_168Ctx) : Int :=
  2

def sequential_generatedRoot_168.w_11_0  (ctx : sequential_generatedRoot_168Ctx) : Bool :=
  decide ((sequential_generatedRoot_168.w_9_0 ctx) = (sequential_generatedRoot_168.w_10_0 ctx))

def sequential_generatedRoot_168.outs  (ctx : sequential_generatedRoot_168Ctx) : Bool :=
  (sequential_generatedRoot_168.w_11_0 ctx)

structure sequential_generatedRoot_168.Facts_0  (ctx : sequential_generatedRoot_168Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_189 params i_0 i (current, ((sequential_generatedRoot_168.w_3_0 ctx), ((sequential_generatedRoot_168.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_168.w_2_0 ctx) (sequential_generatedRoot_168.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_168.outs ctx

theorem sequential_generatedRoot_168.facts_0  {ctx : sequential_generatedRoot_168Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_168.constraints_0 params i_0 (sequential_generatedRoot_168.w_3_0 ctx) (sequential_generatedRoot_168.w_2_0 ctx) (sequential_generatedRoot_168.w_4_0 ctx) (sequential_generatedRoot_168.w_5_0 ctx) (sequential_generatedRoot_168.w_11_0 ctx) outputs) :
    sequential_generatedRoot_168.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_168.Facts  (ctx : sequential_generatedRoot_168Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_168.Facts_0 ctx outputs

theorem sequential_generatedRoot_168.facts  (ctx : sequential_generatedRoot_168Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_168.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_168.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_168.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_170.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_170.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_170Ctx" => (sequential_generatedRoot_170.Ctx params i_0)

def sequential_generatedRoot_170.w_0_0  (ctx : sequential_generatedRoot_170Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_170.w_3_0  (ctx : sequential_generatedRoot_170Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_170.w_1_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  if (sequential_generatedRoot_170.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_170.w_2_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  0

def sequential_generatedRoot_170.w_4_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_170.w_5_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_170.w_6_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  1

def sequential_generatedRoot_170.w_7_0  (ctx : sequential_generatedRoot_170Ctx) : Bool :=
  decide ((sequential_generatedRoot_170.w_5_0 ctx) = (sequential_generatedRoot_170.w_6_0 ctx))

def sequential_generatedRoot_170.w_8_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  if (sequential_generatedRoot_170.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_170.w_9_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  ((sequential_generatedRoot_170.w_1_0 ctx) + (sequential_generatedRoot_170.w_8_0 ctx))

def sequential_generatedRoot_170.w_10_0  (ctx : sequential_generatedRoot_170Ctx) : Int :=
  2

def sequential_generatedRoot_170.w_11_0  (ctx : sequential_generatedRoot_170Ctx) : Bool :=
  decide ((sequential_generatedRoot_170.w_9_0 ctx) = (sequential_generatedRoot_170.w_10_0 ctx))

def sequential_generatedRoot_170.outs  (ctx : sequential_generatedRoot_170Ctx) : Bool :=
  (sequential_generatedRoot_170.w_11_0 ctx)

structure sequential_generatedRoot_170.Facts_0  (ctx : sequential_generatedRoot_170Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_190 params i_0 i (current, ((sequential_generatedRoot_170.w_3_0 ctx), ((sequential_generatedRoot_170.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_170.w_2_0 ctx) (sequential_generatedRoot_170.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_170.outs ctx

theorem sequential_generatedRoot_170.facts_0  {ctx : sequential_generatedRoot_170Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_170.constraints_0 params i_0 (sequential_generatedRoot_170.w_3_0 ctx) (sequential_generatedRoot_170.w_2_0 ctx) (sequential_generatedRoot_170.w_4_0 ctx) (sequential_generatedRoot_170.w_5_0 ctx) (sequential_generatedRoot_170.w_11_0 ctx) outputs) :
    sequential_generatedRoot_170.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_170.Facts  (ctx : sequential_generatedRoot_170Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_170.Facts_0 ctx outputs

theorem sequential_generatedRoot_170.facts  (ctx : sequential_generatedRoot_170Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_170.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_170.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_170.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_172.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_172.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_172Ctx" => (sequential_generatedRoot_172.Ctx params i_0)

def sequential_generatedRoot_172.w_0_0  (ctx : sequential_generatedRoot_172Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_172.w_3_0  (ctx : sequential_generatedRoot_172Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_172.w_1_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  if (sequential_generatedRoot_172.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_172.w_2_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  0

def sequential_generatedRoot_172.w_4_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_172.w_5_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_172.w_6_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  1

def sequential_generatedRoot_172.w_7_0  (ctx : sequential_generatedRoot_172Ctx) : Bool :=
  decide ((sequential_generatedRoot_172.w_5_0 ctx) = (sequential_generatedRoot_172.w_6_0 ctx))

def sequential_generatedRoot_172.w_8_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  if (sequential_generatedRoot_172.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_172.w_9_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  ((sequential_generatedRoot_172.w_1_0 ctx) + (sequential_generatedRoot_172.w_8_0 ctx))

def sequential_generatedRoot_172.w_10_0  (ctx : sequential_generatedRoot_172Ctx) : Int :=
  2

def sequential_generatedRoot_172.w_11_0  (ctx : sequential_generatedRoot_172Ctx) : Bool :=
  decide ((sequential_generatedRoot_172.w_9_0 ctx) = (sequential_generatedRoot_172.w_10_0 ctx))

def sequential_generatedRoot_172.outs  (ctx : sequential_generatedRoot_172Ctx) : Bool :=
  (sequential_generatedRoot_172.w_11_0 ctx)

structure sequential_generatedRoot_172.Facts_0  (ctx : sequential_generatedRoot_172Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_191 params i_0 i (current, ((sequential_generatedRoot_172.w_3_0 ctx), ((sequential_generatedRoot_172.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_172.w_2_0 ctx) (sequential_generatedRoot_172.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_172.outs ctx

theorem sequential_generatedRoot_172.facts_0  {ctx : sequential_generatedRoot_172Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_172.constraints_0 params i_0 (sequential_generatedRoot_172.w_3_0 ctx) (sequential_generatedRoot_172.w_2_0 ctx) (sequential_generatedRoot_172.w_4_0 ctx) (sequential_generatedRoot_172.w_5_0 ctx) (sequential_generatedRoot_172.w_11_0 ctx) outputs) :
    sequential_generatedRoot_172.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_172.Facts  (ctx : sequential_generatedRoot_172Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_172.Facts_0 ctx outputs

theorem sequential_generatedRoot_172.facts  (ctx : sequential_generatedRoot_172Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_172.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_172.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_172.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_174.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_174.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_174Ctx" => (sequential_generatedRoot_174.Ctx params i_0)

def sequential_generatedRoot_174.w_0_0  (ctx : sequential_generatedRoot_174Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_174.w_3_0  (ctx : sequential_generatedRoot_174Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_174.w_1_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  if (sequential_generatedRoot_174.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_174.w_2_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  0

def sequential_generatedRoot_174.w_4_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_174.w_5_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_174.w_6_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  1

def sequential_generatedRoot_174.w_7_0  (ctx : sequential_generatedRoot_174Ctx) : Bool :=
  decide ((sequential_generatedRoot_174.w_5_0 ctx) = (sequential_generatedRoot_174.w_6_0 ctx))

def sequential_generatedRoot_174.w_8_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  if (sequential_generatedRoot_174.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_174.w_9_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  ((sequential_generatedRoot_174.w_1_0 ctx) + (sequential_generatedRoot_174.w_8_0 ctx))

def sequential_generatedRoot_174.w_10_0  (ctx : sequential_generatedRoot_174Ctx) : Int :=
  2

def sequential_generatedRoot_174.w_11_0  (ctx : sequential_generatedRoot_174Ctx) : Bool :=
  decide ((sequential_generatedRoot_174.w_9_0 ctx) = (sequential_generatedRoot_174.w_10_0 ctx))

def sequential_generatedRoot_174.outs  (ctx : sequential_generatedRoot_174Ctx) : Bool :=
  (sequential_generatedRoot_174.w_11_0 ctx)

structure sequential_generatedRoot_174.Facts_0  (ctx : sequential_generatedRoot_174Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_192 params i_0 i (current, ((sequential_generatedRoot_174.w_3_0 ctx), ((sequential_generatedRoot_174.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_174.w_2_0 ctx) (sequential_generatedRoot_174.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_174.outs ctx

theorem sequential_generatedRoot_174.facts_0  {ctx : sequential_generatedRoot_174Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_174.constraints_0 params i_0 (sequential_generatedRoot_174.w_3_0 ctx) (sequential_generatedRoot_174.w_2_0 ctx) (sequential_generatedRoot_174.w_4_0 ctx) (sequential_generatedRoot_174.w_5_0 ctx) (sequential_generatedRoot_174.w_11_0 ctx) outputs) :
    sequential_generatedRoot_174.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_174.Facts  (ctx : sequential_generatedRoot_174Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_174.Facts_0 ctx outputs

theorem sequential_generatedRoot_174.facts  (ctx : sequential_generatedRoot_174Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_174.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_174.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_174.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_176.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_176.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_176Ctx" => (sequential_generatedRoot_176.Ctx params i_0)

def sequential_generatedRoot_176.w_0_0  (ctx : sequential_generatedRoot_176Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_176.w_3_0  (ctx : sequential_generatedRoot_176Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_176.w_1_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  if (sequential_generatedRoot_176.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_176.w_2_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  0

def sequential_generatedRoot_176.w_4_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_176.w_5_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_176.w_6_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  1

def sequential_generatedRoot_176.w_7_0  (ctx : sequential_generatedRoot_176Ctx) : Bool :=
  decide ((sequential_generatedRoot_176.w_5_0 ctx) = (sequential_generatedRoot_176.w_6_0 ctx))

def sequential_generatedRoot_176.w_8_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  if (sequential_generatedRoot_176.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_176.w_9_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  ((sequential_generatedRoot_176.w_1_0 ctx) + (sequential_generatedRoot_176.w_8_0 ctx))

def sequential_generatedRoot_176.w_10_0  (ctx : sequential_generatedRoot_176Ctx) : Int :=
  2

def sequential_generatedRoot_176.w_11_0  (ctx : sequential_generatedRoot_176Ctx) : Bool :=
  decide ((sequential_generatedRoot_176.w_9_0 ctx) = (sequential_generatedRoot_176.w_10_0 ctx))

def sequential_generatedRoot_176.outs  (ctx : sequential_generatedRoot_176Ctx) : Bool :=
  (sequential_generatedRoot_176.w_11_0 ctx)

structure sequential_generatedRoot_176.Facts_0  (ctx : sequential_generatedRoot_176Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_193 params i_0 i (current, ((sequential_generatedRoot_176.w_3_0 ctx), ((sequential_generatedRoot_176.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_176.w_2_0 ctx) (sequential_generatedRoot_176.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_176.outs ctx

theorem sequential_generatedRoot_176.facts_0  {ctx : sequential_generatedRoot_176Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_176.constraints_0 params i_0 (sequential_generatedRoot_176.w_3_0 ctx) (sequential_generatedRoot_176.w_2_0 ctx) (sequential_generatedRoot_176.w_4_0 ctx) (sequential_generatedRoot_176.w_5_0 ctx) (sequential_generatedRoot_176.w_11_0 ctx) outputs) :
    sequential_generatedRoot_176.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_176.Facts  (ctx : sequential_generatedRoot_176Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_176.Facts_0 ctx outputs

theorem sequential_generatedRoot_176.facts  (ctx : sequential_generatedRoot_176Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_176.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_176.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_176.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_178.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_178.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_178Ctx" => (sequential_generatedRoot_178.Ctx params i_0)

def sequential_generatedRoot_178.w_0_0  (ctx : sequential_generatedRoot_178Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_178.w_3_0  (ctx : sequential_generatedRoot_178Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_178.w_1_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  if (sequential_generatedRoot_178.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_178.w_2_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  0

def sequential_generatedRoot_178.w_4_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_178.w_5_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_178.w_6_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  1

def sequential_generatedRoot_178.w_7_0  (ctx : sequential_generatedRoot_178Ctx) : Bool :=
  decide ((sequential_generatedRoot_178.w_5_0 ctx) = (sequential_generatedRoot_178.w_6_0 ctx))

def sequential_generatedRoot_178.w_8_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  if (sequential_generatedRoot_178.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_178.w_9_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  ((sequential_generatedRoot_178.w_1_0 ctx) + (sequential_generatedRoot_178.w_8_0 ctx))

def sequential_generatedRoot_178.w_10_0  (ctx : sequential_generatedRoot_178Ctx) : Int :=
  2

def sequential_generatedRoot_178.w_11_0  (ctx : sequential_generatedRoot_178Ctx) : Bool :=
  decide ((sequential_generatedRoot_178.w_9_0 ctx) = (sequential_generatedRoot_178.w_10_0 ctx))

def sequential_generatedRoot_178.outs  (ctx : sequential_generatedRoot_178Ctx) : Bool :=
  (sequential_generatedRoot_178.w_11_0 ctx)

structure sequential_generatedRoot_178.Facts_0  (ctx : sequential_generatedRoot_178Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_194 params i_0 i (current, ((sequential_generatedRoot_178.w_3_0 ctx), ((sequential_generatedRoot_178.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_178.w_2_0 ctx) (sequential_generatedRoot_178.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_178.outs ctx

theorem sequential_generatedRoot_178.facts_0  {ctx : sequential_generatedRoot_178Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_178.constraints_0 params i_0 (sequential_generatedRoot_178.w_3_0 ctx) (sequential_generatedRoot_178.w_2_0 ctx) (sequential_generatedRoot_178.w_4_0 ctx) (sequential_generatedRoot_178.w_5_0 ctx) (sequential_generatedRoot_178.w_11_0 ctx) outputs) :
    sequential_generatedRoot_178.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_178.Facts  (ctx : sequential_generatedRoot_178Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_178.Facts_0 ctx outputs

theorem sequential_generatedRoot_178.facts  (ctx : sequential_generatedRoot_178Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_178.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_178.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_178.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_180.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_180.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_180Ctx" => (sequential_generatedRoot_180.Ctx params i_0)

def sequential_generatedRoot_180.w_0_0  (ctx : sequential_generatedRoot_180Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_180.w_3_0  (ctx : sequential_generatedRoot_180Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_180.w_1_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  if (sequential_generatedRoot_180.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_180.w_2_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  0

def sequential_generatedRoot_180.w_4_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_180.w_5_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_180.w_6_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  1

def sequential_generatedRoot_180.w_7_0  (ctx : sequential_generatedRoot_180Ctx) : Bool :=
  decide ((sequential_generatedRoot_180.w_5_0 ctx) = (sequential_generatedRoot_180.w_6_0 ctx))

def sequential_generatedRoot_180.w_8_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  if (sequential_generatedRoot_180.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_180.w_9_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  ((sequential_generatedRoot_180.w_1_0 ctx) + (sequential_generatedRoot_180.w_8_0 ctx))

def sequential_generatedRoot_180.w_10_0  (ctx : sequential_generatedRoot_180Ctx) : Int :=
  2

def sequential_generatedRoot_180.w_11_0  (ctx : sequential_generatedRoot_180Ctx) : Bool :=
  decide ((sequential_generatedRoot_180.w_9_0 ctx) = (sequential_generatedRoot_180.w_10_0 ctx))

def sequential_generatedRoot_180.outs  (ctx : sequential_generatedRoot_180Ctx) : Bool :=
  (sequential_generatedRoot_180.w_11_0 ctx)

structure sequential_generatedRoot_180.Facts_0  (ctx : sequential_generatedRoot_180Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_195 params i_0 i (current, ((sequential_generatedRoot_180.w_3_0 ctx), ((sequential_generatedRoot_180.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_180.w_2_0 ctx) (sequential_generatedRoot_180.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_180.outs ctx

theorem sequential_generatedRoot_180.facts_0  {ctx : sequential_generatedRoot_180Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_180.constraints_0 params i_0 (sequential_generatedRoot_180.w_3_0 ctx) (sequential_generatedRoot_180.w_2_0 ctx) (sequential_generatedRoot_180.w_4_0 ctx) (sequential_generatedRoot_180.w_5_0 ctx) (sequential_generatedRoot_180.w_11_0 ctx) outputs) :
    sequential_generatedRoot_180.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_180.Facts  (ctx : sequential_generatedRoot_180Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_180.Facts_0 ctx outputs

theorem sequential_generatedRoot_180.facts  (ctx : sequential_generatedRoot_180Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_180.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_180.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_180.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_182.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_182.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_182Ctx" => (sequential_generatedRoot_182.Ctx params i_0)

def sequential_generatedRoot_182.w_0_0  (ctx : sequential_generatedRoot_182Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_182.w_3_0  (ctx : sequential_generatedRoot_182Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_182.w_1_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  if (sequential_generatedRoot_182.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_182.w_2_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  0

def sequential_generatedRoot_182.w_4_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_182.w_5_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_182.w_6_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  1

def sequential_generatedRoot_182.w_7_0  (ctx : sequential_generatedRoot_182Ctx) : Bool :=
  decide ((sequential_generatedRoot_182.w_5_0 ctx) = (sequential_generatedRoot_182.w_6_0 ctx))

def sequential_generatedRoot_182.w_8_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  if (sequential_generatedRoot_182.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_182.w_9_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  ((sequential_generatedRoot_182.w_1_0 ctx) + (sequential_generatedRoot_182.w_8_0 ctx))

def sequential_generatedRoot_182.w_10_0  (ctx : sequential_generatedRoot_182Ctx) : Int :=
  2

def sequential_generatedRoot_182.w_11_0  (ctx : sequential_generatedRoot_182Ctx) : Bool :=
  decide ((sequential_generatedRoot_182.w_9_0 ctx) = (sequential_generatedRoot_182.w_10_0 ctx))

def sequential_generatedRoot_182.outs  (ctx : sequential_generatedRoot_182Ctx) : Bool :=
  (sequential_generatedRoot_182.w_11_0 ctx)

structure sequential_generatedRoot_182.Facts_0  (ctx : sequential_generatedRoot_182Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_196 params i_0 i (current, ((sequential_generatedRoot_182.w_3_0 ctx), ((sequential_generatedRoot_182.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_182.w_2_0 ctx) (sequential_generatedRoot_182.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_182.outs ctx

theorem sequential_generatedRoot_182.facts_0  {ctx : sequential_generatedRoot_182Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_182.constraints_0 params i_0 (sequential_generatedRoot_182.w_3_0 ctx) (sequential_generatedRoot_182.w_2_0 ctx) (sequential_generatedRoot_182.w_4_0 ctx) (sequential_generatedRoot_182.w_5_0 ctx) (sequential_generatedRoot_182.w_11_0 ctx) outputs) :
    sequential_generatedRoot_182.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_182.Facts  (ctx : sequential_generatedRoot_182Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_182.Facts_0 ctx outputs

theorem sequential_generatedRoot_182.facts  (ctx : sequential_generatedRoot_182Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_182.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_182.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_182.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_184.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_184.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_184Ctx" => (sequential_generatedRoot_184.Ctx params i_0)

def sequential_generatedRoot_184.w_0_0  (ctx : sequential_generatedRoot_184Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_184.w_3_0  (ctx : sequential_generatedRoot_184Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_184.w_1_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  if (sequential_generatedRoot_184.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_184.w_2_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  0

def sequential_generatedRoot_184.w_4_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_184.w_5_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_184.w_6_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  1

def sequential_generatedRoot_184.w_7_0  (ctx : sequential_generatedRoot_184Ctx) : Bool :=
  decide ((sequential_generatedRoot_184.w_5_0 ctx) = (sequential_generatedRoot_184.w_6_0 ctx))

def sequential_generatedRoot_184.w_8_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  if (sequential_generatedRoot_184.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_184.w_9_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  ((sequential_generatedRoot_184.w_1_0 ctx) + (sequential_generatedRoot_184.w_8_0 ctx))

def sequential_generatedRoot_184.w_10_0  (ctx : sequential_generatedRoot_184Ctx) : Int :=
  2

def sequential_generatedRoot_184.w_11_0  (ctx : sequential_generatedRoot_184Ctx) : Bool :=
  decide ((sequential_generatedRoot_184.w_9_0 ctx) = (sequential_generatedRoot_184.w_10_0 ctx))

def sequential_generatedRoot_184.outs  (ctx : sequential_generatedRoot_184Ctx) : Bool :=
  (sequential_generatedRoot_184.w_11_0 ctx)

structure sequential_generatedRoot_184.Facts_0  (ctx : sequential_generatedRoot_184Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_197 params i_0 i (current, ((sequential_generatedRoot_184.w_3_0 ctx), ((sequential_generatedRoot_184.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_184.w_2_0 ctx) (sequential_generatedRoot_184.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_184.outs ctx

theorem sequential_generatedRoot_184.facts_0  {ctx : sequential_generatedRoot_184Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_184.constraints_0 params i_0 (sequential_generatedRoot_184.w_3_0 ctx) (sequential_generatedRoot_184.w_2_0 ctx) (sequential_generatedRoot_184.w_4_0 ctx) (sequential_generatedRoot_184.w_5_0 ctx) (sequential_generatedRoot_184.w_11_0 ctx) outputs) :
    sequential_generatedRoot_184.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_184.Facts  (ctx : sequential_generatedRoot_184Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_184.Facts_0 ctx outputs

theorem sequential_generatedRoot_184.facts  (ctx : sequential_generatedRoot_184Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_184.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_184.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_184.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_186.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_186.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_186Ctx" => (sequential_generatedRoot_186.Ctx params i_0)

def sequential_generatedRoot_186.w_0_0  (ctx : sequential_generatedRoot_186Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_186.w_3_0  (ctx : sequential_generatedRoot_186Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_186.w_1_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  if (sequential_generatedRoot_186.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_186.w_2_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  0

def sequential_generatedRoot_186.w_4_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_186.w_5_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_186.w_6_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  1

def sequential_generatedRoot_186.w_7_0  (ctx : sequential_generatedRoot_186Ctx) : Bool :=
  decide ((sequential_generatedRoot_186.w_5_0 ctx) = (sequential_generatedRoot_186.w_6_0 ctx))

def sequential_generatedRoot_186.w_8_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  if (sequential_generatedRoot_186.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_186.w_9_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  ((sequential_generatedRoot_186.w_1_0 ctx) + (sequential_generatedRoot_186.w_8_0 ctx))

def sequential_generatedRoot_186.w_10_0  (ctx : sequential_generatedRoot_186Ctx) : Int :=
  2

def sequential_generatedRoot_186.w_11_0  (ctx : sequential_generatedRoot_186Ctx) : Bool :=
  decide ((sequential_generatedRoot_186.w_9_0 ctx) = (sequential_generatedRoot_186.w_10_0 ctx))

def sequential_generatedRoot_186.outs  (ctx : sequential_generatedRoot_186Ctx) : Bool :=
  (sequential_generatedRoot_186.w_11_0 ctx)

structure sequential_generatedRoot_186.Facts_0  (ctx : sequential_generatedRoot_186Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_198 params i_0 i (current, ((sequential_generatedRoot_186.w_3_0 ctx), ((sequential_generatedRoot_186.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_186.w_2_0 ctx) (sequential_generatedRoot_186.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_186.outs ctx

theorem sequential_generatedRoot_186.facts_0  {ctx : sequential_generatedRoot_186Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_186.constraints_0 params i_0 (sequential_generatedRoot_186.w_3_0 ctx) (sequential_generatedRoot_186.w_2_0 ctx) (sequential_generatedRoot_186.w_4_0 ctx) (sequential_generatedRoot_186.w_5_0 ctx) (sequential_generatedRoot_186.w_11_0 ctx) outputs) :
    sequential_generatedRoot_186.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_186.Facts  (ctx : sequential_generatedRoot_186Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_186.Facts_0 ctx outputs

theorem sequential_generatedRoot_186.facts  (ctx : sequential_generatedRoot_186Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_186.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_186.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_186.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_188.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_188.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_188Ctx" => (sequential_generatedRoot_188.Ctx params i_0)

def sequential_generatedRoot_188.w_0_0  (ctx : sequential_generatedRoot_188Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_188.w_3_0  (ctx : sequential_generatedRoot_188Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_188.w_1_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  if (sequential_generatedRoot_188.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_188.w_2_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  0

def sequential_generatedRoot_188.w_4_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_188.w_5_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_188.w_6_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  1

def sequential_generatedRoot_188.w_7_0  (ctx : sequential_generatedRoot_188Ctx) : Bool :=
  decide ((sequential_generatedRoot_188.w_5_0 ctx) = (sequential_generatedRoot_188.w_6_0 ctx))

def sequential_generatedRoot_188.w_8_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  if (sequential_generatedRoot_188.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_188.w_9_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  ((sequential_generatedRoot_188.w_1_0 ctx) + (sequential_generatedRoot_188.w_8_0 ctx))

def sequential_generatedRoot_188.w_10_0  (ctx : sequential_generatedRoot_188Ctx) : Int :=
  2

def sequential_generatedRoot_188.w_11_0  (ctx : sequential_generatedRoot_188Ctx) : Bool :=
  decide ((sequential_generatedRoot_188.w_9_0 ctx) = (sequential_generatedRoot_188.w_10_0 ctx))

def sequential_generatedRoot_188.outs  (ctx : sequential_generatedRoot_188Ctx) : Bool :=
  (sequential_generatedRoot_188.w_11_0 ctx)

structure sequential_generatedRoot_188.Facts_0  (ctx : sequential_generatedRoot_188Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_199 params i_0 i (current, ((sequential_generatedRoot_188.w_3_0 ctx), ((sequential_generatedRoot_188.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_188.w_2_0 ctx) (sequential_generatedRoot_188.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_188.outs ctx

theorem sequential_generatedRoot_188.facts_0  {ctx : sequential_generatedRoot_188Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_188.constraints_0 params i_0 (sequential_generatedRoot_188.w_3_0 ctx) (sequential_generatedRoot_188.w_2_0 ctx) (sequential_generatedRoot_188.w_4_0 ctx) (sequential_generatedRoot_188.w_5_0 ctx) (sequential_generatedRoot_188.w_11_0 ctx) outputs) :
    sequential_generatedRoot_188.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_188.Facts  (ctx : sequential_generatedRoot_188Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_188.Facts_0 ctx outputs

theorem sequential_generatedRoot_188.facts  (ctx : sequential_generatedRoot_188Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_188.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_188.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_188.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_190.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_190.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_190Ctx" => (sequential_generatedRoot_190.Ctx params i_0)

def sequential_generatedRoot_190.w_0_0  (ctx : sequential_generatedRoot_190Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_190.w_3_0  (ctx : sequential_generatedRoot_190Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_190.w_1_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  if (sequential_generatedRoot_190.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_190.w_2_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  0

def sequential_generatedRoot_190.w_4_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_190.w_5_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_190.w_6_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  1

def sequential_generatedRoot_190.w_7_0  (ctx : sequential_generatedRoot_190Ctx) : Bool :=
  decide ((sequential_generatedRoot_190.w_5_0 ctx) = (sequential_generatedRoot_190.w_6_0 ctx))

def sequential_generatedRoot_190.w_8_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  if (sequential_generatedRoot_190.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_190.w_9_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  ((sequential_generatedRoot_190.w_1_0 ctx) + (sequential_generatedRoot_190.w_8_0 ctx))

def sequential_generatedRoot_190.w_10_0  (ctx : sequential_generatedRoot_190Ctx) : Int :=
  2

def sequential_generatedRoot_190.w_11_0  (ctx : sequential_generatedRoot_190Ctx) : Bool :=
  decide ((sequential_generatedRoot_190.w_9_0 ctx) = (sequential_generatedRoot_190.w_10_0 ctx))

def sequential_generatedRoot_190.outs  (ctx : sequential_generatedRoot_190Ctx) : Bool :=
  (sequential_generatedRoot_190.w_11_0 ctx)

structure sequential_generatedRoot_190.Facts_0  (ctx : sequential_generatedRoot_190Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_200 params i_0 i (current, ((sequential_generatedRoot_190.w_3_0 ctx), ((sequential_generatedRoot_190.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_190.w_2_0 ctx) (sequential_generatedRoot_190.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_190.outs ctx

theorem sequential_generatedRoot_190.facts_0  {ctx : sequential_generatedRoot_190Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_190.constraints_0 params i_0 (sequential_generatedRoot_190.w_3_0 ctx) (sequential_generatedRoot_190.w_2_0 ctx) (sequential_generatedRoot_190.w_4_0 ctx) (sequential_generatedRoot_190.w_5_0 ctx) (sequential_generatedRoot_190.w_11_0 ctx) outputs) :
    sequential_generatedRoot_190.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_190.Facts  (ctx : sequential_generatedRoot_190Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_190.Facts_0 ctx outputs

theorem sequential_generatedRoot_190.facts  (ctx : sequential_generatedRoot_190Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_190.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_190.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_190.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_192.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_192.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_192Ctx" => (sequential_generatedRoot_192.Ctx params i_0)

def sequential_generatedRoot_192.w_0_0  (ctx : sequential_generatedRoot_192Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_192.w_3_0  (ctx : sequential_generatedRoot_192Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_192.w_1_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  if (sequential_generatedRoot_192.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_192.w_2_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  0

def sequential_generatedRoot_192.w_4_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_192.w_5_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_192.w_6_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  1

def sequential_generatedRoot_192.w_7_0  (ctx : sequential_generatedRoot_192Ctx) : Bool :=
  decide ((sequential_generatedRoot_192.w_5_0 ctx) = (sequential_generatedRoot_192.w_6_0 ctx))

def sequential_generatedRoot_192.w_8_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  if (sequential_generatedRoot_192.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_192.w_9_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  ((sequential_generatedRoot_192.w_1_0 ctx) + (sequential_generatedRoot_192.w_8_0 ctx))

def sequential_generatedRoot_192.w_10_0  (ctx : sequential_generatedRoot_192Ctx) : Int :=
  2

def sequential_generatedRoot_192.w_11_0  (ctx : sequential_generatedRoot_192Ctx) : Bool :=
  decide ((sequential_generatedRoot_192.w_9_0 ctx) = (sequential_generatedRoot_192.w_10_0 ctx))

def sequential_generatedRoot_192.outs  (ctx : sequential_generatedRoot_192Ctx) : Bool :=
  (sequential_generatedRoot_192.w_11_0 ctx)

structure sequential_generatedRoot_192.Facts_0  (ctx : sequential_generatedRoot_192Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_201 params i_0 i (current, ((sequential_generatedRoot_192.w_3_0 ctx), ((sequential_generatedRoot_192.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_192.w_2_0 ctx) (sequential_generatedRoot_192.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_192.outs ctx

theorem sequential_generatedRoot_192.facts_0  {ctx : sequential_generatedRoot_192Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_192.constraints_0 params i_0 (sequential_generatedRoot_192.w_3_0 ctx) (sequential_generatedRoot_192.w_2_0 ctx) (sequential_generatedRoot_192.w_4_0 ctx) (sequential_generatedRoot_192.w_5_0 ctx) (sequential_generatedRoot_192.w_11_0 ctx) outputs) :
    sequential_generatedRoot_192.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_192.Facts  (ctx : sequential_generatedRoot_192Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_192.Facts_0 ctx outputs

theorem sequential_generatedRoot_192.facts  (ctx : sequential_generatedRoot_192Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_192.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_192.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_192.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_194.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_194.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_194Ctx" => (sequential_generatedRoot_194.Ctx params i_0)

def sequential_generatedRoot_194.w_0_0  (ctx : sequential_generatedRoot_194Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_194.w_3_0  (ctx : sequential_generatedRoot_194Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_194.w_1_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  if (sequential_generatedRoot_194.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_194.w_2_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  0

def sequential_generatedRoot_194.w_4_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_194.w_5_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_194.w_6_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  1

def sequential_generatedRoot_194.w_7_0  (ctx : sequential_generatedRoot_194Ctx) : Bool :=
  decide ((sequential_generatedRoot_194.w_5_0 ctx) = (sequential_generatedRoot_194.w_6_0 ctx))

def sequential_generatedRoot_194.w_8_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  if (sequential_generatedRoot_194.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_194.w_9_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  ((sequential_generatedRoot_194.w_1_0 ctx) + (sequential_generatedRoot_194.w_8_0 ctx))

def sequential_generatedRoot_194.w_10_0  (ctx : sequential_generatedRoot_194Ctx) : Int :=
  2

def sequential_generatedRoot_194.w_11_0  (ctx : sequential_generatedRoot_194Ctx) : Bool :=
  decide ((sequential_generatedRoot_194.w_9_0 ctx) = (sequential_generatedRoot_194.w_10_0 ctx))

def sequential_generatedRoot_194.outs  (ctx : sequential_generatedRoot_194Ctx) : Bool :=
  (sequential_generatedRoot_194.w_11_0 ctx)

structure sequential_generatedRoot_194.Facts_0  (ctx : sequential_generatedRoot_194Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_202 params i_0 i (current, ((sequential_generatedRoot_194.w_3_0 ctx), ((sequential_generatedRoot_194.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_194.w_2_0 ctx) (sequential_generatedRoot_194.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_194.outs ctx

theorem sequential_generatedRoot_194.facts_0  {ctx : sequential_generatedRoot_194Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_194.constraints_0 params i_0 (sequential_generatedRoot_194.w_3_0 ctx) (sequential_generatedRoot_194.w_2_0 ctx) (sequential_generatedRoot_194.w_4_0 ctx) (sequential_generatedRoot_194.w_5_0 ctx) (sequential_generatedRoot_194.w_11_0 ctx) outputs) :
    sequential_generatedRoot_194.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_194.Facts  (ctx : sequential_generatedRoot_194Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_194.Facts_0 ctx outputs

theorem sequential_generatedRoot_194.facts  (ctx : sequential_generatedRoot_194Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_194.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_194.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_194.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_196.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_196.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_196Ctx" => (sequential_generatedRoot_196.Ctx params i_0)

def sequential_generatedRoot_196.w_0_0  (ctx : sequential_generatedRoot_196Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_196.w_3_0  (ctx : sequential_generatedRoot_196Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_196.w_1_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  if (sequential_generatedRoot_196.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_196.w_2_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  0

def sequential_generatedRoot_196.w_4_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_196.w_5_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_196.w_6_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  1

def sequential_generatedRoot_196.w_7_0  (ctx : sequential_generatedRoot_196Ctx) : Bool :=
  decide ((sequential_generatedRoot_196.w_5_0 ctx) = (sequential_generatedRoot_196.w_6_0 ctx))

def sequential_generatedRoot_196.w_8_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  if (sequential_generatedRoot_196.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_196.w_9_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  ((sequential_generatedRoot_196.w_1_0 ctx) + (sequential_generatedRoot_196.w_8_0 ctx))

def sequential_generatedRoot_196.w_10_0  (ctx : sequential_generatedRoot_196Ctx) : Int :=
  2

def sequential_generatedRoot_196.w_11_0  (ctx : sequential_generatedRoot_196Ctx) : Bool :=
  decide ((sequential_generatedRoot_196.w_9_0 ctx) = (sequential_generatedRoot_196.w_10_0 ctx))

def sequential_generatedRoot_196.outs  (ctx : sequential_generatedRoot_196Ctx) : Bool :=
  (sequential_generatedRoot_196.w_11_0 ctx)

structure sequential_generatedRoot_196.Facts_0  (ctx : sequential_generatedRoot_196Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_203 params i_0 i (current, ((sequential_generatedRoot_196.w_3_0 ctx), ((sequential_generatedRoot_196.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_196.w_2_0 ctx) (sequential_generatedRoot_196.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_196.outs ctx

theorem sequential_generatedRoot_196.facts_0  {ctx : sequential_generatedRoot_196Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_196.constraints_0 params i_0 (sequential_generatedRoot_196.w_3_0 ctx) (sequential_generatedRoot_196.w_2_0 ctx) (sequential_generatedRoot_196.w_4_0 ctx) (sequential_generatedRoot_196.w_5_0 ctx) (sequential_generatedRoot_196.w_11_0 ctx) outputs) :
    sequential_generatedRoot_196.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_196.Facts  (ctx : sequential_generatedRoot_196Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_196.Facts_0 ctx outputs

theorem sequential_generatedRoot_196.facts  (ctx : sequential_generatedRoot_196Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_196.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_196.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_196.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_198.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_198.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_198Ctx" => (sequential_generatedRoot_198.Ctx params i_0)

def sequential_generatedRoot_198.w_0_0  (ctx : sequential_generatedRoot_198Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_198.w_3_0  (ctx : sequential_generatedRoot_198Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_198.w_1_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  if (sequential_generatedRoot_198.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_198.w_2_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  0

def sequential_generatedRoot_198.w_4_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_198.w_5_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_198.w_6_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  1

def sequential_generatedRoot_198.w_7_0  (ctx : sequential_generatedRoot_198Ctx) : Bool :=
  decide ((sequential_generatedRoot_198.w_5_0 ctx) = (sequential_generatedRoot_198.w_6_0 ctx))

def sequential_generatedRoot_198.w_8_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  if (sequential_generatedRoot_198.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_198.w_9_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  ((sequential_generatedRoot_198.w_1_0 ctx) + (sequential_generatedRoot_198.w_8_0 ctx))

def sequential_generatedRoot_198.w_10_0  (ctx : sequential_generatedRoot_198Ctx) : Int :=
  2

def sequential_generatedRoot_198.w_11_0  (ctx : sequential_generatedRoot_198Ctx) : Bool :=
  decide ((sequential_generatedRoot_198.w_9_0 ctx) = (sequential_generatedRoot_198.w_10_0 ctx))

def sequential_generatedRoot_198.outs  (ctx : sequential_generatedRoot_198Ctx) : Bool :=
  (sequential_generatedRoot_198.w_11_0 ctx)

structure sequential_generatedRoot_198.Facts_0  (ctx : sequential_generatedRoot_198Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_204 params i_0 i (current, ((sequential_generatedRoot_198.w_3_0 ctx), ((sequential_generatedRoot_198.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_198.w_2_0 ctx) (sequential_generatedRoot_198.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_198.outs ctx

theorem sequential_generatedRoot_198.facts_0  {ctx : sequential_generatedRoot_198Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_198.constraints_0 params i_0 (sequential_generatedRoot_198.w_3_0 ctx) (sequential_generatedRoot_198.w_2_0 ctx) (sequential_generatedRoot_198.w_4_0 ctx) (sequential_generatedRoot_198.w_5_0 ctx) (sequential_generatedRoot_198.w_11_0 ctx) outputs) :
    sequential_generatedRoot_198.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_198.Facts  (ctx : sequential_generatedRoot_198Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_198.Facts_0 ctx outputs

theorem sequential_generatedRoot_198.facts  (ctx : sequential_generatedRoot_198Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_198.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_198.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_198.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_200.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_200.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_200Ctx" => (sequential_generatedRoot_200.Ctx params i_0)

def sequential_generatedRoot_200.w_0_0  (ctx : sequential_generatedRoot_200Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_200.w_3_0  (ctx : sequential_generatedRoot_200Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_200.w_1_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  if (sequential_generatedRoot_200.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_200.w_2_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  0

def sequential_generatedRoot_200.w_4_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_200.w_5_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_200.w_6_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  1

def sequential_generatedRoot_200.w_7_0  (ctx : sequential_generatedRoot_200Ctx) : Bool :=
  decide ((sequential_generatedRoot_200.w_5_0 ctx) = (sequential_generatedRoot_200.w_6_0 ctx))

def sequential_generatedRoot_200.w_8_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  if (sequential_generatedRoot_200.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_200.w_9_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  ((sequential_generatedRoot_200.w_1_0 ctx) + (sequential_generatedRoot_200.w_8_0 ctx))

def sequential_generatedRoot_200.w_10_0  (ctx : sequential_generatedRoot_200Ctx) : Int :=
  2

def sequential_generatedRoot_200.w_11_0  (ctx : sequential_generatedRoot_200Ctx) : Bool :=
  decide ((sequential_generatedRoot_200.w_9_0 ctx) = (sequential_generatedRoot_200.w_10_0 ctx))

def sequential_generatedRoot_200.outs  (ctx : sequential_generatedRoot_200Ctx) : Bool :=
  (sequential_generatedRoot_200.w_11_0 ctx)

structure sequential_generatedRoot_200.Facts_0  (ctx : sequential_generatedRoot_200Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_205 params i_0 i (current, ((sequential_generatedRoot_200.w_3_0 ctx), ((sequential_generatedRoot_200.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_200.w_2_0 ctx) (sequential_generatedRoot_200.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_200.outs ctx

theorem sequential_generatedRoot_200.facts_0  {ctx : sequential_generatedRoot_200Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_200.constraints_0 params i_0 (sequential_generatedRoot_200.w_3_0 ctx) (sequential_generatedRoot_200.w_2_0 ctx) (sequential_generatedRoot_200.w_4_0 ctx) (sequential_generatedRoot_200.w_5_0 ctx) (sequential_generatedRoot_200.w_11_0 ctx) outputs) :
    sequential_generatedRoot_200.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_200.Facts  (ctx : sequential_generatedRoot_200Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_200.Facts_0 ctx outputs

theorem sequential_generatedRoot_200.facts  (ctx : sequential_generatedRoot_200Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_200.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_200.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_200.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_202.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_202.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_202Ctx" => (sequential_generatedRoot_202.Ctx params i_0)

def sequential_generatedRoot_202.w_0_0  (ctx : sequential_generatedRoot_202Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_202.w_3_0  (ctx : sequential_generatedRoot_202Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_202.w_1_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  if (sequential_generatedRoot_202.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_202.w_2_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  0

def sequential_generatedRoot_202.w_4_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_202.w_5_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_202.w_6_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  1

def sequential_generatedRoot_202.w_7_0  (ctx : sequential_generatedRoot_202Ctx) : Bool :=
  decide ((sequential_generatedRoot_202.w_5_0 ctx) = (sequential_generatedRoot_202.w_6_0 ctx))

def sequential_generatedRoot_202.w_8_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  if (sequential_generatedRoot_202.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_202.w_9_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  ((sequential_generatedRoot_202.w_1_0 ctx) + (sequential_generatedRoot_202.w_8_0 ctx))

def sequential_generatedRoot_202.w_10_0  (ctx : sequential_generatedRoot_202Ctx) : Int :=
  2

def sequential_generatedRoot_202.w_11_0  (ctx : sequential_generatedRoot_202Ctx) : Bool :=
  decide ((sequential_generatedRoot_202.w_9_0 ctx) = (sequential_generatedRoot_202.w_10_0 ctx))

def sequential_generatedRoot_202.outs  (ctx : sequential_generatedRoot_202Ctx) : Bool :=
  (sequential_generatedRoot_202.w_11_0 ctx)

structure sequential_generatedRoot_202.Facts_0  (ctx : sequential_generatedRoot_202Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_206 params i_0 i (current, ((sequential_generatedRoot_202.w_3_0 ctx), ((sequential_generatedRoot_202.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_202.w_2_0 ctx) (sequential_generatedRoot_202.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_202.outs ctx

theorem sequential_generatedRoot_202.facts_0  {ctx : sequential_generatedRoot_202Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_202.constraints_0 params i_0 (sequential_generatedRoot_202.w_3_0 ctx) (sequential_generatedRoot_202.w_2_0 ctx) (sequential_generatedRoot_202.w_4_0 ctx) (sequential_generatedRoot_202.w_5_0 ctx) (sequential_generatedRoot_202.w_11_0 ctx) outputs) :
    sequential_generatedRoot_202.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_202.Facts  (ctx : sequential_generatedRoot_202Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_202.Facts_0 ctx outputs

theorem sequential_generatedRoot_202.facts  (ctx : sequential_generatedRoot_202Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_202.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_202.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_202.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_204.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_204.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_204Ctx" => (sequential_generatedRoot_204.Ctx params i_0)

def sequential_generatedRoot_204.w_0_0  (ctx : sequential_generatedRoot_204Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_204.w_3_0  (ctx : sequential_generatedRoot_204Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_204.w_1_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  if (sequential_generatedRoot_204.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_204.w_2_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  0

def sequential_generatedRoot_204.w_4_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_204.w_5_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_204.w_6_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  1

def sequential_generatedRoot_204.w_7_0  (ctx : sequential_generatedRoot_204Ctx) : Bool :=
  decide ((sequential_generatedRoot_204.w_5_0 ctx) = (sequential_generatedRoot_204.w_6_0 ctx))

def sequential_generatedRoot_204.w_8_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  if (sequential_generatedRoot_204.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_204.w_9_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  ((sequential_generatedRoot_204.w_1_0 ctx) + (sequential_generatedRoot_204.w_8_0 ctx))

def sequential_generatedRoot_204.w_10_0  (ctx : sequential_generatedRoot_204Ctx) : Int :=
  2

def sequential_generatedRoot_204.w_11_0  (ctx : sequential_generatedRoot_204Ctx) : Bool :=
  decide ((sequential_generatedRoot_204.w_9_0 ctx) = (sequential_generatedRoot_204.w_10_0 ctx))

def sequential_generatedRoot_204.outs  (ctx : sequential_generatedRoot_204Ctx) : Bool :=
  (sequential_generatedRoot_204.w_11_0 ctx)

structure sequential_generatedRoot_204.Facts_0  (ctx : sequential_generatedRoot_204Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_207 params i_0 i (current, ((sequential_generatedRoot_204.w_3_0 ctx), ((sequential_generatedRoot_204.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_204.w_2_0 ctx) (sequential_generatedRoot_204.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_204.outs ctx

theorem sequential_generatedRoot_204.facts_0  {ctx : sequential_generatedRoot_204Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_204.constraints_0 params i_0 (sequential_generatedRoot_204.w_3_0 ctx) (sequential_generatedRoot_204.w_2_0 ctx) (sequential_generatedRoot_204.w_4_0 ctx) (sequential_generatedRoot_204.w_5_0 ctx) (sequential_generatedRoot_204.w_11_0 ctx) outputs) :
    sequential_generatedRoot_204.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_204.Facts  (ctx : sequential_generatedRoot_204Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_204.Facts_0 ctx outputs

theorem sequential_generatedRoot_204.facts  (ctx : sequential_generatedRoot_204Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_204.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_204.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_204.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_206.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_206.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_206Ctx" => (sequential_generatedRoot_206.Ctx params i_0)

def sequential_generatedRoot_206.w_0_0  (ctx : sequential_generatedRoot_206Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_206.w_3_0  (ctx : sequential_generatedRoot_206Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_206.w_1_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  if (sequential_generatedRoot_206.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_206.w_2_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  0

def sequential_generatedRoot_206.w_4_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_206.w_5_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_206.w_6_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  1

def sequential_generatedRoot_206.w_7_0  (ctx : sequential_generatedRoot_206Ctx) : Bool :=
  decide ((sequential_generatedRoot_206.w_5_0 ctx) = (sequential_generatedRoot_206.w_6_0 ctx))

def sequential_generatedRoot_206.w_8_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  if (sequential_generatedRoot_206.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_206.w_9_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  ((sequential_generatedRoot_206.w_1_0 ctx) + (sequential_generatedRoot_206.w_8_0 ctx))

def sequential_generatedRoot_206.w_10_0  (ctx : sequential_generatedRoot_206Ctx) : Int :=
  2

def sequential_generatedRoot_206.w_11_0  (ctx : sequential_generatedRoot_206Ctx) : Bool :=
  decide ((sequential_generatedRoot_206.w_9_0 ctx) = (sequential_generatedRoot_206.w_10_0 ctx))

def sequential_generatedRoot_206.outs  (ctx : sequential_generatedRoot_206Ctx) : Bool :=
  (sequential_generatedRoot_206.w_11_0 ctx)

structure sequential_generatedRoot_206.Facts_0  (ctx : sequential_generatedRoot_206Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_208 params i_0 i (current, ((sequential_generatedRoot_206.w_3_0 ctx), ((sequential_generatedRoot_206.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_206.w_2_0 ctx) (sequential_generatedRoot_206.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_206.outs ctx

theorem sequential_generatedRoot_206.facts_0  {ctx : sequential_generatedRoot_206Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_206.constraints_0 params i_0 (sequential_generatedRoot_206.w_3_0 ctx) (sequential_generatedRoot_206.w_2_0 ctx) (sequential_generatedRoot_206.w_4_0 ctx) (sequential_generatedRoot_206.w_5_0 ctx) (sequential_generatedRoot_206.w_11_0 ctx) outputs) :
    sequential_generatedRoot_206.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_206.Facts  (ctx : sequential_generatedRoot_206Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_206.Facts_0 ctx outputs

theorem sequential_generatedRoot_206.facts  (ctx : sequential_generatedRoot_206Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_206.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_206.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_206.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_208.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8519680 → Int) × Unit
  witness : sequential_generatedRoot_208.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_208Ctx" => (sequential_generatedRoot_208.Ctx params i_0)

def sequential_generatedRoot_208.w_0_0  (ctx : sequential_generatedRoot_208Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_208.w_3_0  (ctx : sequential_generatedRoot_208Ctx) : Fin 8519680 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_208.w_1_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  if (sequential_generatedRoot_208.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_208.w_2_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  0

def sequential_generatedRoot_208.w_4_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_208.w_5_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_208.w_6_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  1

def sequential_generatedRoot_208.w_7_0  (ctx : sequential_generatedRoot_208Ctx) : Bool :=
  decide ((sequential_generatedRoot_208.w_5_0 ctx) = (sequential_generatedRoot_208.w_6_0 ctx))

def sequential_generatedRoot_208.w_8_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  if (sequential_generatedRoot_208.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_208.w_9_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  ((sequential_generatedRoot_208.w_1_0 ctx) + (sequential_generatedRoot_208.w_8_0 ctx))

def sequential_generatedRoot_208.w_10_0  (ctx : sequential_generatedRoot_208Ctx) : Int :=
  2

def sequential_generatedRoot_208.w_11_0  (ctx : sequential_generatedRoot_208Ctx) : Bool :=
  decide ((sequential_generatedRoot_208.w_9_0 ctx) = (sequential_generatedRoot_208.w_10_0 ctx))

def sequential_generatedRoot_208.outs  (ctx : sequential_generatedRoot_208Ctx) : Bool :=
  (sequential_generatedRoot_208.w_11_0 ctx)

structure sequential_generatedRoot_208.Facts_0  (ctx : sequential_generatedRoot_208Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_209 params i_0 i (current, ((sequential_generatedRoot_208.w_3_0 ctx), ((sequential_generatedRoot_208.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_208.w_2_0 ctx) (sequential_generatedRoot_208.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_208.outs ctx

theorem sequential_generatedRoot_208.facts_0  {ctx : sequential_generatedRoot_208Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_208.constraints_0 params i_0 (sequential_generatedRoot_208.w_3_0 ctx) (sequential_generatedRoot_208.w_2_0 ctx) (sequential_generatedRoot_208.w_4_0 ctx) (sequential_generatedRoot_208.w_5_0 ctx) (sequential_generatedRoot_208.w_11_0 ctx) outputs) :
    sequential_generatedRoot_208.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_208.Facts  (ctx : sequential_generatedRoot_208Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_208.Facts_0 ctx outputs

theorem sequential_generatedRoot_208.facts  (ctx : sequential_generatedRoot_208Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_208.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_208.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_208.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_210.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_210.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_210Ctx" => (sequential_generatedRoot_210.Ctx params i_0)

def sequential_generatedRoot_210.w_0_0  (ctx : sequential_generatedRoot_210Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_210.w_3_0  (ctx : sequential_generatedRoot_210Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_210.w_1_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  if (sequential_generatedRoot_210.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_210.w_2_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  0

def sequential_generatedRoot_210.w_4_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_210.w_5_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_210.w_6_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  1

def sequential_generatedRoot_210.w_7_0  (ctx : sequential_generatedRoot_210Ctx) : Bool :=
  decide ((sequential_generatedRoot_210.w_5_0 ctx) = (sequential_generatedRoot_210.w_6_0 ctx))

def sequential_generatedRoot_210.w_8_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  if (sequential_generatedRoot_210.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_210.w_9_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  ((sequential_generatedRoot_210.w_1_0 ctx) + (sequential_generatedRoot_210.w_8_0 ctx))

def sequential_generatedRoot_210.w_10_0  (ctx : sequential_generatedRoot_210Ctx) : Int :=
  2

def sequential_generatedRoot_210.w_11_0  (ctx : sequential_generatedRoot_210Ctx) : Bool :=
  decide ((sequential_generatedRoot_210.w_9_0 ctx) = (sequential_generatedRoot_210.w_10_0 ctx))

def sequential_generatedRoot_210.outs  (ctx : sequential_generatedRoot_210Ctx) : Bool :=
  (sequential_generatedRoot_210.w_11_0 ctx)

structure sequential_generatedRoot_210.Facts_0  (ctx : sequential_generatedRoot_210Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_210 params i_0 i (current, ((sequential_generatedRoot_210.w_3_0 ctx), ((sequential_generatedRoot_210.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_210.w_2_0 ctx) (sequential_generatedRoot_210.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_210.outs ctx

theorem sequential_generatedRoot_210.facts_0  {ctx : sequential_generatedRoot_210Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_210.constraints_0 params i_0 (sequential_generatedRoot_210.w_3_0 ctx) (sequential_generatedRoot_210.w_2_0 ctx) (sequential_generatedRoot_210.w_4_0 ctx) (sequential_generatedRoot_210.w_5_0 ctx) (sequential_generatedRoot_210.w_11_0 ctx) outputs) :
    sequential_generatedRoot_210.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_210.Facts  (ctx : sequential_generatedRoot_210Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_210.Facts_0 ctx outputs

theorem sequential_generatedRoot_210.facts  (ctx : sequential_generatedRoot_210Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_210.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_210.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_210.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_212.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_212.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_212Ctx" => (sequential_generatedRoot_212.Ctx params i_0)

def sequential_generatedRoot_212.w_0_0  (ctx : sequential_generatedRoot_212Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_212.w_3_0  (ctx : sequential_generatedRoot_212Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_212.w_1_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  if (sequential_generatedRoot_212.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_212.w_2_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  0

def sequential_generatedRoot_212.w_4_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_212.w_5_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_212.w_6_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  1

def sequential_generatedRoot_212.w_7_0  (ctx : sequential_generatedRoot_212Ctx) : Bool :=
  decide ((sequential_generatedRoot_212.w_5_0 ctx) = (sequential_generatedRoot_212.w_6_0 ctx))

def sequential_generatedRoot_212.w_8_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  if (sequential_generatedRoot_212.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_212.w_9_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  ((sequential_generatedRoot_212.w_1_0 ctx) + (sequential_generatedRoot_212.w_8_0 ctx))

def sequential_generatedRoot_212.w_10_0  (ctx : sequential_generatedRoot_212Ctx) : Int :=
  2

def sequential_generatedRoot_212.w_11_0  (ctx : sequential_generatedRoot_212Ctx) : Bool :=
  decide ((sequential_generatedRoot_212.w_9_0 ctx) = (sequential_generatedRoot_212.w_10_0 ctx))

def sequential_generatedRoot_212.outs  (ctx : sequential_generatedRoot_212Ctx) : Bool :=
  (sequential_generatedRoot_212.w_11_0 ctx)

structure sequential_generatedRoot_212.Facts_0  (ctx : sequential_generatedRoot_212Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_211 params i_0 i (current, ((sequential_generatedRoot_212.w_3_0 ctx), ((sequential_generatedRoot_212.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_212.w_2_0 ctx) (sequential_generatedRoot_212.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_212.outs ctx

theorem sequential_generatedRoot_212.facts_0  {ctx : sequential_generatedRoot_212Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_212.constraints_0 params i_0 (sequential_generatedRoot_212.w_3_0 ctx) (sequential_generatedRoot_212.w_2_0 ctx) (sequential_generatedRoot_212.w_4_0 ctx) (sequential_generatedRoot_212.w_5_0 ctx) (sequential_generatedRoot_212.w_11_0 ctx) outputs) :
    sequential_generatedRoot_212.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_212.Facts  (ctx : sequential_generatedRoot_212Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_212.Facts_0 ctx outputs

theorem sequential_generatedRoot_212.facts  (ctx : sequential_generatedRoot_212Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_212.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_212.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_212.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_214.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_214.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_214Ctx" => (sequential_generatedRoot_214.Ctx params i_0)

def sequential_generatedRoot_214.w_0_0  (ctx : sequential_generatedRoot_214Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_214.w_3_0  (ctx : sequential_generatedRoot_214Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_214.w_1_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  if (sequential_generatedRoot_214.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_214.w_2_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  0

def sequential_generatedRoot_214.w_4_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_214.w_5_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_214.w_6_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  1

def sequential_generatedRoot_214.w_7_0  (ctx : sequential_generatedRoot_214Ctx) : Bool :=
  decide ((sequential_generatedRoot_214.w_5_0 ctx) = (sequential_generatedRoot_214.w_6_0 ctx))

def sequential_generatedRoot_214.w_8_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  if (sequential_generatedRoot_214.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_214.w_9_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  ((sequential_generatedRoot_214.w_1_0 ctx) + (sequential_generatedRoot_214.w_8_0 ctx))

def sequential_generatedRoot_214.w_10_0  (ctx : sequential_generatedRoot_214Ctx) : Int :=
  2

def sequential_generatedRoot_214.w_11_0  (ctx : sequential_generatedRoot_214Ctx) : Bool :=
  decide ((sequential_generatedRoot_214.w_9_0 ctx) = (sequential_generatedRoot_214.w_10_0 ctx))

def sequential_generatedRoot_214.outs  (ctx : sequential_generatedRoot_214Ctx) : Bool :=
  (sequential_generatedRoot_214.w_11_0 ctx)

structure sequential_generatedRoot_214.Facts_0  (ctx : sequential_generatedRoot_214Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_212 params i_0 i (current, ((sequential_generatedRoot_214.w_3_0 ctx), ((sequential_generatedRoot_214.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_214.w_2_0 ctx) (sequential_generatedRoot_214.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_214.outs ctx

theorem sequential_generatedRoot_214.facts_0  {ctx : sequential_generatedRoot_214Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_214.constraints_0 params i_0 (sequential_generatedRoot_214.w_3_0 ctx) (sequential_generatedRoot_214.w_2_0 ctx) (sequential_generatedRoot_214.w_4_0 ctx) (sequential_generatedRoot_214.w_5_0 ctx) (sequential_generatedRoot_214.w_11_0 ctx) outputs) :
    sequential_generatedRoot_214.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_214.Facts  (ctx : sequential_generatedRoot_214Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_214.Facts_0 ctx outputs

theorem sequential_generatedRoot_214.facts  (ctx : sequential_generatedRoot_214Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_214.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_214.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_214.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_216.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_216.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_216Ctx" => (sequential_generatedRoot_216.Ctx params i_0)

def sequential_generatedRoot_216.w_0_0  (ctx : sequential_generatedRoot_216Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_216.w_3_0  (ctx : sequential_generatedRoot_216Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_216.w_1_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  if (sequential_generatedRoot_216.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_216.w_2_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  0

def sequential_generatedRoot_216.w_4_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_216.w_5_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_216.w_6_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  1

def sequential_generatedRoot_216.w_7_0  (ctx : sequential_generatedRoot_216Ctx) : Bool :=
  decide ((sequential_generatedRoot_216.w_5_0 ctx) = (sequential_generatedRoot_216.w_6_0 ctx))

def sequential_generatedRoot_216.w_8_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  if (sequential_generatedRoot_216.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_216.w_9_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  ((sequential_generatedRoot_216.w_1_0 ctx) + (sequential_generatedRoot_216.w_8_0 ctx))

def sequential_generatedRoot_216.w_10_0  (ctx : sequential_generatedRoot_216Ctx) : Int :=
  2

def sequential_generatedRoot_216.w_11_0  (ctx : sequential_generatedRoot_216Ctx) : Bool :=
  decide ((sequential_generatedRoot_216.w_9_0 ctx) = (sequential_generatedRoot_216.w_10_0 ctx))

def sequential_generatedRoot_216.outs  (ctx : sequential_generatedRoot_216Ctx) : Bool :=
  (sequential_generatedRoot_216.w_11_0 ctx)

structure sequential_generatedRoot_216.Facts_0  (ctx : sequential_generatedRoot_216Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_213 params i_0 i (current, ((sequential_generatedRoot_216.w_3_0 ctx), ((sequential_generatedRoot_216.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_216.w_2_0 ctx) (sequential_generatedRoot_216.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_216.outs ctx

theorem sequential_generatedRoot_216.facts_0  {ctx : sequential_generatedRoot_216Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_216.constraints_0 params i_0 (sequential_generatedRoot_216.w_3_0 ctx) (sequential_generatedRoot_216.w_2_0 ctx) (sequential_generatedRoot_216.w_4_0 ctx) (sequential_generatedRoot_216.w_5_0 ctx) (sequential_generatedRoot_216.w_11_0 ctx) outputs) :
    sequential_generatedRoot_216.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_216.Facts  (ctx : sequential_generatedRoot_216Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_216.Facts_0 ctx outputs

theorem sequential_generatedRoot_216.facts  (ctx : sequential_generatedRoot_216Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_216.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_216.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_216.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_218.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_218.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_218Ctx" => (sequential_generatedRoot_218.Ctx params i_0)

def sequential_generatedRoot_218.w_0_0  (ctx : sequential_generatedRoot_218Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_218.w_3_0  (ctx : sequential_generatedRoot_218Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_218.w_1_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  if (sequential_generatedRoot_218.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_218.w_2_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  0

def sequential_generatedRoot_218.w_4_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_218.w_5_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_218.w_6_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  1

def sequential_generatedRoot_218.w_7_0  (ctx : sequential_generatedRoot_218Ctx) : Bool :=
  decide ((sequential_generatedRoot_218.w_5_0 ctx) = (sequential_generatedRoot_218.w_6_0 ctx))

def sequential_generatedRoot_218.w_8_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  if (sequential_generatedRoot_218.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_218.w_9_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  ((sequential_generatedRoot_218.w_1_0 ctx) + (sequential_generatedRoot_218.w_8_0 ctx))

def sequential_generatedRoot_218.w_10_0  (ctx : sequential_generatedRoot_218Ctx) : Int :=
  2

def sequential_generatedRoot_218.w_11_0  (ctx : sequential_generatedRoot_218Ctx) : Bool :=
  decide ((sequential_generatedRoot_218.w_9_0 ctx) = (sequential_generatedRoot_218.w_10_0 ctx))

def sequential_generatedRoot_218.outs  (ctx : sequential_generatedRoot_218Ctx) : Bool :=
  (sequential_generatedRoot_218.w_11_0 ctx)

structure sequential_generatedRoot_218.Facts_0  (ctx : sequential_generatedRoot_218Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_214 params i_0 i (current, ((sequential_generatedRoot_218.w_3_0 ctx), ((sequential_generatedRoot_218.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_218.w_2_0 ctx) (sequential_generatedRoot_218.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_218.outs ctx

theorem sequential_generatedRoot_218.facts_0  {ctx : sequential_generatedRoot_218Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_218.constraints_0 params i_0 (sequential_generatedRoot_218.w_3_0 ctx) (sequential_generatedRoot_218.w_2_0 ctx) (sequential_generatedRoot_218.w_4_0 ctx) (sequential_generatedRoot_218.w_5_0 ctx) (sequential_generatedRoot_218.w_11_0 ctx) outputs) :
    sequential_generatedRoot_218.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_218.Facts  (ctx : sequential_generatedRoot_218Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_218.Facts_0 ctx outputs

theorem sequential_generatedRoot_218.facts  (ctx : sequential_generatedRoot_218Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_218.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_218.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_218.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_220.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_220.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_220Ctx" => (sequential_generatedRoot_220.Ctx params i_0)

def sequential_generatedRoot_220.w_0_0  (ctx : sequential_generatedRoot_220Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_220.w_3_0  (ctx : sequential_generatedRoot_220Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_220.w_1_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  if (sequential_generatedRoot_220.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_220.w_2_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  0

def sequential_generatedRoot_220.w_4_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_220.w_5_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_220.w_6_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  1

def sequential_generatedRoot_220.w_7_0  (ctx : sequential_generatedRoot_220Ctx) : Bool :=
  decide ((sequential_generatedRoot_220.w_5_0 ctx) = (sequential_generatedRoot_220.w_6_0 ctx))

def sequential_generatedRoot_220.w_8_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  if (sequential_generatedRoot_220.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_220.w_9_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  ((sequential_generatedRoot_220.w_1_0 ctx) + (sequential_generatedRoot_220.w_8_0 ctx))

def sequential_generatedRoot_220.w_10_0  (ctx : sequential_generatedRoot_220Ctx) : Int :=
  2

def sequential_generatedRoot_220.w_11_0  (ctx : sequential_generatedRoot_220Ctx) : Bool :=
  decide ((sequential_generatedRoot_220.w_9_0 ctx) = (sequential_generatedRoot_220.w_10_0 ctx))

def sequential_generatedRoot_220.outs  (ctx : sequential_generatedRoot_220Ctx) : Bool :=
  (sequential_generatedRoot_220.w_11_0 ctx)

structure sequential_generatedRoot_220.Facts_0  (ctx : sequential_generatedRoot_220Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_215 params i_0 i (current, ((sequential_generatedRoot_220.w_3_0 ctx), ((sequential_generatedRoot_220.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_220.w_2_0 ctx) (sequential_generatedRoot_220.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_220.outs ctx

theorem sequential_generatedRoot_220.facts_0  {ctx : sequential_generatedRoot_220Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_220.constraints_0 params i_0 (sequential_generatedRoot_220.w_3_0 ctx) (sequential_generatedRoot_220.w_2_0 ctx) (sequential_generatedRoot_220.w_4_0 ctx) (sequential_generatedRoot_220.w_5_0 ctx) (sequential_generatedRoot_220.w_11_0 ctx) outputs) :
    sequential_generatedRoot_220.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_220.Facts  (ctx : sequential_generatedRoot_220Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_220.Facts_0 ctx outputs

theorem sequential_generatedRoot_220.facts  (ctx : sequential_generatedRoot_220Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_220.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_220.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_220.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_222.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_222.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_222Ctx" => (sequential_generatedRoot_222.Ctx params i_0)

def sequential_generatedRoot_222.w_0_0  (ctx : sequential_generatedRoot_222Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_222.w_3_0  (ctx : sequential_generatedRoot_222Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_222.w_1_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  if (sequential_generatedRoot_222.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_222.w_2_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  0

def sequential_generatedRoot_222.w_4_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_222.w_5_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_222.w_6_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  1

def sequential_generatedRoot_222.w_7_0  (ctx : sequential_generatedRoot_222Ctx) : Bool :=
  decide ((sequential_generatedRoot_222.w_5_0 ctx) = (sequential_generatedRoot_222.w_6_0 ctx))

def sequential_generatedRoot_222.w_8_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  if (sequential_generatedRoot_222.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_222.w_9_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  ((sequential_generatedRoot_222.w_1_0 ctx) + (sequential_generatedRoot_222.w_8_0 ctx))

def sequential_generatedRoot_222.w_10_0  (ctx : sequential_generatedRoot_222Ctx) : Int :=
  2

def sequential_generatedRoot_222.w_11_0  (ctx : sequential_generatedRoot_222Ctx) : Bool :=
  decide ((sequential_generatedRoot_222.w_9_0 ctx) = (sequential_generatedRoot_222.w_10_0 ctx))

def sequential_generatedRoot_222.outs  (ctx : sequential_generatedRoot_222Ctx) : Bool :=
  (sequential_generatedRoot_222.w_11_0 ctx)

structure sequential_generatedRoot_222.Facts_0  (ctx : sequential_generatedRoot_222Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_216 params i_0 i (current, ((sequential_generatedRoot_222.w_3_0 ctx), ((sequential_generatedRoot_222.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_222.w_2_0 ctx) (sequential_generatedRoot_222.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_222.outs ctx

theorem sequential_generatedRoot_222.facts_0  {ctx : sequential_generatedRoot_222Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_222.constraints_0 params i_0 (sequential_generatedRoot_222.w_3_0 ctx) (sequential_generatedRoot_222.w_2_0 ctx) (sequential_generatedRoot_222.w_4_0 ctx) (sequential_generatedRoot_222.w_5_0 ctx) (sequential_generatedRoot_222.w_11_0 ctx) outputs) :
    sequential_generatedRoot_222.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_222.Facts  (ctx : sequential_generatedRoot_222Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_222.Facts_0 ctx outputs

theorem sequential_generatedRoot_222.facts  (ctx : sequential_generatedRoot_222Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_222.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_222.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_222.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_224.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_224.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_224Ctx" => (sequential_generatedRoot_224.Ctx params i_0)

def sequential_generatedRoot_224.w_0_0  (ctx : sequential_generatedRoot_224Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_224.w_3_0  (ctx : sequential_generatedRoot_224Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_224.w_1_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  if (sequential_generatedRoot_224.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_224.w_2_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  0

def sequential_generatedRoot_224.w_4_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_224.w_5_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_224.w_6_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  1

def sequential_generatedRoot_224.w_7_0  (ctx : sequential_generatedRoot_224Ctx) : Bool :=
  decide ((sequential_generatedRoot_224.w_5_0 ctx) = (sequential_generatedRoot_224.w_6_0 ctx))

def sequential_generatedRoot_224.w_8_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  if (sequential_generatedRoot_224.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_224.w_9_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  ((sequential_generatedRoot_224.w_1_0 ctx) + (sequential_generatedRoot_224.w_8_0 ctx))

def sequential_generatedRoot_224.w_10_0  (ctx : sequential_generatedRoot_224Ctx) : Int :=
  2

def sequential_generatedRoot_224.w_11_0  (ctx : sequential_generatedRoot_224Ctx) : Bool :=
  decide ((sequential_generatedRoot_224.w_9_0 ctx) = (sequential_generatedRoot_224.w_10_0 ctx))

def sequential_generatedRoot_224.outs  (ctx : sequential_generatedRoot_224Ctx) : Bool :=
  (sequential_generatedRoot_224.w_11_0 ctx)

structure sequential_generatedRoot_224.Facts_0  (ctx : sequential_generatedRoot_224Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_217 params i_0 i (current, ((sequential_generatedRoot_224.w_3_0 ctx), ((sequential_generatedRoot_224.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_224.w_2_0 ctx) (sequential_generatedRoot_224.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_224.outs ctx

theorem sequential_generatedRoot_224.facts_0  {ctx : sequential_generatedRoot_224Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_224.constraints_0 params i_0 (sequential_generatedRoot_224.w_3_0 ctx) (sequential_generatedRoot_224.w_2_0 ctx) (sequential_generatedRoot_224.w_4_0 ctx) (sequential_generatedRoot_224.w_5_0 ctx) (sequential_generatedRoot_224.w_11_0 ctx) outputs) :
    sequential_generatedRoot_224.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_224.Facts  (ctx : sequential_generatedRoot_224Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_224.Facts_0 ctx outputs

theorem sequential_generatedRoot_224.facts  (ctx : sequential_generatedRoot_224Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_224.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_224.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_224.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_226.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_226.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_226Ctx" => (sequential_generatedRoot_226.Ctx params i_0)

def sequential_generatedRoot_226.w_0_0  (ctx : sequential_generatedRoot_226Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_226.w_3_0  (ctx : sequential_generatedRoot_226Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_226.w_1_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  if (sequential_generatedRoot_226.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_226.w_2_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  0

def sequential_generatedRoot_226.w_4_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_226.w_5_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_226.w_6_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  1

def sequential_generatedRoot_226.w_7_0  (ctx : sequential_generatedRoot_226Ctx) : Bool :=
  decide ((sequential_generatedRoot_226.w_5_0 ctx) = (sequential_generatedRoot_226.w_6_0 ctx))

def sequential_generatedRoot_226.w_8_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  if (sequential_generatedRoot_226.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_226.w_9_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  ((sequential_generatedRoot_226.w_1_0 ctx) + (sequential_generatedRoot_226.w_8_0 ctx))

def sequential_generatedRoot_226.w_10_0  (ctx : sequential_generatedRoot_226Ctx) : Int :=
  2

def sequential_generatedRoot_226.w_11_0  (ctx : sequential_generatedRoot_226Ctx) : Bool :=
  decide ((sequential_generatedRoot_226.w_9_0 ctx) = (sequential_generatedRoot_226.w_10_0 ctx))

def sequential_generatedRoot_226.outs  (ctx : sequential_generatedRoot_226Ctx) : Bool :=
  (sequential_generatedRoot_226.w_11_0 ctx)

structure sequential_generatedRoot_226.Facts_0  (ctx : sequential_generatedRoot_226Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_218 params i_0 i (current, ((sequential_generatedRoot_226.w_3_0 ctx), ((sequential_generatedRoot_226.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_226.w_2_0 ctx) (sequential_generatedRoot_226.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_226.outs ctx

theorem sequential_generatedRoot_226.facts_0  {ctx : sequential_generatedRoot_226Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_226.constraints_0 params i_0 (sequential_generatedRoot_226.w_3_0 ctx) (sequential_generatedRoot_226.w_2_0 ctx) (sequential_generatedRoot_226.w_4_0 ctx) (sequential_generatedRoot_226.w_5_0 ctx) (sequential_generatedRoot_226.w_11_0 ctx) outputs) :
    sequential_generatedRoot_226.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_226.Facts  (ctx : sequential_generatedRoot_226Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_226.Facts_0 ctx outputs

theorem sequential_generatedRoot_226.facts  (ctx : sequential_generatedRoot_226Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_226.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_226.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_226.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_228.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_228.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_228Ctx" => (sequential_generatedRoot_228.Ctx params i_0)

def sequential_generatedRoot_228.w_0_0  (ctx : sequential_generatedRoot_228Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_228.w_3_0  (ctx : sequential_generatedRoot_228Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_228.w_1_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  if (sequential_generatedRoot_228.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_228.w_2_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  0

def sequential_generatedRoot_228.w_4_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_228.w_5_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_228.w_6_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  1

def sequential_generatedRoot_228.w_7_0  (ctx : sequential_generatedRoot_228Ctx) : Bool :=
  decide ((sequential_generatedRoot_228.w_5_0 ctx) = (sequential_generatedRoot_228.w_6_0 ctx))

def sequential_generatedRoot_228.w_8_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  if (sequential_generatedRoot_228.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_228.w_9_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  ((sequential_generatedRoot_228.w_1_0 ctx) + (sequential_generatedRoot_228.w_8_0 ctx))

def sequential_generatedRoot_228.w_10_0  (ctx : sequential_generatedRoot_228Ctx) : Int :=
  2

def sequential_generatedRoot_228.w_11_0  (ctx : sequential_generatedRoot_228Ctx) : Bool :=
  decide ((sequential_generatedRoot_228.w_9_0 ctx) = (sequential_generatedRoot_228.w_10_0 ctx))

def sequential_generatedRoot_228.outs  (ctx : sequential_generatedRoot_228Ctx) : Bool :=
  (sequential_generatedRoot_228.w_11_0 ctx)

structure sequential_generatedRoot_228.Facts_0  (ctx : sequential_generatedRoot_228Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_219 params i_0 i (current, ((sequential_generatedRoot_228.w_3_0 ctx), ((sequential_generatedRoot_228.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_228.w_2_0 ctx) (sequential_generatedRoot_228.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_228.outs ctx

theorem sequential_generatedRoot_228.facts_0  {ctx : sequential_generatedRoot_228Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_228.constraints_0 params i_0 (sequential_generatedRoot_228.w_3_0 ctx) (sequential_generatedRoot_228.w_2_0 ctx) (sequential_generatedRoot_228.w_4_0 ctx) (sequential_generatedRoot_228.w_5_0 ctx) (sequential_generatedRoot_228.w_11_0 ctx) outputs) :
    sequential_generatedRoot_228.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_228.Facts  (ctx : sequential_generatedRoot_228Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_228.Facts_0 ctx outputs

theorem sequential_generatedRoot_228.facts  (ctx : sequential_generatedRoot_228Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_228.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_228.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_228.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_230.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_230.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_230Ctx" => (sequential_generatedRoot_230.Ctx params i_0)

def sequential_generatedRoot_230.w_0_0  (ctx : sequential_generatedRoot_230Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_230.w_3_0  (ctx : sequential_generatedRoot_230Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_230.w_1_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  if (sequential_generatedRoot_230.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_230.w_2_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  0

def sequential_generatedRoot_230.w_4_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_230.w_5_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_230.w_6_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  1

def sequential_generatedRoot_230.w_7_0  (ctx : sequential_generatedRoot_230Ctx) : Bool :=
  decide ((sequential_generatedRoot_230.w_5_0 ctx) = (sequential_generatedRoot_230.w_6_0 ctx))

def sequential_generatedRoot_230.w_8_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  if (sequential_generatedRoot_230.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_230.w_9_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  ((sequential_generatedRoot_230.w_1_0 ctx) + (sequential_generatedRoot_230.w_8_0 ctx))

def sequential_generatedRoot_230.w_10_0  (ctx : sequential_generatedRoot_230Ctx) : Int :=
  2

def sequential_generatedRoot_230.w_11_0  (ctx : sequential_generatedRoot_230Ctx) : Bool :=
  decide ((sequential_generatedRoot_230.w_9_0 ctx) = (sequential_generatedRoot_230.w_10_0 ctx))

def sequential_generatedRoot_230.outs  (ctx : sequential_generatedRoot_230Ctx) : Bool :=
  (sequential_generatedRoot_230.w_11_0 ctx)

structure sequential_generatedRoot_230.Facts_0  (ctx : sequential_generatedRoot_230Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_220 params i_0 i (current, ((sequential_generatedRoot_230.w_3_0 ctx), ((sequential_generatedRoot_230.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_230.w_2_0 ctx) (sequential_generatedRoot_230.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_230.outs ctx

theorem sequential_generatedRoot_230.facts_0  {ctx : sequential_generatedRoot_230Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_230.constraints_0 params i_0 (sequential_generatedRoot_230.w_3_0 ctx) (sequential_generatedRoot_230.w_2_0 ctx) (sequential_generatedRoot_230.w_4_0 ctx) (sequential_generatedRoot_230.w_5_0 ctx) (sequential_generatedRoot_230.w_11_0 ctx) outputs) :
    sequential_generatedRoot_230.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_230.Facts  (ctx : sequential_generatedRoot_230Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_230.Facts_0 ctx outputs

theorem sequential_generatedRoot_230.facts  (ctx : sequential_generatedRoot_230Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_230.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_230.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_230.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_232.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_232.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_232Ctx" => (sequential_generatedRoot_232.Ctx params i_0)

def sequential_generatedRoot_232.w_0_0  (ctx : sequential_generatedRoot_232Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_232.w_3_0  (ctx : sequential_generatedRoot_232Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_232.w_1_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  if (sequential_generatedRoot_232.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_232.w_2_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  0

def sequential_generatedRoot_232.w_4_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_232.w_5_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_232.w_6_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  1

def sequential_generatedRoot_232.w_7_0  (ctx : sequential_generatedRoot_232Ctx) : Bool :=
  decide ((sequential_generatedRoot_232.w_5_0 ctx) = (sequential_generatedRoot_232.w_6_0 ctx))

def sequential_generatedRoot_232.w_8_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  if (sequential_generatedRoot_232.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_232.w_9_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  ((sequential_generatedRoot_232.w_1_0 ctx) + (sequential_generatedRoot_232.w_8_0 ctx))

def sequential_generatedRoot_232.w_10_0  (ctx : sequential_generatedRoot_232Ctx) : Int :=
  2

def sequential_generatedRoot_232.w_11_0  (ctx : sequential_generatedRoot_232Ctx) : Bool :=
  decide ((sequential_generatedRoot_232.w_9_0 ctx) = (sequential_generatedRoot_232.w_10_0 ctx))

def sequential_generatedRoot_232.outs  (ctx : sequential_generatedRoot_232Ctx) : Bool :=
  (sequential_generatedRoot_232.w_11_0 ctx)

structure sequential_generatedRoot_232.Facts_0  (ctx : sequential_generatedRoot_232Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_221 params i_0 i (current, ((sequential_generatedRoot_232.w_3_0 ctx), ((sequential_generatedRoot_232.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_232.w_2_0 ctx) (sequential_generatedRoot_232.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_232.outs ctx

theorem sequential_generatedRoot_232.facts_0  {ctx : sequential_generatedRoot_232Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_232.constraints_0 params i_0 (sequential_generatedRoot_232.w_3_0 ctx) (sequential_generatedRoot_232.w_2_0 ctx) (sequential_generatedRoot_232.w_4_0 ctx) (sequential_generatedRoot_232.w_5_0 ctx) (sequential_generatedRoot_232.w_11_0 ctx) outputs) :
    sequential_generatedRoot_232.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_232.Facts  (ctx : sequential_generatedRoot_232Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_232.Facts_0 ctx outputs

theorem sequential_generatedRoot_232.facts  (ctx : sequential_generatedRoot_232Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_232.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_232.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_232.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_234.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_234.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_234Ctx" => (sequential_generatedRoot_234.Ctx params i_0)

def sequential_generatedRoot_234.w_0_0  (ctx : sequential_generatedRoot_234Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_234.w_3_0  (ctx : sequential_generatedRoot_234Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_234.w_1_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  if (sequential_generatedRoot_234.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_234.w_2_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  0

def sequential_generatedRoot_234.w_4_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_234.w_5_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_234.w_6_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  1

def sequential_generatedRoot_234.w_7_0  (ctx : sequential_generatedRoot_234Ctx) : Bool :=
  decide ((sequential_generatedRoot_234.w_5_0 ctx) = (sequential_generatedRoot_234.w_6_0 ctx))

def sequential_generatedRoot_234.w_8_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  if (sequential_generatedRoot_234.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_234.w_9_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  ((sequential_generatedRoot_234.w_1_0 ctx) + (sequential_generatedRoot_234.w_8_0 ctx))

def sequential_generatedRoot_234.w_10_0  (ctx : sequential_generatedRoot_234Ctx) : Int :=
  2

def sequential_generatedRoot_234.w_11_0  (ctx : sequential_generatedRoot_234Ctx) : Bool :=
  decide ((sequential_generatedRoot_234.w_9_0 ctx) = (sequential_generatedRoot_234.w_10_0 ctx))

def sequential_generatedRoot_234.outs  (ctx : sequential_generatedRoot_234Ctx) : Bool :=
  (sequential_generatedRoot_234.w_11_0 ctx)

structure sequential_generatedRoot_234.Facts_0  (ctx : sequential_generatedRoot_234Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_222 params i_0 i (current, ((sequential_generatedRoot_234.w_3_0 ctx), ((sequential_generatedRoot_234.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_234.w_2_0 ctx) (sequential_generatedRoot_234.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_234.outs ctx

theorem sequential_generatedRoot_234.facts_0  {ctx : sequential_generatedRoot_234Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_234.constraints_0 params i_0 (sequential_generatedRoot_234.w_3_0 ctx) (sequential_generatedRoot_234.w_2_0 ctx) (sequential_generatedRoot_234.w_4_0 ctx) (sequential_generatedRoot_234.w_5_0 ctx) (sequential_generatedRoot_234.w_11_0 ctx) outputs) :
    sequential_generatedRoot_234.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_234.Facts  (ctx : sequential_generatedRoot_234Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_234.Facts_0 ctx outputs

theorem sequential_generatedRoot_234.facts  (ctx : sequential_generatedRoot_234Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_234.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_234.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_234.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_236.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_236.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_236Ctx" => (sequential_generatedRoot_236.Ctx params i_0)

def sequential_generatedRoot_236.w_0_0  (ctx : sequential_generatedRoot_236Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_236.w_3_0  (ctx : sequential_generatedRoot_236Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_236.w_1_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  if (sequential_generatedRoot_236.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_236.w_2_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  0

def sequential_generatedRoot_236.w_4_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_236.w_5_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_236.w_6_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  1

def sequential_generatedRoot_236.w_7_0  (ctx : sequential_generatedRoot_236Ctx) : Bool :=
  decide ((sequential_generatedRoot_236.w_5_0 ctx) = (sequential_generatedRoot_236.w_6_0 ctx))

def sequential_generatedRoot_236.w_8_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  if (sequential_generatedRoot_236.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_236.w_9_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  ((sequential_generatedRoot_236.w_1_0 ctx) + (sequential_generatedRoot_236.w_8_0 ctx))

def sequential_generatedRoot_236.w_10_0  (ctx : sequential_generatedRoot_236Ctx) : Int :=
  2

def sequential_generatedRoot_236.w_11_0  (ctx : sequential_generatedRoot_236Ctx) : Bool :=
  decide ((sequential_generatedRoot_236.w_9_0 ctx) = (sequential_generatedRoot_236.w_10_0 ctx))

def sequential_generatedRoot_236.outs  (ctx : sequential_generatedRoot_236Ctx) : Bool :=
  (sequential_generatedRoot_236.w_11_0 ctx)

structure sequential_generatedRoot_236.Facts_0  (ctx : sequential_generatedRoot_236Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_223 params i_0 i (current, ((sequential_generatedRoot_236.w_3_0 ctx), ((sequential_generatedRoot_236.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_236.w_2_0 ctx) (sequential_generatedRoot_236.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_236.outs ctx

theorem sequential_generatedRoot_236.facts_0  {ctx : sequential_generatedRoot_236Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_236.constraints_0 params i_0 (sequential_generatedRoot_236.w_3_0 ctx) (sequential_generatedRoot_236.w_2_0 ctx) (sequential_generatedRoot_236.w_4_0 ctx) (sequential_generatedRoot_236.w_5_0 ctx) (sequential_generatedRoot_236.w_11_0 ctx) outputs) :
    sequential_generatedRoot_236.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_236.Facts  (ctx : sequential_generatedRoot_236Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_236.Facts_0 ctx outputs

theorem sequential_generatedRoot_236.facts  (ctx : sequential_generatedRoot_236Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_236.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_236.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_236.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_238.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_238.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_238Ctx" => (sequential_generatedRoot_238.Ctx params i_0)

def sequential_generatedRoot_238.w_0_0  (ctx : sequential_generatedRoot_238Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_238.w_3_0  (ctx : sequential_generatedRoot_238Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_238.w_1_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  if (sequential_generatedRoot_238.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_238.w_2_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  0

def sequential_generatedRoot_238.w_4_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_238.w_5_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_238.w_6_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  1

def sequential_generatedRoot_238.w_7_0  (ctx : sequential_generatedRoot_238Ctx) : Bool :=
  decide ((sequential_generatedRoot_238.w_5_0 ctx) = (sequential_generatedRoot_238.w_6_0 ctx))

def sequential_generatedRoot_238.w_8_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  if (sequential_generatedRoot_238.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_238.w_9_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  ((sequential_generatedRoot_238.w_1_0 ctx) + (sequential_generatedRoot_238.w_8_0 ctx))

def sequential_generatedRoot_238.w_10_0  (ctx : sequential_generatedRoot_238Ctx) : Int :=
  2

def sequential_generatedRoot_238.w_11_0  (ctx : sequential_generatedRoot_238Ctx) : Bool :=
  decide ((sequential_generatedRoot_238.w_9_0 ctx) = (sequential_generatedRoot_238.w_10_0 ctx))

def sequential_generatedRoot_238.outs  (ctx : sequential_generatedRoot_238Ctx) : Bool :=
  (sequential_generatedRoot_238.w_11_0 ctx)

structure sequential_generatedRoot_238.Facts_0  (ctx : sequential_generatedRoot_238Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_224 params i_0 i (current, ((sequential_generatedRoot_238.w_3_0 ctx), ((sequential_generatedRoot_238.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_238.w_2_0 ctx) (sequential_generatedRoot_238.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_238.outs ctx

theorem sequential_generatedRoot_238.facts_0  {ctx : sequential_generatedRoot_238Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_238.constraints_0 params i_0 (sequential_generatedRoot_238.w_3_0 ctx) (sequential_generatedRoot_238.w_2_0 ctx) (sequential_generatedRoot_238.w_4_0 ctx) (sequential_generatedRoot_238.w_5_0 ctx) (sequential_generatedRoot_238.w_11_0 ctx) outputs) :
    sequential_generatedRoot_238.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_238.Facts  (ctx : sequential_generatedRoot_238Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_238.Facts_0 ctx outputs

theorem sequential_generatedRoot_238.facts  (ctx : sequential_generatedRoot_238Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_238.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_238.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_238.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_240.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_240.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_240Ctx" => (sequential_generatedRoot_240.Ctx params i_0)

def sequential_generatedRoot_240.w_0_0  (ctx : sequential_generatedRoot_240Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_240.w_3_0  (ctx : sequential_generatedRoot_240Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_240.w_1_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  if (sequential_generatedRoot_240.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_240.w_2_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  0

def sequential_generatedRoot_240.w_4_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_240.w_5_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_240.w_6_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  1

def sequential_generatedRoot_240.w_7_0  (ctx : sequential_generatedRoot_240Ctx) : Bool :=
  decide ((sequential_generatedRoot_240.w_5_0 ctx) = (sequential_generatedRoot_240.w_6_0 ctx))

def sequential_generatedRoot_240.w_8_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  if (sequential_generatedRoot_240.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_240.w_9_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  ((sequential_generatedRoot_240.w_1_0 ctx) + (sequential_generatedRoot_240.w_8_0 ctx))

def sequential_generatedRoot_240.w_10_0  (ctx : sequential_generatedRoot_240Ctx) : Int :=
  2

def sequential_generatedRoot_240.w_11_0  (ctx : sequential_generatedRoot_240Ctx) : Bool :=
  decide ((sequential_generatedRoot_240.w_9_0 ctx) = (sequential_generatedRoot_240.w_10_0 ctx))

def sequential_generatedRoot_240.outs  (ctx : sequential_generatedRoot_240Ctx) : Bool :=
  (sequential_generatedRoot_240.w_11_0 ctx)

structure sequential_generatedRoot_240.Facts_0  (ctx : sequential_generatedRoot_240Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_225 params i_0 i (current, ((sequential_generatedRoot_240.w_3_0 ctx), ((sequential_generatedRoot_240.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_240.w_2_0 ctx) (sequential_generatedRoot_240.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_240.outs ctx

theorem sequential_generatedRoot_240.facts_0  {ctx : sequential_generatedRoot_240Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_240.constraints_0 params i_0 (sequential_generatedRoot_240.w_3_0 ctx) (sequential_generatedRoot_240.w_2_0 ctx) (sequential_generatedRoot_240.w_4_0 ctx) (sequential_generatedRoot_240.w_5_0 ctx) (sequential_generatedRoot_240.w_11_0 ctx) outputs) :
    sequential_generatedRoot_240.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_240.Facts  (ctx : sequential_generatedRoot_240Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_240.Facts_0 ctx outputs

theorem sequential_generatedRoot_240.facts  (ctx : sequential_generatedRoot_240Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_240.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_240.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_240.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_242.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_242.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_242Ctx" => (sequential_generatedRoot_242.Ctx params i_0)

def sequential_generatedRoot_242.w_0_0  (ctx : sequential_generatedRoot_242Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_242.w_3_0  (ctx : sequential_generatedRoot_242Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_242.w_1_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  if (sequential_generatedRoot_242.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_242.w_2_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  0

def sequential_generatedRoot_242.w_4_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_242.w_5_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_242.w_6_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  1

def sequential_generatedRoot_242.w_7_0  (ctx : sequential_generatedRoot_242Ctx) : Bool :=
  decide ((sequential_generatedRoot_242.w_5_0 ctx) = (sequential_generatedRoot_242.w_6_0 ctx))

def sequential_generatedRoot_242.w_8_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  if (sequential_generatedRoot_242.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_242.w_9_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  ((sequential_generatedRoot_242.w_1_0 ctx) + (sequential_generatedRoot_242.w_8_0 ctx))

def sequential_generatedRoot_242.w_10_0  (ctx : sequential_generatedRoot_242Ctx) : Int :=
  2

def sequential_generatedRoot_242.w_11_0  (ctx : sequential_generatedRoot_242Ctx) : Bool :=
  decide ((sequential_generatedRoot_242.w_9_0 ctx) = (sequential_generatedRoot_242.w_10_0 ctx))

def sequential_generatedRoot_242.outs  (ctx : sequential_generatedRoot_242Ctx) : Bool :=
  (sequential_generatedRoot_242.w_11_0 ctx)

structure sequential_generatedRoot_242.Facts_0  (ctx : sequential_generatedRoot_242Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_226 params i_0 i (current, ((sequential_generatedRoot_242.w_3_0 ctx), ((sequential_generatedRoot_242.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_242.w_2_0 ctx) (sequential_generatedRoot_242.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_242.outs ctx

theorem sequential_generatedRoot_242.facts_0  {ctx : sequential_generatedRoot_242Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_242.constraints_0 params i_0 (sequential_generatedRoot_242.w_3_0 ctx) (sequential_generatedRoot_242.w_2_0 ctx) (sequential_generatedRoot_242.w_4_0 ctx) (sequential_generatedRoot_242.w_5_0 ctx) (sequential_generatedRoot_242.w_11_0 ctx) outputs) :
    sequential_generatedRoot_242.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_242.Facts  (ctx : sequential_generatedRoot_242Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_242.Facts_0 ctx outputs

theorem sequential_generatedRoot_242.facts  (ctx : sequential_generatedRoot_242Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_242.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_242.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_242.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_244.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_244.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_244Ctx" => (sequential_generatedRoot_244.Ctx params i_0)

def sequential_generatedRoot_244.w_0_0  (ctx : sequential_generatedRoot_244Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_244.w_3_0  (ctx : sequential_generatedRoot_244Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_244.w_1_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  if (sequential_generatedRoot_244.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_244.w_2_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  0

def sequential_generatedRoot_244.w_4_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_244.w_5_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_244.w_6_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  1

def sequential_generatedRoot_244.w_7_0  (ctx : sequential_generatedRoot_244Ctx) : Bool :=
  decide ((sequential_generatedRoot_244.w_5_0 ctx) = (sequential_generatedRoot_244.w_6_0 ctx))

def sequential_generatedRoot_244.w_8_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  if (sequential_generatedRoot_244.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_244.w_9_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  ((sequential_generatedRoot_244.w_1_0 ctx) + (sequential_generatedRoot_244.w_8_0 ctx))

def sequential_generatedRoot_244.w_10_0  (ctx : sequential_generatedRoot_244Ctx) : Int :=
  2

def sequential_generatedRoot_244.w_11_0  (ctx : sequential_generatedRoot_244Ctx) : Bool :=
  decide ((sequential_generatedRoot_244.w_9_0 ctx) = (sequential_generatedRoot_244.w_10_0 ctx))

def sequential_generatedRoot_244.outs  (ctx : sequential_generatedRoot_244Ctx) : Bool :=
  (sequential_generatedRoot_244.w_11_0 ctx)

structure sequential_generatedRoot_244.Facts_0  (ctx : sequential_generatedRoot_244Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_227 params i_0 i (current, ((sequential_generatedRoot_244.w_3_0 ctx), ((sequential_generatedRoot_244.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_244.w_2_0 ctx) (sequential_generatedRoot_244.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_244.outs ctx

theorem sequential_generatedRoot_244.facts_0  {ctx : sequential_generatedRoot_244Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_244.constraints_0 params i_0 (sequential_generatedRoot_244.w_3_0 ctx) (sequential_generatedRoot_244.w_2_0 ctx) (sequential_generatedRoot_244.w_4_0 ctx) (sequential_generatedRoot_244.w_5_0 ctx) (sequential_generatedRoot_244.w_11_0 ctx) outputs) :
    sequential_generatedRoot_244.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_244.Facts  (ctx : sequential_generatedRoot_244Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_244.Facts_0 ctx outputs

theorem sequential_generatedRoot_244.facts  (ctx : sequential_generatedRoot_244Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_244.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_244.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_244.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_246.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_246.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_246Ctx" => (sequential_generatedRoot_246.Ctx params i_0)

def sequential_generatedRoot_246.w_0_0  (ctx : sequential_generatedRoot_246Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_246.w_3_0  (ctx : sequential_generatedRoot_246Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_246.w_1_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  if (sequential_generatedRoot_246.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_246.w_2_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  0

def sequential_generatedRoot_246.w_4_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_246.w_5_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_246.w_6_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  1

def sequential_generatedRoot_246.w_7_0  (ctx : sequential_generatedRoot_246Ctx) : Bool :=
  decide ((sequential_generatedRoot_246.w_5_0 ctx) = (sequential_generatedRoot_246.w_6_0 ctx))

def sequential_generatedRoot_246.w_8_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  if (sequential_generatedRoot_246.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_246.w_9_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  ((sequential_generatedRoot_246.w_1_0 ctx) + (sequential_generatedRoot_246.w_8_0 ctx))

def sequential_generatedRoot_246.w_10_0  (ctx : sequential_generatedRoot_246Ctx) : Int :=
  2

def sequential_generatedRoot_246.w_11_0  (ctx : sequential_generatedRoot_246Ctx) : Bool :=
  decide ((sequential_generatedRoot_246.w_9_0 ctx) = (sequential_generatedRoot_246.w_10_0 ctx))

def sequential_generatedRoot_246.outs  (ctx : sequential_generatedRoot_246Ctx) : Bool :=
  (sequential_generatedRoot_246.w_11_0 ctx)

structure sequential_generatedRoot_246.Facts_0  (ctx : sequential_generatedRoot_246Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_228 params i_0 i (current, ((sequential_generatedRoot_246.w_3_0 ctx), ((sequential_generatedRoot_246.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_246.w_2_0 ctx) (sequential_generatedRoot_246.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_246.outs ctx

theorem sequential_generatedRoot_246.facts_0  {ctx : sequential_generatedRoot_246Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_246.constraints_0 params i_0 (sequential_generatedRoot_246.w_3_0 ctx) (sequential_generatedRoot_246.w_2_0 ctx) (sequential_generatedRoot_246.w_4_0 ctx) (sequential_generatedRoot_246.w_5_0 ctx) (sequential_generatedRoot_246.w_11_0 ctx) outputs) :
    sequential_generatedRoot_246.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_246.Facts  (ctx : sequential_generatedRoot_246Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_246.Facts_0 ctx outputs

theorem sequential_generatedRoot_246.facts  (ctx : sequential_generatedRoot_246Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_246.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_246.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_246.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_248.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_248.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_248Ctx" => (sequential_generatedRoot_248.Ctx params i_0)

def sequential_generatedRoot_248.w_0_0  (ctx : sequential_generatedRoot_248Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_248.w_3_0  (ctx : sequential_generatedRoot_248Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_248.w_1_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  if (sequential_generatedRoot_248.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_248.w_2_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  0

def sequential_generatedRoot_248.w_4_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_248.w_5_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_248.w_6_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  1

def sequential_generatedRoot_248.w_7_0  (ctx : sequential_generatedRoot_248Ctx) : Bool :=
  decide ((sequential_generatedRoot_248.w_5_0 ctx) = (sequential_generatedRoot_248.w_6_0 ctx))

def sequential_generatedRoot_248.w_8_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  if (sequential_generatedRoot_248.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_248.w_9_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  ((sequential_generatedRoot_248.w_1_0 ctx) + (sequential_generatedRoot_248.w_8_0 ctx))

def sequential_generatedRoot_248.w_10_0  (ctx : sequential_generatedRoot_248Ctx) : Int :=
  2

def sequential_generatedRoot_248.w_11_0  (ctx : sequential_generatedRoot_248Ctx) : Bool :=
  decide ((sequential_generatedRoot_248.w_9_0 ctx) = (sequential_generatedRoot_248.w_10_0 ctx))

def sequential_generatedRoot_248.outs  (ctx : sequential_generatedRoot_248Ctx) : Bool :=
  (sequential_generatedRoot_248.w_11_0 ctx)

structure sequential_generatedRoot_248.Facts_0  (ctx : sequential_generatedRoot_248Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_229 params i_0 i (current, ((sequential_generatedRoot_248.w_3_0 ctx), ((sequential_generatedRoot_248.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_248.w_2_0 ctx) (sequential_generatedRoot_248.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_248.outs ctx

theorem sequential_generatedRoot_248.facts_0  {ctx : sequential_generatedRoot_248Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_248.constraints_0 params i_0 (sequential_generatedRoot_248.w_3_0 ctx) (sequential_generatedRoot_248.w_2_0 ctx) (sequential_generatedRoot_248.w_4_0 ctx) (sequential_generatedRoot_248.w_5_0 ctx) (sequential_generatedRoot_248.w_11_0 ctx) outputs) :
    sequential_generatedRoot_248.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_248.Facts  (ctx : sequential_generatedRoot_248Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_248.Facts_0 ctx outputs

theorem sequential_generatedRoot_248.facts  (ctx : sequential_generatedRoot_248Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_248.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_248.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_248.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_250.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_250.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_250Ctx" => (sequential_generatedRoot_250.Ctx params i_0)

def sequential_generatedRoot_250.w_0_0  (ctx : sequential_generatedRoot_250Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_250.w_3_0  (ctx : sequential_generatedRoot_250Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_250.w_1_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  if (sequential_generatedRoot_250.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_250.w_2_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  0

def sequential_generatedRoot_250.w_4_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_250.w_5_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_250.w_6_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  1

def sequential_generatedRoot_250.w_7_0  (ctx : sequential_generatedRoot_250Ctx) : Bool :=
  decide ((sequential_generatedRoot_250.w_5_0 ctx) = (sequential_generatedRoot_250.w_6_0 ctx))

def sequential_generatedRoot_250.w_8_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  if (sequential_generatedRoot_250.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_250.w_9_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  ((sequential_generatedRoot_250.w_1_0 ctx) + (sequential_generatedRoot_250.w_8_0 ctx))

def sequential_generatedRoot_250.w_10_0  (ctx : sequential_generatedRoot_250Ctx) : Int :=
  2

def sequential_generatedRoot_250.w_11_0  (ctx : sequential_generatedRoot_250Ctx) : Bool :=
  decide ((sequential_generatedRoot_250.w_9_0 ctx) = (sequential_generatedRoot_250.w_10_0 ctx))

def sequential_generatedRoot_250.outs  (ctx : sequential_generatedRoot_250Ctx) : Bool :=
  (sequential_generatedRoot_250.w_11_0 ctx)

structure sequential_generatedRoot_250.Facts_0  (ctx : sequential_generatedRoot_250Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_230 params i_0 i (current, ((sequential_generatedRoot_250.w_3_0 ctx), ((sequential_generatedRoot_250.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_250.w_2_0 ctx) (sequential_generatedRoot_250.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_250.outs ctx

theorem sequential_generatedRoot_250.facts_0  {ctx : sequential_generatedRoot_250Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_250.constraints_0 params i_0 (sequential_generatedRoot_250.w_3_0 ctx) (sequential_generatedRoot_250.w_2_0 ctx) (sequential_generatedRoot_250.w_4_0 ctx) (sequential_generatedRoot_250.w_5_0 ctx) (sequential_generatedRoot_250.w_11_0 ctx) outputs) :
    sequential_generatedRoot_250.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_250.Facts  (ctx : sequential_generatedRoot_250Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_250.Facts_0 ctx outputs

theorem sequential_generatedRoot_250.facts  (ctx : sequential_generatedRoot_250Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_250.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_250.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_250.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_252.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_252.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_252Ctx" => (sequential_generatedRoot_252.Ctx params i_0)

def sequential_generatedRoot_252.w_0_0  (ctx : sequential_generatedRoot_252Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_252.w_3_0  (ctx : sequential_generatedRoot_252Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_252.w_1_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  if (sequential_generatedRoot_252.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_252.w_2_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  0

def sequential_generatedRoot_252.w_4_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_252.w_5_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_252.w_6_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  1

def sequential_generatedRoot_252.w_7_0  (ctx : sequential_generatedRoot_252Ctx) : Bool :=
  decide ((sequential_generatedRoot_252.w_5_0 ctx) = (sequential_generatedRoot_252.w_6_0 ctx))

def sequential_generatedRoot_252.w_8_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  if (sequential_generatedRoot_252.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_252.w_9_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  ((sequential_generatedRoot_252.w_1_0 ctx) + (sequential_generatedRoot_252.w_8_0 ctx))

def sequential_generatedRoot_252.w_10_0  (ctx : sequential_generatedRoot_252Ctx) : Int :=
  2

def sequential_generatedRoot_252.w_11_0  (ctx : sequential_generatedRoot_252Ctx) : Bool :=
  decide ((sequential_generatedRoot_252.w_9_0 ctx) = (sequential_generatedRoot_252.w_10_0 ctx))

def sequential_generatedRoot_252.outs  (ctx : sequential_generatedRoot_252Ctx) : Bool :=
  (sequential_generatedRoot_252.w_11_0 ctx)

structure sequential_generatedRoot_252.Facts_0  (ctx : sequential_generatedRoot_252Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_231 params i_0 i (current, ((sequential_generatedRoot_252.w_3_0 ctx), ((sequential_generatedRoot_252.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_252.w_2_0 ctx) (sequential_generatedRoot_252.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_252.outs ctx

theorem sequential_generatedRoot_252.facts_0  {ctx : sequential_generatedRoot_252Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_252.constraints_0 params i_0 (sequential_generatedRoot_252.w_3_0 ctx) (sequential_generatedRoot_252.w_2_0 ctx) (sequential_generatedRoot_252.w_4_0 ctx) (sequential_generatedRoot_252.w_5_0 ctx) (sequential_generatedRoot_252.w_11_0 ctx) outputs) :
    sequential_generatedRoot_252.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_252.Facts  (ctx : sequential_generatedRoot_252Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_252.Facts_0 ctx outputs

theorem sequential_generatedRoot_252.facts  (ctx : sequential_generatedRoot_252Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_252.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_252.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_252.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_254.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_254.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_254Ctx" => (sequential_generatedRoot_254.Ctx params i_0)

def sequential_generatedRoot_254.w_0_0  (ctx : sequential_generatedRoot_254Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_254.w_3_0  (ctx : sequential_generatedRoot_254Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_254.w_1_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  if (sequential_generatedRoot_254.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_254.w_2_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  0

def sequential_generatedRoot_254.w_4_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_254.w_5_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_254.w_6_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  1

def sequential_generatedRoot_254.w_7_0  (ctx : sequential_generatedRoot_254Ctx) : Bool :=
  decide ((sequential_generatedRoot_254.w_5_0 ctx) = (sequential_generatedRoot_254.w_6_0 ctx))

def sequential_generatedRoot_254.w_8_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  if (sequential_generatedRoot_254.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_254.w_9_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  ((sequential_generatedRoot_254.w_1_0 ctx) + (sequential_generatedRoot_254.w_8_0 ctx))

def sequential_generatedRoot_254.w_10_0  (ctx : sequential_generatedRoot_254Ctx) : Int :=
  2

def sequential_generatedRoot_254.w_11_0  (ctx : sequential_generatedRoot_254Ctx) : Bool :=
  decide ((sequential_generatedRoot_254.w_9_0 ctx) = (sequential_generatedRoot_254.w_10_0 ctx))

def sequential_generatedRoot_254.outs  (ctx : sequential_generatedRoot_254Ctx) : Bool :=
  (sequential_generatedRoot_254.w_11_0 ctx)

structure sequential_generatedRoot_254.Facts_0  (ctx : sequential_generatedRoot_254Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_232 params i_0 i (current, ((sequential_generatedRoot_254.w_3_0 ctx), ((sequential_generatedRoot_254.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_254.w_2_0 ctx) (sequential_generatedRoot_254.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_254.outs ctx

theorem sequential_generatedRoot_254.facts_0  {ctx : sequential_generatedRoot_254Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_254.constraints_0 params i_0 (sequential_generatedRoot_254.w_3_0 ctx) (sequential_generatedRoot_254.w_2_0 ctx) (sequential_generatedRoot_254.w_4_0 ctx) (sequential_generatedRoot_254.w_5_0 ctx) (sequential_generatedRoot_254.w_11_0 ctx) outputs) :
    sequential_generatedRoot_254.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_254.Facts  (ctx : sequential_generatedRoot_254Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_254.Facts_0 ctx outputs

theorem sequential_generatedRoot_254.facts  (ctx : sequential_generatedRoot_254Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_254.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_254.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_254.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_256.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_256.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_256Ctx" => (sequential_generatedRoot_256.Ctx params i_0)

def sequential_generatedRoot_256.w_0_0  (ctx : sequential_generatedRoot_256Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_256.w_3_0  (ctx : sequential_generatedRoot_256Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_256.w_1_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  if (sequential_generatedRoot_256.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_256.w_2_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  0

def sequential_generatedRoot_256.w_4_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_256.w_5_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_256.w_6_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  1

def sequential_generatedRoot_256.w_7_0  (ctx : sequential_generatedRoot_256Ctx) : Bool :=
  decide ((sequential_generatedRoot_256.w_5_0 ctx) = (sequential_generatedRoot_256.w_6_0 ctx))

def sequential_generatedRoot_256.w_8_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  if (sequential_generatedRoot_256.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_256.w_9_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  ((sequential_generatedRoot_256.w_1_0 ctx) + (sequential_generatedRoot_256.w_8_0 ctx))

def sequential_generatedRoot_256.w_10_0  (ctx : sequential_generatedRoot_256Ctx) : Int :=
  2

def sequential_generatedRoot_256.w_11_0  (ctx : sequential_generatedRoot_256Ctx) : Bool :=
  decide ((sequential_generatedRoot_256.w_9_0 ctx) = (sequential_generatedRoot_256.w_10_0 ctx))

def sequential_generatedRoot_256.outs  (ctx : sequential_generatedRoot_256Ctx) : Bool :=
  (sequential_generatedRoot_256.w_11_0 ctx)

structure sequential_generatedRoot_256.Facts_0  (ctx : sequential_generatedRoot_256Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_233 params i_0 i (current, ((sequential_generatedRoot_256.w_3_0 ctx), ((sequential_generatedRoot_256.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_256.w_2_0 ctx) (sequential_generatedRoot_256.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_256.outs ctx

theorem sequential_generatedRoot_256.facts_0  {ctx : sequential_generatedRoot_256Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_256.constraints_0 params i_0 (sequential_generatedRoot_256.w_3_0 ctx) (sequential_generatedRoot_256.w_2_0 ctx) (sequential_generatedRoot_256.w_4_0 ctx) (sequential_generatedRoot_256.w_5_0 ctx) (sequential_generatedRoot_256.w_11_0 ctx) outputs) :
    sequential_generatedRoot_256.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_256.Facts  (ctx : sequential_generatedRoot_256Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_256.Facts_0 ctx outputs

theorem sequential_generatedRoot_256.facts  (ctx : sequential_generatedRoot_256Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_256.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_256.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_256.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_258.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_258.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_258Ctx" => (sequential_generatedRoot_258.Ctx params i_0)

def sequential_generatedRoot_258.w_0_0  (ctx : sequential_generatedRoot_258Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_258.w_3_0  (ctx : sequential_generatedRoot_258Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_258.w_1_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  if (sequential_generatedRoot_258.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_258.w_2_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  0

def sequential_generatedRoot_258.w_4_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_258.w_5_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_258.w_6_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  1

def sequential_generatedRoot_258.w_7_0  (ctx : sequential_generatedRoot_258Ctx) : Bool :=
  decide ((sequential_generatedRoot_258.w_5_0 ctx) = (sequential_generatedRoot_258.w_6_0 ctx))

def sequential_generatedRoot_258.w_8_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  if (sequential_generatedRoot_258.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_258.w_9_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  ((sequential_generatedRoot_258.w_1_0 ctx) + (sequential_generatedRoot_258.w_8_0 ctx))

def sequential_generatedRoot_258.w_10_0  (ctx : sequential_generatedRoot_258Ctx) : Int :=
  2

def sequential_generatedRoot_258.w_11_0  (ctx : sequential_generatedRoot_258Ctx) : Bool :=
  decide ((sequential_generatedRoot_258.w_9_0 ctx) = (sequential_generatedRoot_258.w_10_0 ctx))

def sequential_generatedRoot_258.outs  (ctx : sequential_generatedRoot_258Ctx) : Bool :=
  (sequential_generatedRoot_258.w_11_0 ctx)

structure sequential_generatedRoot_258.Facts_0  (ctx : sequential_generatedRoot_258Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_234 params i_0 i (current, ((sequential_generatedRoot_258.w_3_0 ctx), ((sequential_generatedRoot_258.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_258.w_2_0 ctx) (sequential_generatedRoot_258.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_258.outs ctx

theorem sequential_generatedRoot_258.facts_0  {ctx : sequential_generatedRoot_258Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_258.constraints_0 params i_0 (sequential_generatedRoot_258.w_3_0 ctx) (sequential_generatedRoot_258.w_2_0 ctx) (sequential_generatedRoot_258.w_4_0 ctx) (sequential_generatedRoot_258.w_5_0 ctx) (sequential_generatedRoot_258.w_11_0 ctx) outputs) :
    sequential_generatedRoot_258.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_258.Facts  (ctx : sequential_generatedRoot_258Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_258.Facts_0 ctx outputs

theorem sequential_generatedRoot_258.facts  (ctx : sequential_generatedRoot_258Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_258.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_258.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_258.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_260.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_260.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_260Ctx" => (sequential_generatedRoot_260.Ctx params i_0)

def sequential_generatedRoot_260.w_0_0  (ctx : sequential_generatedRoot_260Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_260.w_3_0  (ctx : sequential_generatedRoot_260Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_260.w_1_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  if (sequential_generatedRoot_260.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_260.w_2_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  0

def sequential_generatedRoot_260.w_4_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_260.w_5_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_260.w_6_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  1

def sequential_generatedRoot_260.w_7_0  (ctx : sequential_generatedRoot_260Ctx) : Bool :=
  decide ((sequential_generatedRoot_260.w_5_0 ctx) = (sequential_generatedRoot_260.w_6_0 ctx))

def sequential_generatedRoot_260.w_8_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  if (sequential_generatedRoot_260.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_260.w_9_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  ((sequential_generatedRoot_260.w_1_0 ctx) + (sequential_generatedRoot_260.w_8_0 ctx))

def sequential_generatedRoot_260.w_10_0  (ctx : sequential_generatedRoot_260Ctx) : Int :=
  2

def sequential_generatedRoot_260.w_11_0  (ctx : sequential_generatedRoot_260Ctx) : Bool :=
  decide ((sequential_generatedRoot_260.w_9_0 ctx) = (sequential_generatedRoot_260.w_10_0 ctx))

def sequential_generatedRoot_260.outs  (ctx : sequential_generatedRoot_260Ctx) : Bool :=
  (sequential_generatedRoot_260.w_11_0 ctx)

structure sequential_generatedRoot_260.Facts_0  (ctx : sequential_generatedRoot_260Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_235 params i_0 i (current, ((sequential_generatedRoot_260.w_3_0 ctx), ((sequential_generatedRoot_260.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_260.w_2_0 ctx) (sequential_generatedRoot_260.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_260.outs ctx

theorem sequential_generatedRoot_260.facts_0  {ctx : sequential_generatedRoot_260Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_260.constraints_0 params i_0 (sequential_generatedRoot_260.w_3_0 ctx) (sequential_generatedRoot_260.w_2_0 ctx) (sequential_generatedRoot_260.w_4_0 ctx) (sequential_generatedRoot_260.w_5_0 ctx) (sequential_generatedRoot_260.w_11_0 ctx) outputs) :
    sequential_generatedRoot_260.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_260.Facts  (ctx : sequential_generatedRoot_260Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_260.Facts_0 ctx outputs

theorem sequential_generatedRoot_260.facts  (ctx : sequential_generatedRoot_260Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_260.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_260.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_260.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_262.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_262.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_262Ctx" => (sequential_generatedRoot_262.Ctx params i_0)

def sequential_generatedRoot_262.w_0_0  (ctx : sequential_generatedRoot_262Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_262.w_3_0  (ctx : sequential_generatedRoot_262Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_262.w_1_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  if (sequential_generatedRoot_262.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_262.w_2_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  0

def sequential_generatedRoot_262.w_4_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_262.w_5_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_262.w_6_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  1

def sequential_generatedRoot_262.w_7_0  (ctx : sequential_generatedRoot_262Ctx) : Bool :=
  decide ((sequential_generatedRoot_262.w_5_0 ctx) = (sequential_generatedRoot_262.w_6_0 ctx))

def sequential_generatedRoot_262.w_8_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  if (sequential_generatedRoot_262.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_262.w_9_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  ((sequential_generatedRoot_262.w_1_0 ctx) + (sequential_generatedRoot_262.w_8_0 ctx))

def sequential_generatedRoot_262.w_10_0  (ctx : sequential_generatedRoot_262Ctx) : Int :=
  2

def sequential_generatedRoot_262.w_11_0  (ctx : sequential_generatedRoot_262Ctx) : Bool :=
  decide ((sequential_generatedRoot_262.w_9_0 ctx) = (sequential_generatedRoot_262.w_10_0 ctx))

def sequential_generatedRoot_262.outs  (ctx : sequential_generatedRoot_262Ctx) : Bool :=
  (sequential_generatedRoot_262.w_11_0 ctx)

structure sequential_generatedRoot_262.Facts_0  (ctx : sequential_generatedRoot_262Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_236 params i_0 i (current, ((sequential_generatedRoot_262.w_3_0 ctx), ((sequential_generatedRoot_262.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_262.w_2_0 ctx) (sequential_generatedRoot_262.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_262.outs ctx

theorem sequential_generatedRoot_262.facts_0  {ctx : sequential_generatedRoot_262Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_262.constraints_0 params i_0 (sequential_generatedRoot_262.w_3_0 ctx) (sequential_generatedRoot_262.w_2_0 ctx) (sequential_generatedRoot_262.w_4_0 ctx) (sequential_generatedRoot_262.w_5_0 ctx) (sequential_generatedRoot_262.w_11_0 ctx) outputs) :
    sequential_generatedRoot_262.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_262.Facts  (ctx : sequential_generatedRoot_262Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_262.Facts_0 ctx outputs

theorem sequential_generatedRoot_262.facts  (ctx : sequential_generatedRoot_262Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_262.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_262.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_262.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_264.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_264.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_264Ctx" => (sequential_generatedRoot_264.Ctx params i_0)

def sequential_generatedRoot_264.w_0_0  (ctx : sequential_generatedRoot_264Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_264.w_3_0  (ctx : sequential_generatedRoot_264Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_264.w_1_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  if (sequential_generatedRoot_264.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_264.w_2_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  0

def sequential_generatedRoot_264.w_4_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_264.w_5_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_264.w_6_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  1

def sequential_generatedRoot_264.w_7_0  (ctx : sequential_generatedRoot_264Ctx) : Bool :=
  decide ((sequential_generatedRoot_264.w_5_0 ctx) = (sequential_generatedRoot_264.w_6_0 ctx))

def sequential_generatedRoot_264.w_8_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  if (sequential_generatedRoot_264.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_264.w_9_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  ((sequential_generatedRoot_264.w_1_0 ctx) + (sequential_generatedRoot_264.w_8_0 ctx))

def sequential_generatedRoot_264.w_10_0  (ctx : sequential_generatedRoot_264Ctx) : Int :=
  2

def sequential_generatedRoot_264.w_11_0  (ctx : sequential_generatedRoot_264Ctx) : Bool :=
  decide ((sequential_generatedRoot_264.w_9_0 ctx) = (sequential_generatedRoot_264.w_10_0 ctx))

def sequential_generatedRoot_264.outs  (ctx : sequential_generatedRoot_264Ctx) : Bool :=
  (sequential_generatedRoot_264.w_11_0 ctx)

structure sequential_generatedRoot_264.Facts_0  (ctx : sequential_generatedRoot_264Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_237 params i_0 i (current, ((sequential_generatedRoot_264.w_3_0 ctx), ((sequential_generatedRoot_264.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_264.w_2_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_264.outs ctx

theorem sequential_generatedRoot_264.facts_0  {ctx : sequential_generatedRoot_264Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_264.constraints_0 params i_0 (sequential_generatedRoot_264.w_3_0 ctx) (sequential_generatedRoot_264.w_2_0 ctx) (sequential_generatedRoot_264.w_4_0 ctx) (sequential_generatedRoot_264.w_5_0 ctx) (sequential_generatedRoot_264.w_11_0 ctx) outputs) :
    sequential_generatedRoot_264.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_264.Facts  (ctx : sequential_generatedRoot_264Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_264.Facts_0 ctx outputs

theorem sequential_generatedRoot_264.facts  (ctx : sequential_generatedRoot_264Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_264.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_264.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_264.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_266.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_266.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_266Ctx" => (sequential_generatedRoot_266.Ctx params i_0)

def sequential_generatedRoot_266.w_0_0  (ctx : sequential_generatedRoot_266Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_266.w_3_0  (ctx : sequential_generatedRoot_266Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_266.w_1_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  if (sequential_generatedRoot_266.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_266.w_2_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  0

def sequential_generatedRoot_266.w_4_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_266.w_5_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_266.w_6_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  1

def sequential_generatedRoot_266.w_7_0  (ctx : sequential_generatedRoot_266Ctx) : Bool :=
  decide ((sequential_generatedRoot_266.w_5_0 ctx) = (sequential_generatedRoot_266.w_6_0 ctx))

def sequential_generatedRoot_266.w_8_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  if (sequential_generatedRoot_266.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_266.w_9_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  ((sequential_generatedRoot_266.w_1_0 ctx) + (sequential_generatedRoot_266.w_8_0 ctx))

def sequential_generatedRoot_266.w_10_0  (ctx : sequential_generatedRoot_266Ctx) : Int :=
  2

def sequential_generatedRoot_266.w_11_0  (ctx : sequential_generatedRoot_266Ctx) : Bool :=
  decide ((sequential_generatedRoot_266.w_9_0 ctx) = (sequential_generatedRoot_266.w_10_0 ctx))

def sequential_generatedRoot_266.outs  (ctx : sequential_generatedRoot_266Ctx) : Bool :=
  (sequential_generatedRoot_266.w_11_0 ctx)

structure sequential_generatedRoot_266.Facts_0  (ctx : sequential_generatedRoot_266Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_238 params i_0 i (current, ((sequential_generatedRoot_266.w_3_0 ctx), ((sequential_generatedRoot_266.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_266.w_2_0 ctx) (sequential_generatedRoot_266.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_266.outs ctx

theorem sequential_generatedRoot_266.facts_0  {ctx : sequential_generatedRoot_266Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_266.constraints_0 params i_0 (sequential_generatedRoot_266.w_3_0 ctx) (sequential_generatedRoot_266.w_2_0 ctx) (sequential_generatedRoot_266.w_4_0 ctx) (sequential_generatedRoot_266.w_5_0 ctx) (sequential_generatedRoot_266.w_11_0 ctx) outputs) :
    sequential_generatedRoot_266.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_266.Facts  (ctx : sequential_generatedRoot_266Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_266.Facts_0 ctx outputs

theorem sequential_generatedRoot_266.facts  (ctx : sequential_generatedRoot_266Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_266.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_266.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_266.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_268.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_268.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_268Ctx" => (sequential_generatedRoot_268.Ctx params i_0)

def sequential_generatedRoot_268.w_0_0  (ctx : sequential_generatedRoot_268Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_268.w_3_0  (ctx : sequential_generatedRoot_268Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_268.w_1_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  if (sequential_generatedRoot_268.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_268.w_2_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  0

def sequential_generatedRoot_268.w_4_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_268.w_5_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_268.w_6_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  1

def sequential_generatedRoot_268.w_7_0  (ctx : sequential_generatedRoot_268Ctx) : Bool :=
  decide ((sequential_generatedRoot_268.w_5_0 ctx) = (sequential_generatedRoot_268.w_6_0 ctx))

def sequential_generatedRoot_268.w_8_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  if (sequential_generatedRoot_268.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_268.w_9_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  ((sequential_generatedRoot_268.w_1_0 ctx) + (sequential_generatedRoot_268.w_8_0 ctx))

def sequential_generatedRoot_268.w_10_0  (ctx : sequential_generatedRoot_268Ctx) : Int :=
  2

def sequential_generatedRoot_268.w_11_0  (ctx : sequential_generatedRoot_268Ctx) : Bool :=
  decide ((sequential_generatedRoot_268.w_9_0 ctx) = (sequential_generatedRoot_268.w_10_0 ctx))

def sequential_generatedRoot_268.outs  (ctx : sequential_generatedRoot_268Ctx) : Bool :=
  (sequential_generatedRoot_268.w_11_0 ctx)

structure sequential_generatedRoot_268.Facts_0  (ctx : sequential_generatedRoot_268Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_239 params i_0 i (current, ((sequential_generatedRoot_268.w_3_0 ctx), ((sequential_generatedRoot_268.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_268.w_2_0 ctx) (sequential_generatedRoot_268.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_268.outs ctx

theorem sequential_generatedRoot_268.facts_0  {ctx : sequential_generatedRoot_268Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_268.constraints_0 params i_0 (sequential_generatedRoot_268.w_3_0 ctx) (sequential_generatedRoot_268.w_2_0 ctx) (sequential_generatedRoot_268.w_4_0 ctx) (sequential_generatedRoot_268.w_5_0 ctx) (sequential_generatedRoot_268.w_11_0 ctx) outputs) :
    sequential_generatedRoot_268.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_268.Facts  (ctx : sequential_generatedRoot_268Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_268.Facts_0 ctx outputs

theorem sequential_generatedRoot_268.facts  (ctx : sequential_generatedRoot_268Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_268.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_268.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_268.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_270.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_270.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_270Ctx" => (sequential_generatedRoot_270.Ctx params i_0)

def sequential_generatedRoot_270.w_0_0  (ctx : sequential_generatedRoot_270Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_270.w_3_0  (ctx : sequential_generatedRoot_270Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_270.w_1_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  if (sequential_generatedRoot_270.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_270.w_2_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  0

def sequential_generatedRoot_270.w_4_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_270.w_5_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_270.w_6_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  1

def sequential_generatedRoot_270.w_7_0  (ctx : sequential_generatedRoot_270Ctx) : Bool :=
  decide ((sequential_generatedRoot_270.w_5_0 ctx) = (sequential_generatedRoot_270.w_6_0 ctx))

def sequential_generatedRoot_270.w_8_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  if (sequential_generatedRoot_270.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_270.w_9_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  ((sequential_generatedRoot_270.w_1_0 ctx) + (sequential_generatedRoot_270.w_8_0 ctx))

def sequential_generatedRoot_270.w_10_0  (ctx : sequential_generatedRoot_270Ctx) : Int :=
  2

def sequential_generatedRoot_270.w_11_0  (ctx : sequential_generatedRoot_270Ctx) : Bool :=
  decide ((sequential_generatedRoot_270.w_9_0 ctx) = (sequential_generatedRoot_270.w_10_0 ctx))

def sequential_generatedRoot_270.outs  (ctx : sequential_generatedRoot_270Ctx) : Bool :=
  (sequential_generatedRoot_270.w_11_0 ctx)

structure sequential_generatedRoot_270.Facts_0  (ctx : sequential_generatedRoot_270Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_240 params i_0 i (current, ((sequential_generatedRoot_270.w_3_0 ctx), ((sequential_generatedRoot_270.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_270.w_2_0 ctx) (sequential_generatedRoot_270.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_270.outs ctx

theorem sequential_generatedRoot_270.facts_0  {ctx : sequential_generatedRoot_270Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_270.constraints_0 params i_0 (sequential_generatedRoot_270.w_3_0 ctx) (sequential_generatedRoot_270.w_2_0 ctx) (sequential_generatedRoot_270.w_4_0 ctx) (sequential_generatedRoot_270.w_5_0 ctx) (sequential_generatedRoot_270.w_11_0 ctx) outputs) :
    sequential_generatedRoot_270.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_270.Facts  (ctx : sequential_generatedRoot_270Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_270.Facts_0 ctx outputs

theorem sequential_generatedRoot_270.facts  (ctx : sequential_generatedRoot_270Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_270.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_270.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_270.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_272.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_272.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_272Ctx" => (sequential_generatedRoot_272.Ctx params i_0)

def sequential_generatedRoot_272.w_0_0  (ctx : sequential_generatedRoot_272Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_272.w_3_0  (ctx : sequential_generatedRoot_272Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_272.w_1_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  if (sequential_generatedRoot_272.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_272.w_2_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  0

def sequential_generatedRoot_272.w_4_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_272.w_5_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_272.w_6_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  1

def sequential_generatedRoot_272.w_7_0  (ctx : sequential_generatedRoot_272Ctx) : Bool :=
  decide ((sequential_generatedRoot_272.w_5_0 ctx) = (sequential_generatedRoot_272.w_6_0 ctx))

def sequential_generatedRoot_272.w_8_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  if (sequential_generatedRoot_272.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_272.w_9_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  ((sequential_generatedRoot_272.w_1_0 ctx) + (sequential_generatedRoot_272.w_8_0 ctx))

def sequential_generatedRoot_272.w_10_0  (ctx : sequential_generatedRoot_272Ctx) : Int :=
  2

def sequential_generatedRoot_272.w_11_0  (ctx : sequential_generatedRoot_272Ctx) : Bool :=
  decide ((sequential_generatedRoot_272.w_9_0 ctx) = (sequential_generatedRoot_272.w_10_0 ctx))

def sequential_generatedRoot_272.outs  (ctx : sequential_generatedRoot_272Ctx) : Bool :=
  (sequential_generatedRoot_272.w_11_0 ctx)

structure sequential_generatedRoot_272.Facts_0  (ctx : sequential_generatedRoot_272Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_241 params i_0 i (current, ((sequential_generatedRoot_272.w_3_0 ctx), ((sequential_generatedRoot_272.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_272.w_2_0 ctx) (sequential_generatedRoot_272.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_272.outs ctx

theorem sequential_generatedRoot_272.facts_0  {ctx : sequential_generatedRoot_272Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_272.constraints_0 params i_0 (sequential_generatedRoot_272.w_3_0 ctx) (sequential_generatedRoot_272.w_2_0 ctx) (sequential_generatedRoot_272.w_4_0 ctx) (sequential_generatedRoot_272.w_5_0 ctx) (sequential_generatedRoot_272.w_11_0 ctx) outputs) :
    sequential_generatedRoot_272.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_272.Facts  (ctx : sequential_generatedRoot_272Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_272.Facts_0 ctx outputs

theorem sequential_generatedRoot_272.facts  (ctx : sequential_generatedRoot_272Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_272.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_272.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_272.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_274.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_274.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_274Ctx" => (sequential_generatedRoot_274.Ctx params i_0)

def sequential_generatedRoot_274.w_0_0  (ctx : sequential_generatedRoot_274Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_274.w_3_0  (ctx : sequential_generatedRoot_274Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_274.w_1_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  if (sequential_generatedRoot_274.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_274.w_2_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  0

def sequential_generatedRoot_274.w_4_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_274.w_5_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_274.w_6_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  1

def sequential_generatedRoot_274.w_7_0  (ctx : sequential_generatedRoot_274Ctx) : Bool :=
  decide ((sequential_generatedRoot_274.w_5_0 ctx) = (sequential_generatedRoot_274.w_6_0 ctx))

def sequential_generatedRoot_274.w_8_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  if (sequential_generatedRoot_274.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_274.w_9_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  ((sequential_generatedRoot_274.w_1_0 ctx) + (sequential_generatedRoot_274.w_8_0 ctx))

def sequential_generatedRoot_274.w_10_0  (ctx : sequential_generatedRoot_274Ctx) : Int :=
  2

def sequential_generatedRoot_274.w_11_0  (ctx : sequential_generatedRoot_274Ctx) : Bool :=
  decide ((sequential_generatedRoot_274.w_9_0 ctx) = (sequential_generatedRoot_274.w_10_0 ctx))

def sequential_generatedRoot_274.outs  (ctx : sequential_generatedRoot_274Ctx) : Bool :=
  (sequential_generatedRoot_274.w_11_0 ctx)

structure sequential_generatedRoot_274.Facts_0  (ctx : sequential_generatedRoot_274Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_242 params i_0 i (current, ((sequential_generatedRoot_274.w_3_0 ctx), ((sequential_generatedRoot_274.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_274.w_2_0 ctx) (sequential_generatedRoot_274.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_274.outs ctx

theorem sequential_generatedRoot_274.facts_0  {ctx : sequential_generatedRoot_274Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_274.constraints_0 params i_0 (sequential_generatedRoot_274.w_3_0 ctx) (sequential_generatedRoot_274.w_2_0 ctx) (sequential_generatedRoot_274.w_4_0 ctx) (sequential_generatedRoot_274.w_5_0 ctx) (sequential_generatedRoot_274.w_11_0 ctx) outputs) :
    sequential_generatedRoot_274.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_274.Facts  (ctx : sequential_generatedRoot_274Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_274.Facts_0 ctx outputs

theorem sequential_generatedRoot_274.facts  (ctx : sequential_generatedRoot_274Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_274.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_274.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_274.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_276.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 38598194862489600 → Int) × Unit
  witness : sequential_generatedRoot_276.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_276Ctx" => (sequential_generatedRoot_276.Ctx params i_0)

def sequential_generatedRoot_276.w_0_0  (ctx : sequential_generatedRoot_276Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_276.w_3_0  (ctx : sequential_generatedRoot_276Ctx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_276.w_1_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  if (sequential_generatedRoot_276.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_276.w_2_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  0

def sequential_generatedRoot_276.w_4_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_276.w_5_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_276.w_6_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  1

def sequential_generatedRoot_276.w_7_0  (ctx : sequential_generatedRoot_276Ctx) : Bool :=
  decide ((sequential_generatedRoot_276.w_5_0 ctx) = (sequential_generatedRoot_276.w_6_0 ctx))

def sequential_generatedRoot_276.w_8_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  if (sequential_generatedRoot_276.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_276.w_9_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  ((sequential_generatedRoot_276.w_1_0 ctx) + (sequential_generatedRoot_276.w_8_0 ctx))

def sequential_generatedRoot_276.w_10_0  (ctx : sequential_generatedRoot_276Ctx) : Int :=
  2

def sequential_generatedRoot_276.w_11_0  (ctx : sequential_generatedRoot_276Ctx) : Bool :=
  decide ((sequential_generatedRoot_276.w_9_0 ctx) = (sequential_generatedRoot_276.w_10_0 ctx))

def sequential_generatedRoot_276.outs  (ctx : sequential_generatedRoot_276Ctx) : Bool :=
  (sequential_generatedRoot_276.w_11_0 ctx)

structure sequential_generatedRoot_276.Facts_0  (ctx : sequential_generatedRoot_276Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_243 params i_0 i (current, ((sequential_generatedRoot_276.w_3_0 ctx), ((sequential_generatedRoot_276.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_276.w_2_0 ctx) (sequential_generatedRoot_276.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_276.outs ctx

theorem sequential_generatedRoot_276.facts_0  {ctx : sequential_generatedRoot_276Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_276.constraints_0 params i_0 (sequential_generatedRoot_276.w_3_0 ctx) (sequential_generatedRoot_276.w_2_0 ctx) (sequential_generatedRoot_276.w_4_0 ctx) (sequential_generatedRoot_276.w_5_0 ctx) (sequential_generatedRoot_276.w_11_0 ctx) outputs) :
    sequential_generatedRoot_276.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_276.Facts  (ctx : sequential_generatedRoot_276Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_276.Facts_0 ctx outputs

theorem sequential_generatedRoot_276.facts  (ctx : sequential_generatedRoot_276Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_276.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_276.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_276.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_278.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_278.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_278Ctx" => (sequential_generatedRoot_278.Ctx params i_0)

def sequential_generatedRoot_278.w_0_0  (ctx : sequential_generatedRoot_278Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_278.w_3_0  (ctx : sequential_generatedRoot_278Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_278.w_1_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  if (sequential_generatedRoot_278.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_278.w_2_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  0

def sequential_generatedRoot_278.w_4_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_278.w_5_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_278.w_6_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  1

def sequential_generatedRoot_278.w_7_0  (ctx : sequential_generatedRoot_278Ctx) : Bool :=
  decide ((sequential_generatedRoot_278.w_5_0 ctx) = (sequential_generatedRoot_278.w_6_0 ctx))

def sequential_generatedRoot_278.w_8_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  if (sequential_generatedRoot_278.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_278.w_9_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  ((sequential_generatedRoot_278.w_1_0 ctx) + (sequential_generatedRoot_278.w_8_0 ctx))

def sequential_generatedRoot_278.w_10_0  (ctx : sequential_generatedRoot_278Ctx) : Int :=
  2

def sequential_generatedRoot_278.w_11_0  (ctx : sequential_generatedRoot_278Ctx) : Bool :=
  decide ((sequential_generatedRoot_278.w_9_0 ctx) = (sequential_generatedRoot_278.w_10_0 ctx))

def sequential_generatedRoot_278.outs  (ctx : sequential_generatedRoot_278Ctx) : Bool :=
  (sequential_generatedRoot_278.w_11_0 ctx)

structure sequential_generatedRoot_278.Facts_0  (ctx : sequential_generatedRoot_278Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_244 params i_0 i (current, ((sequential_generatedRoot_278.w_3_0 ctx), ((sequential_generatedRoot_278.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_278.w_2_0 ctx) (sequential_generatedRoot_278.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_278.outs ctx

theorem sequential_generatedRoot_278.facts_0  {ctx : sequential_generatedRoot_278Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_278.constraints_0 params i_0 (sequential_generatedRoot_278.w_3_0 ctx) (sequential_generatedRoot_278.w_2_0 ctx) (sequential_generatedRoot_278.w_4_0 ctx) (sequential_generatedRoot_278.w_5_0 ctx) (sequential_generatedRoot_278.w_11_0 ctx) outputs) :
    sequential_generatedRoot_278.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_278.Facts  (ctx : sequential_generatedRoot_278Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_278.Facts_0 ctx outputs

theorem sequential_generatedRoot_278.facts  (ctx : sequential_generatedRoot_278Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_278.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_278.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_278.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_280.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_280.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_280Ctx" => (sequential_generatedRoot_280.Ctx params i_0)

def sequential_generatedRoot_280.w_0_0  (ctx : sequential_generatedRoot_280Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_280.w_3_0  (ctx : sequential_generatedRoot_280Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_280.w_1_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  if (sequential_generatedRoot_280.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_280.w_2_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  0

def sequential_generatedRoot_280.w_4_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_280.w_5_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_280.w_6_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  1

def sequential_generatedRoot_280.w_7_0  (ctx : sequential_generatedRoot_280Ctx) : Bool :=
  decide ((sequential_generatedRoot_280.w_5_0 ctx) = (sequential_generatedRoot_280.w_6_0 ctx))

def sequential_generatedRoot_280.w_8_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  if (sequential_generatedRoot_280.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_280.w_9_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  ((sequential_generatedRoot_280.w_1_0 ctx) + (sequential_generatedRoot_280.w_8_0 ctx))

def sequential_generatedRoot_280.w_10_0  (ctx : sequential_generatedRoot_280Ctx) : Int :=
  2

def sequential_generatedRoot_280.w_11_0  (ctx : sequential_generatedRoot_280Ctx) : Bool :=
  decide ((sequential_generatedRoot_280.w_9_0 ctx) = (sequential_generatedRoot_280.w_10_0 ctx))

def sequential_generatedRoot_280.outs  (ctx : sequential_generatedRoot_280Ctx) : Bool :=
  (sequential_generatedRoot_280.w_11_0 ctx)

structure sequential_generatedRoot_280.Facts_0  (ctx : sequential_generatedRoot_280Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_245 params i_0 i (current, ((sequential_generatedRoot_280.w_3_0 ctx), ((sequential_generatedRoot_280.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_280.w_2_0 ctx) (sequential_generatedRoot_280.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_280.outs ctx

theorem sequential_generatedRoot_280.facts_0  {ctx : sequential_generatedRoot_280Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_280.constraints_0 params i_0 (sequential_generatedRoot_280.w_3_0 ctx) (sequential_generatedRoot_280.w_2_0 ctx) (sequential_generatedRoot_280.w_4_0 ctx) (sequential_generatedRoot_280.w_5_0 ctx) (sequential_generatedRoot_280.w_11_0 ctx) outputs) :
    sequential_generatedRoot_280.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_280.Facts  (ctx : sequential_generatedRoot_280Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_280.Facts_0 ctx outputs

theorem sequential_generatedRoot_280.facts  (ctx : sequential_generatedRoot_280Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_280.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_280.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_280.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_282.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_282.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_282Ctx" => (sequential_generatedRoot_282.Ctx params i_0)

def sequential_generatedRoot_282.w_0_0  (ctx : sequential_generatedRoot_282Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_282.w_3_0  (ctx : sequential_generatedRoot_282Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_282.w_1_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  if (sequential_generatedRoot_282.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_282.w_2_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  0

def sequential_generatedRoot_282.w_4_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_282.w_5_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_282.w_6_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  1

def sequential_generatedRoot_282.w_7_0  (ctx : sequential_generatedRoot_282Ctx) : Bool :=
  decide ((sequential_generatedRoot_282.w_5_0 ctx) = (sequential_generatedRoot_282.w_6_0 ctx))

def sequential_generatedRoot_282.w_8_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  if (sequential_generatedRoot_282.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_282.w_9_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  ((sequential_generatedRoot_282.w_1_0 ctx) + (sequential_generatedRoot_282.w_8_0 ctx))

def sequential_generatedRoot_282.w_10_0  (ctx : sequential_generatedRoot_282Ctx) : Int :=
  2

def sequential_generatedRoot_282.w_11_0  (ctx : sequential_generatedRoot_282Ctx) : Bool :=
  decide ((sequential_generatedRoot_282.w_9_0 ctx) = (sequential_generatedRoot_282.w_10_0 ctx))

def sequential_generatedRoot_282.outs  (ctx : sequential_generatedRoot_282Ctx) : Bool :=
  (sequential_generatedRoot_282.w_11_0 ctx)

structure sequential_generatedRoot_282.Facts_0  (ctx : sequential_generatedRoot_282Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_246 params i_0 i (current, ((sequential_generatedRoot_282.w_3_0 ctx), ((sequential_generatedRoot_282.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_282.w_2_0 ctx) (sequential_generatedRoot_282.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_282.outs ctx

theorem sequential_generatedRoot_282.facts_0  {ctx : sequential_generatedRoot_282Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_282.constraints_0 params i_0 (sequential_generatedRoot_282.w_3_0 ctx) (sequential_generatedRoot_282.w_2_0 ctx) (sequential_generatedRoot_282.w_4_0 ctx) (sequential_generatedRoot_282.w_5_0 ctx) (sequential_generatedRoot_282.w_11_0 ctx) outputs) :
    sequential_generatedRoot_282.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_282.Facts  (ctx : sequential_generatedRoot_282Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_282.Facts_0 ctx outputs

theorem sequential_generatedRoot_282.facts  (ctx : sequential_generatedRoot_282Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_282.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_282.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_282.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_284.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_284.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_284Ctx" => (sequential_generatedRoot_284.Ctx params i_0)

def sequential_generatedRoot_284.w_0_0  (ctx : sequential_generatedRoot_284Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_284.w_3_0  (ctx : sequential_generatedRoot_284Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_284.w_1_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  if (sequential_generatedRoot_284.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_284.w_2_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  0

def sequential_generatedRoot_284.w_4_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_284.w_5_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_284.w_6_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  1

def sequential_generatedRoot_284.w_7_0  (ctx : sequential_generatedRoot_284Ctx) : Bool :=
  decide ((sequential_generatedRoot_284.w_5_0 ctx) = (sequential_generatedRoot_284.w_6_0 ctx))

def sequential_generatedRoot_284.w_8_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  if (sequential_generatedRoot_284.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_284.w_9_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  ((sequential_generatedRoot_284.w_1_0 ctx) + (sequential_generatedRoot_284.w_8_0 ctx))

def sequential_generatedRoot_284.w_10_0  (ctx : sequential_generatedRoot_284Ctx) : Int :=
  2

def sequential_generatedRoot_284.w_11_0  (ctx : sequential_generatedRoot_284Ctx) : Bool :=
  decide ((sequential_generatedRoot_284.w_9_0 ctx) = (sequential_generatedRoot_284.w_10_0 ctx))

def sequential_generatedRoot_284.outs  (ctx : sequential_generatedRoot_284Ctx) : Bool :=
  (sequential_generatedRoot_284.w_11_0 ctx)

structure sequential_generatedRoot_284.Facts_0  (ctx : sequential_generatedRoot_284Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_247 params i_0 i (current, ((sequential_generatedRoot_284.w_3_0 ctx), ((sequential_generatedRoot_284.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_284.w_2_0 ctx) (sequential_generatedRoot_284.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_284.outs ctx

theorem sequential_generatedRoot_284.facts_0  {ctx : sequential_generatedRoot_284Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_284.constraints_0 params i_0 (sequential_generatedRoot_284.w_3_0 ctx) (sequential_generatedRoot_284.w_2_0 ctx) (sequential_generatedRoot_284.w_4_0 ctx) (sequential_generatedRoot_284.w_5_0 ctx) (sequential_generatedRoot_284.w_11_0 ctx) outputs) :
    sequential_generatedRoot_284.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_284.Facts  (ctx : sequential_generatedRoot_284Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_284.Facts_0 ctx outputs

theorem sequential_generatedRoot_284.facts  (ctx : sequential_generatedRoot_284Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_284.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_284.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_284.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_286.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_286.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_286Ctx" => (sequential_generatedRoot_286.Ctx params i_0)

def sequential_generatedRoot_286.w_0_0  (ctx : sequential_generatedRoot_286Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_286.w_3_0  (ctx : sequential_generatedRoot_286Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_286.w_1_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  if (sequential_generatedRoot_286.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_286.w_2_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  0

def sequential_generatedRoot_286.w_4_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_286.w_5_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_286.w_6_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  1

def sequential_generatedRoot_286.w_7_0  (ctx : sequential_generatedRoot_286Ctx) : Bool :=
  decide ((sequential_generatedRoot_286.w_5_0 ctx) = (sequential_generatedRoot_286.w_6_0 ctx))

def sequential_generatedRoot_286.w_8_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  if (sequential_generatedRoot_286.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_286.w_9_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  ((sequential_generatedRoot_286.w_1_0 ctx) + (sequential_generatedRoot_286.w_8_0 ctx))

def sequential_generatedRoot_286.w_10_0  (ctx : sequential_generatedRoot_286Ctx) : Int :=
  2

def sequential_generatedRoot_286.w_11_0  (ctx : sequential_generatedRoot_286Ctx) : Bool :=
  decide ((sequential_generatedRoot_286.w_9_0 ctx) = (sequential_generatedRoot_286.w_10_0 ctx))

def sequential_generatedRoot_286.outs  (ctx : sequential_generatedRoot_286Ctx) : Bool :=
  (sequential_generatedRoot_286.w_11_0 ctx)

structure sequential_generatedRoot_286.Facts_0  (ctx : sequential_generatedRoot_286Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_248 params i_0 i (current, ((sequential_generatedRoot_286.w_3_0 ctx), ((sequential_generatedRoot_286.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_286.w_2_0 ctx) (sequential_generatedRoot_286.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_286.outs ctx

theorem sequential_generatedRoot_286.facts_0  {ctx : sequential_generatedRoot_286Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_286.constraints_0 params i_0 (sequential_generatedRoot_286.w_3_0 ctx) (sequential_generatedRoot_286.w_2_0 ctx) (sequential_generatedRoot_286.w_4_0 ctx) (sequential_generatedRoot_286.w_5_0 ctx) (sequential_generatedRoot_286.w_11_0 ctx) outputs) :
    sequential_generatedRoot_286.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_286.Facts  (ctx : sequential_generatedRoot_286Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_286.Facts_0 ctx outputs

theorem sequential_generatedRoot_286.facts  (ctx : sequential_generatedRoot_286Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_286.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_286.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_286.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_288.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_288.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_288Ctx" => (sequential_generatedRoot_288.Ctx params i_0)

def sequential_generatedRoot_288.w_0_0  (ctx : sequential_generatedRoot_288Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_288.w_3_0  (ctx : sequential_generatedRoot_288Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_288.w_1_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  if (sequential_generatedRoot_288.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_288.w_2_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  0

def sequential_generatedRoot_288.w_4_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_288.w_5_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_288.w_6_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  1

def sequential_generatedRoot_288.w_7_0  (ctx : sequential_generatedRoot_288Ctx) : Bool :=
  decide ((sequential_generatedRoot_288.w_5_0 ctx) = (sequential_generatedRoot_288.w_6_0 ctx))

def sequential_generatedRoot_288.w_8_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  if (sequential_generatedRoot_288.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_288.w_9_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  ((sequential_generatedRoot_288.w_1_0 ctx) + (sequential_generatedRoot_288.w_8_0 ctx))

def sequential_generatedRoot_288.w_10_0  (ctx : sequential_generatedRoot_288Ctx) : Int :=
  2

def sequential_generatedRoot_288.w_11_0  (ctx : sequential_generatedRoot_288Ctx) : Bool :=
  decide ((sequential_generatedRoot_288.w_9_0 ctx) = (sequential_generatedRoot_288.w_10_0 ctx))

def sequential_generatedRoot_288.outs  (ctx : sequential_generatedRoot_288Ctx) : Bool :=
  (sequential_generatedRoot_288.w_11_0 ctx)

structure sequential_generatedRoot_288.Facts_0  (ctx : sequential_generatedRoot_288Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_249 params i_0 i (current, ((sequential_generatedRoot_288.w_3_0 ctx), ((sequential_generatedRoot_288.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_288.w_2_0 ctx) (sequential_generatedRoot_288.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_288.outs ctx

theorem sequential_generatedRoot_288.facts_0  {ctx : sequential_generatedRoot_288Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_288.constraints_0 params i_0 (sequential_generatedRoot_288.w_3_0 ctx) (sequential_generatedRoot_288.w_2_0 ctx) (sequential_generatedRoot_288.w_4_0 ctx) (sequential_generatedRoot_288.w_5_0 ctx) (sequential_generatedRoot_288.w_11_0 ctx) outputs) :
    sequential_generatedRoot_288.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_288.Facts  (ctx : sequential_generatedRoot_288Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_288.Facts_0 ctx outputs

theorem sequential_generatedRoot_288.facts  (ctx : sequential_generatedRoot_288Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_288.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_288.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_288.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_290.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_290.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_290Ctx" => (sequential_generatedRoot_290.Ctx params i_0)

def sequential_generatedRoot_290.w_0_0  (ctx : sequential_generatedRoot_290Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_290.w_3_0  (ctx : sequential_generatedRoot_290Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_290.w_1_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  if (sequential_generatedRoot_290.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_290.w_2_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  0

def sequential_generatedRoot_290.w_4_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_290.w_5_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_290.w_6_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  1

def sequential_generatedRoot_290.w_7_0  (ctx : sequential_generatedRoot_290Ctx) : Bool :=
  decide ((sequential_generatedRoot_290.w_5_0 ctx) = (sequential_generatedRoot_290.w_6_0 ctx))

def sequential_generatedRoot_290.w_8_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  if (sequential_generatedRoot_290.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_290.w_9_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  ((sequential_generatedRoot_290.w_1_0 ctx) + (sequential_generatedRoot_290.w_8_0 ctx))

def sequential_generatedRoot_290.w_10_0  (ctx : sequential_generatedRoot_290Ctx) : Int :=
  2

def sequential_generatedRoot_290.w_11_0  (ctx : sequential_generatedRoot_290Ctx) : Bool :=
  decide ((sequential_generatedRoot_290.w_9_0 ctx) = (sequential_generatedRoot_290.w_10_0 ctx))

def sequential_generatedRoot_290.outs  (ctx : sequential_generatedRoot_290Ctx) : Bool :=
  (sequential_generatedRoot_290.w_11_0 ctx)

structure sequential_generatedRoot_290.Facts_0  (ctx : sequential_generatedRoot_290Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_250 params i_0 i (current, ((sequential_generatedRoot_290.w_3_0 ctx), ((sequential_generatedRoot_290.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_290.w_2_0 ctx) (sequential_generatedRoot_290.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_290.outs ctx

theorem sequential_generatedRoot_290.facts_0  {ctx : sequential_generatedRoot_290Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_290.constraints_0 params i_0 (sequential_generatedRoot_290.w_3_0 ctx) (sequential_generatedRoot_290.w_2_0 ctx) (sequential_generatedRoot_290.w_4_0 ctx) (sequential_generatedRoot_290.w_5_0 ctx) (sequential_generatedRoot_290.w_11_0 ctx) outputs) :
    sequential_generatedRoot_290.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_290.Facts  (ctx : sequential_generatedRoot_290Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_290.Facts_0 ctx outputs

theorem sequential_generatedRoot_290.facts  (ctx : sequential_generatedRoot_290Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_290.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_290.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_290.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_292.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_292.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_292Ctx" => (sequential_generatedRoot_292.Ctx params i_0)

def sequential_generatedRoot_292.w_0_0  (ctx : sequential_generatedRoot_292Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_292.w_3_0  (ctx : sequential_generatedRoot_292Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_292.w_1_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  if (sequential_generatedRoot_292.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_292.w_2_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  0

def sequential_generatedRoot_292.w_4_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_292.w_5_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_292.w_6_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  1

def sequential_generatedRoot_292.w_7_0  (ctx : sequential_generatedRoot_292Ctx) : Bool :=
  decide ((sequential_generatedRoot_292.w_5_0 ctx) = (sequential_generatedRoot_292.w_6_0 ctx))

def sequential_generatedRoot_292.w_8_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  if (sequential_generatedRoot_292.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_292.w_9_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  ((sequential_generatedRoot_292.w_1_0 ctx) + (sequential_generatedRoot_292.w_8_0 ctx))

def sequential_generatedRoot_292.w_10_0  (ctx : sequential_generatedRoot_292Ctx) : Int :=
  2

def sequential_generatedRoot_292.w_11_0  (ctx : sequential_generatedRoot_292Ctx) : Bool :=
  decide ((sequential_generatedRoot_292.w_9_0 ctx) = (sequential_generatedRoot_292.w_10_0 ctx))

def sequential_generatedRoot_292.outs  (ctx : sequential_generatedRoot_292Ctx) : Bool :=
  (sequential_generatedRoot_292.w_11_0 ctx)

structure sequential_generatedRoot_292.Facts_0  (ctx : sequential_generatedRoot_292Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_251 params i_0 i (current, ((sequential_generatedRoot_292.w_3_0 ctx), ((sequential_generatedRoot_292.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_292.w_2_0 ctx) (sequential_generatedRoot_292.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_292.outs ctx

theorem sequential_generatedRoot_292.facts_0  {ctx : sequential_generatedRoot_292Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_292.constraints_0 params i_0 (sequential_generatedRoot_292.w_3_0 ctx) (sequential_generatedRoot_292.w_2_0 ctx) (sequential_generatedRoot_292.w_4_0 ctx) (sequential_generatedRoot_292.w_5_0 ctx) (sequential_generatedRoot_292.w_11_0 ctx) outputs) :
    sequential_generatedRoot_292.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_292.Facts  (ctx : sequential_generatedRoot_292Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_292.Facts_0 ctx outputs

theorem sequential_generatedRoot_292.facts  (ctx : sequential_generatedRoot_292Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_292.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_292.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_292.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_294.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_294.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_294Ctx" => (sequential_generatedRoot_294.Ctx params i_0)

def sequential_generatedRoot_294.w_0_0  (ctx : sequential_generatedRoot_294Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_294.w_3_0  (ctx : sequential_generatedRoot_294Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_294.w_1_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  if (sequential_generatedRoot_294.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_294.w_2_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  0

def sequential_generatedRoot_294.w_4_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_294.w_5_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_294.w_6_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  1

def sequential_generatedRoot_294.w_7_0  (ctx : sequential_generatedRoot_294Ctx) : Bool :=
  decide ((sequential_generatedRoot_294.w_5_0 ctx) = (sequential_generatedRoot_294.w_6_0 ctx))

def sequential_generatedRoot_294.w_8_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  if (sequential_generatedRoot_294.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_294.w_9_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  ((sequential_generatedRoot_294.w_1_0 ctx) + (sequential_generatedRoot_294.w_8_0 ctx))

def sequential_generatedRoot_294.w_10_0  (ctx : sequential_generatedRoot_294Ctx) : Int :=
  2

def sequential_generatedRoot_294.w_11_0  (ctx : sequential_generatedRoot_294Ctx) : Bool :=
  decide ((sequential_generatedRoot_294.w_9_0 ctx) = (sequential_generatedRoot_294.w_10_0 ctx))

def sequential_generatedRoot_294.outs  (ctx : sequential_generatedRoot_294Ctx) : Bool :=
  (sequential_generatedRoot_294.w_11_0 ctx)

structure sequential_generatedRoot_294.Facts_0  (ctx : sequential_generatedRoot_294Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_252 params i_0 i (current, ((sequential_generatedRoot_294.w_3_0 ctx), ((sequential_generatedRoot_294.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_294.w_2_0 ctx) (sequential_generatedRoot_294.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_294.outs ctx

theorem sequential_generatedRoot_294.facts_0  {ctx : sequential_generatedRoot_294Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_294.constraints_0 params i_0 (sequential_generatedRoot_294.w_3_0 ctx) (sequential_generatedRoot_294.w_2_0 ctx) (sequential_generatedRoot_294.w_4_0 ctx) (sequential_generatedRoot_294.w_5_0 ctx) (sequential_generatedRoot_294.w_11_0 ctx) outputs) :
    sequential_generatedRoot_294.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_294.Facts  (ctx : sequential_generatedRoot_294Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_294.Facts_0 ctx outputs

theorem sequential_generatedRoot_294.facts  (ctx : sequential_generatedRoot_294Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_294.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_294.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_294.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_296.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_296.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_296Ctx" => (sequential_generatedRoot_296.Ctx params i_0)

def sequential_generatedRoot_296.w_0_0  (ctx : sequential_generatedRoot_296Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_296.w_3_0  (ctx : sequential_generatedRoot_296Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_296.w_1_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  if (sequential_generatedRoot_296.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_296.w_2_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  0

def sequential_generatedRoot_296.w_4_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_296.w_5_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_296.w_6_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  1

def sequential_generatedRoot_296.w_7_0  (ctx : sequential_generatedRoot_296Ctx) : Bool :=
  decide ((sequential_generatedRoot_296.w_5_0 ctx) = (sequential_generatedRoot_296.w_6_0 ctx))

def sequential_generatedRoot_296.w_8_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  if (sequential_generatedRoot_296.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_296.w_9_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  ((sequential_generatedRoot_296.w_1_0 ctx) + (sequential_generatedRoot_296.w_8_0 ctx))

def sequential_generatedRoot_296.w_10_0  (ctx : sequential_generatedRoot_296Ctx) : Int :=
  2

def sequential_generatedRoot_296.w_11_0  (ctx : sequential_generatedRoot_296Ctx) : Bool :=
  decide ((sequential_generatedRoot_296.w_9_0 ctx) = (sequential_generatedRoot_296.w_10_0 ctx))

def sequential_generatedRoot_296.outs  (ctx : sequential_generatedRoot_296Ctx) : Bool :=
  (sequential_generatedRoot_296.w_11_0 ctx)

structure sequential_generatedRoot_296.Facts_0  (ctx : sequential_generatedRoot_296Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_253 params i_0 i (current, ((sequential_generatedRoot_296.w_3_0 ctx), ((sequential_generatedRoot_296.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_296.w_2_0 ctx) (sequential_generatedRoot_296.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_296.outs ctx

theorem sequential_generatedRoot_296.facts_0  {ctx : sequential_generatedRoot_296Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_296.constraints_0 params i_0 (sequential_generatedRoot_296.w_3_0 ctx) (sequential_generatedRoot_296.w_2_0 ctx) (sequential_generatedRoot_296.w_4_0 ctx) (sequential_generatedRoot_296.w_5_0 ctx) (sequential_generatedRoot_296.w_11_0 ctx) outputs) :
    sequential_generatedRoot_296.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_296.Facts  (ctx : sequential_generatedRoot_296Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_296.Facts_0 ctx outputs

theorem sequential_generatedRoot_296.facts  (ctx : sequential_generatedRoot_296Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_296.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_296.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_296.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_298.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_298.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_298Ctx" => (sequential_generatedRoot_298.Ctx params i_0)

def sequential_generatedRoot_298.w_0_0  (ctx : sequential_generatedRoot_298Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_298.w_3_0  (ctx : sequential_generatedRoot_298Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_298.w_1_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  if (sequential_generatedRoot_298.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_298.w_2_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  0

def sequential_generatedRoot_298.w_4_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_298.w_5_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_298.w_6_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  1

def sequential_generatedRoot_298.w_7_0  (ctx : sequential_generatedRoot_298Ctx) : Bool :=
  decide ((sequential_generatedRoot_298.w_5_0 ctx) = (sequential_generatedRoot_298.w_6_0 ctx))

def sequential_generatedRoot_298.w_8_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  if (sequential_generatedRoot_298.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_298.w_9_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  ((sequential_generatedRoot_298.w_1_0 ctx) + (sequential_generatedRoot_298.w_8_0 ctx))

def sequential_generatedRoot_298.w_10_0  (ctx : sequential_generatedRoot_298Ctx) : Int :=
  2

def sequential_generatedRoot_298.w_11_0  (ctx : sequential_generatedRoot_298Ctx) : Bool :=
  decide ((sequential_generatedRoot_298.w_9_0 ctx) = (sequential_generatedRoot_298.w_10_0 ctx))

def sequential_generatedRoot_298.outs  (ctx : sequential_generatedRoot_298Ctx) : Bool :=
  (sequential_generatedRoot_298.w_11_0 ctx)

structure sequential_generatedRoot_298.Facts_0  (ctx : sequential_generatedRoot_298Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_254 params i_0 i (current, ((sequential_generatedRoot_298.w_3_0 ctx), ((sequential_generatedRoot_298.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_298.w_2_0 ctx) (sequential_generatedRoot_298.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_298.outs ctx

theorem sequential_generatedRoot_298.facts_0  {ctx : sequential_generatedRoot_298Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_298.constraints_0 params i_0 (sequential_generatedRoot_298.w_3_0 ctx) (sequential_generatedRoot_298.w_2_0 ctx) (sequential_generatedRoot_298.w_4_0 ctx) (sequential_generatedRoot_298.w_5_0 ctx) (sequential_generatedRoot_298.w_11_0 ctx) outputs) :
    sequential_generatedRoot_298.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_298.Facts  (ctx : sequential_generatedRoot_298Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_298.Facts_0 ctx outputs

theorem sequential_generatedRoot_298.facts  (ctx : sequential_generatedRoot_298Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_298.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_298.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_298.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_300.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_300.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_300Ctx" => (sequential_generatedRoot_300.Ctx params i_0)

def sequential_generatedRoot_300.w_0_0  (ctx : sequential_generatedRoot_300Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_300.w_3_0  (ctx : sequential_generatedRoot_300Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_300.w_1_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  if (sequential_generatedRoot_300.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_300.w_2_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  0

def sequential_generatedRoot_300.w_4_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_300.w_5_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_300.w_6_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  1

def sequential_generatedRoot_300.w_7_0  (ctx : sequential_generatedRoot_300Ctx) : Bool :=
  decide ((sequential_generatedRoot_300.w_5_0 ctx) = (sequential_generatedRoot_300.w_6_0 ctx))

def sequential_generatedRoot_300.w_8_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  if (sequential_generatedRoot_300.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_300.w_9_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  ((sequential_generatedRoot_300.w_1_0 ctx) + (sequential_generatedRoot_300.w_8_0 ctx))

def sequential_generatedRoot_300.w_10_0  (ctx : sequential_generatedRoot_300Ctx) : Int :=
  2

def sequential_generatedRoot_300.w_11_0  (ctx : sequential_generatedRoot_300Ctx) : Bool :=
  decide ((sequential_generatedRoot_300.w_9_0 ctx) = (sequential_generatedRoot_300.w_10_0 ctx))

def sequential_generatedRoot_300.outs  (ctx : sequential_generatedRoot_300Ctx) : Bool :=
  (sequential_generatedRoot_300.w_11_0 ctx)

structure sequential_generatedRoot_300.Facts_0  (ctx : sequential_generatedRoot_300Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_255 params i_0 i (current, ((sequential_generatedRoot_300.w_3_0 ctx), ((sequential_generatedRoot_300.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_300.w_2_0 ctx) (sequential_generatedRoot_300.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_300.outs ctx

theorem sequential_generatedRoot_300.facts_0  {ctx : sequential_generatedRoot_300Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_300.constraints_0 params i_0 (sequential_generatedRoot_300.w_3_0 ctx) (sequential_generatedRoot_300.w_2_0 ctx) (sequential_generatedRoot_300.w_4_0 ctx) (sequential_generatedRoot_300.w_5_0 ctx) (sequential_generatedRoot_300.w_11_0 ctx) outputs) :
    sequential_generatedRoot_300.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_300.Facts  (ctx : sequential_generatedRoot_300Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_300.Facts_0 ctx outputs

theorem sequential_generatedRoot_300.facts  (ctx : sequential_generatedRoot_300Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_300.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_300.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_300.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_302.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_302.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_302Ctx" => (sequential_generatedRoot_302.Ctx params i_0)

def sequential_generatedRoot_302.w_0_0  (ctx : sequential_generatedRoot_302Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_302.w_3_0  (ctx : sequential_generatedRoot_302Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_302.w_1_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  if (sequential_generatedRoot_302.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_302.w_2_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  0

def sequential_generatedRoot_302.w_4_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_302.w_5_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_302.w_6_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  1

def sequential_generatedRoot_302.w_7_0  (ctx : sequential_generatedRoot_302Ctx) : Bool :=
  decide ((sequential_generatedRoot_302.w_5_0 ctx) = (sequential_generatedRoot_302.w_6_0 ctx))

def sequential_generatedRoot_302.w_8_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  if (sequential_generatedRoot_302.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_302.w_9_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  ((sequential_generatedRoot_302.w_1_0 ctx) + (sequential_generatedRoot_302.w_8_0 ctx))

def sequential_generatedRoot_302.w_10_0  (ctx : sequential_generatedRoot_302Ctx) : Int :=
  2

def sequential_generatedRoot_302.w_11_0  (ctx : sequential_generatedRoot_302Ctx) : Bool :=
  decide ((sequential_generatedRoot_302.w_9_0 ctx) = (sequential_generatedRoot_302.w_10_0 ctx))

def sequential_generatedRoot_302.outs  (ctx : sequential_generatedRoot_302Ctx) : Bool :=
  (sequential_generatedRoot_302.w_11_0 ctx)

structure sequential_generatedRoot_302.Facts_0  (ctx : sequential_generatedRoot_302Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_256 params i_0 i (current, ((sequential_generatedRoot_302.w_3_0 ctx), ((sequential_generatedRoot_302.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_302.w_2_0 ctx) (sequential_generatedRoot_302.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_302.outs ctx

theorem sequential_generatedRoot_302.facts_0  {ctx : sequential_generatedRoot_302Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_302.constraints_0 params i_0 (sequential_generatedRoot_302.w_3_0 ctx) (sequential_generatedRoot_302.w_2_0 ctx) (sequential_generatedRoot_302.w_4_0 ctx) (sequential_generatedRoot_302.w_5_0 ctx) (sequential_generatedRoot_302.w_11_0 ctx) outputs) :
    sequential_generatedRoot_302.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_302.Facts  (ctx : sequential_generatedRoot_302Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_302.Facts_0 ctx outputs

theorem sequential_generatedRoot_302.facts  (ctx : sequential_generatedRoot_302Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_302.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_302.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_302.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_304.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_304.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_304Ctx" => (sequential_generatedRoot_304.Ctx params i_0)

def sequential_generatedRoot_304.w_0_0  (ctx : sequential_generatedRoot_304Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_304.w_3_0  (ctx : sequential_generatedRoot_304Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_304.w_1_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  if (sequential_generatedRoot_304.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_304.w_2_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  0

def sequential_generatedRoot_304.w_4_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_304.w_5_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_304.w_6_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  1

def sequential_generatedRoot_304.w_7_0  (ctx : sequential_generatedRoot_304Ctx) : Bool :=
  decide ((sequential_generatedRoot_304.w_5_0 ctx) = (sequential_generatedRoot_304.w_6_0 ctx))

def sequential_generatedRoot_304.w_8_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  if (sequential_generatedRoot_304.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_304.w_9_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  ((sequential_generatedRoot_304.w_1_0 ctx) + (sequential_generatedRoot_304.w_8_0 ctx))

def sequential_generatedRoot_304.w_10_0  (ctx : sequential_generatedRoot_304Ctx) : Int :=
  2

def sequential_generatedRoot_304.w_11_0  (ctx : sequential_generatedRoot_304Ctx) : Bool :=
  decide ((sequential_generatedRoot_304.w_9_0 ctx) = (sequential_generatedRoot_304.w_10_0 ctx))

def sequential_generatedRoot_304.outs  (ctx : sequential_generatedRoot_304Ctx) : Bool :=
  (sequential_generatedRoot_304.w_11_0 ctx)

structure sequential_generatedRoot_304.Facts_0  (ctx : sequential_generatedRoot_304Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_257 params i_0 i (current, ((sequential_generatedRoot_304.w_3_0 ctx), ((sequential_generatedRoot_304.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_304.w_2_0 ctx) (sequential_generatedRoot_304.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_304.outs ctx

theorem sequential_generatedRoot_304.facts_0  {ctx : sequential_generatedRoot_304Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_304.constraints_0 params i_0 (sequential_generatedRoot_304.w_3_0 ctx) (sequential_generatedRoot_304.w_2_0 ctx) (sequential_generatedRoot_304.w_4_0 ctx) (sequential_generatedRoot_304.w_5_0 ctx) (sequential_generatedRoot_304.w_11_0 ctx) outputs) :
    sequential_generatedRoot_304.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_304.Facts  (ctx : sequential_generatedRoot_304Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_304.Facts_0 ctx outputs

theorem sequential_generatedRoot_304.facts  (ctx : sequential_generatedRoot_304Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_304.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_304.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_304.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_306.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_306.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_306Ctx" => (sequential_generatedRoot_306.Ctx params i_0)

def sequential_generatedRoot_306.w_0_0  (ctx : sequential_generatedRoot_306Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_306.w_3_0  (ctx : sequential_generatedRoot_306Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_306.w_1_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  if (sequential_generatedRoot_306.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_306.w_2_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  0

def sequential_generatedRoot_306.w_4_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_306.w_5_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_306.w_6_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  1

def sequential_generatedRoot_306.w_7_0  (ctx : sequential_generatedRoot_306Ctx) : Bool :=
  decide ((sequential_generatedRoot_306.w_5_0 ctx) = (sequential_generatedRoot_306.w_6_0 ctx))

def sequential_generatedRoot_306.w_8_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  if (sequential_generatedRoot_306.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_306.w_9_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  ((sequential_generatedRoot_306.w_1_0 ctx) + (sequential_generatedRoot_306.w_8_0 ctx))

def sequential_generatedRoot_306.w_10_0  (ctx : sequential_generatedRoot_306Ctx) : Int :=
  2

def sequential_generatedRoot_306.w_11_0  (ctx : sequential_generatedRoot_306Ctx) : Bool :=
  decide ((sequential_generatedRoot_306.w_9_0 ctx) = (sequential_generatedRoot_306.w_10_0 ctx))

def sequential_generatedRoot_306.outs  (ctx : sequential_generatedRoot_306Ctx) : Bool :=
  (sequential_generatedRoot_306.w_11_0 ctx)

structure sequential_generatedRoot_306.Facts_0  (ctx : sequential_generatedRoot_306Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_258 params i_0 i (current, ((sequential_generatedRoot_306.w_3_0 ctx), ((sequential_generatedRoot_306.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_306.w_2_0 ctx) (sequential_generatedRoot_306.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_306.outs ctx

theorem sequential_generatedRoot_306.facts_0  {ctx : sequential_generatedRoot_306Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_306.constraints_0 params i_0 (sequential_generatedRoot_306.w_3_0 ctx) (sequential_generatedRoot_306.w_2_0 ctx) (sequential_generatedRoot_306.w_4_0 ctx) (sequential_generatedRoot_306.w_5_0 ctx) (sequential_generatedRoot_306.w_11_0 ctx) outputs) :
    sequential_generatedRoot_306.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_306.Facts  (ctx : sequential_generatedRoot_306Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_306.Facts_0 ctx outputs

theorem sequential_generatedRoot_306.facts  (ctx : sequential_generatedRoot_306Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_306.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_306.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_306.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_308.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_308.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_308Ctx" => (sequential_generatedRoot_308.Ctx params i_0)

def sequential_generatedRoot_308.w_0_0  (ctx : sequential_generatedRoot_308Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_308.w_3_0  (ctx : sequential_generatedRoot_308Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_308.w_1_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  if (sequential_generatedRoot_308.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_308.w_2_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  0

def sequential_generatedRoot_308.w_4_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_308.w_5_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_308.w_6_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  1

def sequential_generatedRoot_308.w_7_0  (ctx : sequential_generatedRoot_308Ctx) : Bool :=
  decide ((sequential_generatedRoot_308.w_5_0 ctx) = (sequential_generatedRoot_308.w_6_0 ctx))

def sequential_generatedRoot_308.w_8_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  if (sequential_generatedRoot_308.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_308.w_9_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  ((sequential_generatedRoot_308.w_1_0 ctx) + (sequential_generatedRoot_308.w_8_0 ctx))

def sequential_generatedRoot_308.w_10_0  (ctx : sequential_generatedRoot_308Ctx) : Int :=
  2

def sequential_generatedRoot_308.w_11_0  (ctx : sequential_generatedRoot_308Ctx) : Bool :=
  decide ((sequential_generatedRoot_308.w_9_0 ctx) = (sequential_generatedRoot_308.w_10_0 ctx))

def sequential_generatedRoot_308.outs  (ctx : sequential_generatedRoot_308Ctx) : Bool :=
  (sequential_generatedRoot_308.w_11_0 ctx)

structure sequential_generatedRoot_308.Facts_0  (ctx : sequential_generatedRoot_308Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_259 params i_0 i (current, ((sequential_generatedRoot_308.w_3_0 ctx), ((sequential_generatedRoot_308.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_308.w_2_0 ctx) (sequential_generatedRoot_308.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_308.outs ctx

theorem sequential_generatedRoot_308.facts_0  {ctx : sequential_generatedRoot_308Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_308.constraints_0 params i_0 (sequential_generatedRoot_308.w_3_0 ctx) (sequential_generatedRoot_308.w_2_0 ctx) (sequential_generatedRoot_308.w_4_0 ctx) (sequential_generatedRoot_308.w_5_0 ctx) (sequential_generatedRoot_308.w_11_0 ctx) outputs) :
    sequential_generatedRoot_308.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_308.Facts  (ctx : sequential_generatedRoot_308Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_308.Facts_0 ctx outputs

theorem sequential_generatedRoot_308.facts  (ctx : sequential_generatedRoot_308Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_308.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_308.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_308.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_310.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_310.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_310Ctx" => (sequential_generatedRoot_310.Ctx params i_0)

def sequential_generatedRoot_310.w_0_0  (ctx : sequential_generatedRoot_310Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_310.w_3_0  (ctx : sequential_generatedRoot_310Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_310.w_1_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  if (sequential_generatedRoot_310.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_310.w_2_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  0

def sequential_generatedRoot_310.w_4_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_310.w_5_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_310.w_6_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  1

def sequential_generatedRoot_310.w_7_0  (ctx : sequential_generatedRoot_310Ctx) : Bool :=
  decide ((sequential_generatedRoot_310.w_5_0 ctx) = (sequential_generatedRoot_310.w_6_0 ctx))

def sequential_generatedRoot_310.w_8_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  if (sequential_generatedRoot_310.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_310.w_9_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  ((sequential_generatedRoot_310.w_1_0 ctx) + (sequential_generatedRoot_310.w_8_0 ctx))

def sequential_generatedRoot_310.w_10_0  (ctx : sequential_generatedRoot_310Ctx) : Int :=
  2

def sequential_generatedRoot_310.w_11_0  (ctx : sequential_generatedRoot_310Ctx) : Bool :=
  decide ((sequential_generatedRoot_310.w_9_0 ctx) = (sequential_generatedRoot_310.w_10_0 ctx))

def sequential_generatedRoot_310.outs  (ctx : sequential_generatedRoot_310Ctx) : Bool :=
  (sequential_generatedRoot_310.w_11_0 ctx)

structure sequential_generatedRoot_310.Facts_0  (ctx : sequential_generatedRoot_310Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_260 params i_0 i (current, ((sequential_generatedRoot_310.w_3_0 ctx), ((sequential_generatedRoot_310.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_310.w_2_0 ctx) (sequential_generatedRoot_310.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_310.outs ctx

theorem sequential_generatedRoot_310.facts_0  {ctx : sequential_generatedRoot_310Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_310.constraints_0 params i_0 (sequential_generatedRoot_310.w_3_0 ctx) (sequential_generatedRoot_310.w_2_0 ctx) (sequential_generatedRoot_310.w_4_0 ctx) (sequential_generatedRoot_310.w_5_0 ctx) (sequential_generatedRoot_310.w_11_0 ctx) outputs) :
    sequential_generatedRoot_310.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_310.Facts  (ctx : sequential_generatedRoot_310Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_310.Facts_0 ctx outputs

theorem sequential_generatedRoot_310.facts  (ctx : sequential_generatedRoot_310Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_310.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_310.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_310.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_312.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_312.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_312Ctx" => (sequential_generatedRoot_312.Ctx params i_0)

def sequential_generatedRoot_312.w_0_0  (ctx : sequential_generatedRoot_312Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_312.w_3_0  (ctx : sequential_generatedRoot_312Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_312.w_1_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  if (sequential_generatedRoot_312.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_312.w_2_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  0

def sequential_generatedRoot_312.w_4_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_312.w_5_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_312.w_6_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  1

def sequential_generatedRoot_312.w_7_0  (ctx : sequential_generatedRoot_312Ctx) : Bool :=
  decide ((sequential_generatedRoot_312.w_5_0 ctx) = (sequential_generatedRoot_312.w_6_0 ctx))

def sequential_generatedRoot_312.w_8_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  if (sequential_generatedRoot_312.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_312.w_9_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  ((sequential_generatedRoot_312.w_1_0 ctx) + (sequential_generatedRoot_312.w_8_0 ctx))

def sequential_generatedRoot_312.w_10_0  (ctx : sequential_generatedRoot_312Ctx) : Int :=
  2

def sequential_generatedRoot_312.w_11_0  (ctx : sequential_generatedRoot_312Ctx) : Bool :=
  decide ((sequential_generatedRoot_312.w_9_0 ctx) = (sequential_generatedRoot_312.w_10_0 ctx))

def sequential_generatedRoot_312.outs  (ctx : sequential_generatedRoot_312Ctx) : Bool :=
  (sequential_generatedRoot_312.w_11_0 ctx)

structure sequential_generatedRoot_312.Facts_0  (ctx : sequential_generatedRoot_312Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_261 params i_0 i (current, ((sequential_generatedRoot_312.w_3_0 ctx), ((sequential_generatedRoot_312.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_312.w_2_0 ctx) (sequential_generatedRoot_312.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_312.outs ctx

theorem sequential_generatedRoot_312.facts_0  {ctx : sequential_generatedRoot_312Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_312.constraints_0 params i_0 (sequential_generatedRoot_312.w_3_0 ctx) (sequential_generatedRoot_312.w_2_0 ctx) (sequential_generatedRoot_312.w_4_0 ctx) (sequential_generatedRoot_312.w_5_0 ctx) (sequential_generatedRoot_312.w_11_0 ctx) outputs) :
    sequential_generatedRoot_312.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_312.Facts  (ctx : sequential_generatedRoot_312Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_312.Facts_0 ctx outputs

theorem sequential_generatedRoot_312.facts  (ctx : sequential_generatedRoot_312Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_312.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_312.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_312.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_314.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_314.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_314Ctx" => (sequential_generatedRoot_314.Ctx params i_0)

def sequential_generatedRoot_314.w_0_0  (ctx : sequential_generatedRoot_314Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_314.w_3_0  (ctx : sequential_generatedRoot_314Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_314.w_1_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  if (sequential_generatedRoot_314.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_314.w_2_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  0

def sequential_generatedRoot_314.w_4_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_314.w_5_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_314.w_6_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  1

def sequential_generatedRoot_314.w_7_0  (ctx : sequential_generatedRoot_314Ctx) : Bool :=
  decide ((sequential_generatedRoot_314.w_5_0 ctx) = (sequential_generatedRoot_314.w_6_0 ctx))

def sequential_generatedRoot_314.w_8_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  if (sequential_generatedRoot_314.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_314.w_9_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  ((sequential_generatedRoot_314.w_1_0 ctx) + (sequential_generatedRoot_314.w_8_0 ctx))

def sequential_generatedRoot_314.w_10_0  (ctx : sequential_generatedRoot_314Ctx) : Int :=
  2

def sequential_generatedRoot_314.w_11_0  (ctx : sequential_generatedRoot_314Ctx) : Bool :=
  decide ((sequential_generatedRoot_314.w_9_0 ctx) = (sequential_generatedRoot_314.w_10_0 ctx))

def sequential_generatedRoot_314.outs  (ctx : sequential_generatedRoot_314Ctx) : Bool :=
  (sequential_generatedRoot_314.w_11_0 ctx)

structure sequential_generatedRoot_314.Facts_0  (ctx : sequential_generatedRoot_314Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_262 params i_0 i (current, ((sequential_generatedRoot_314.w_3_0 ctx), ((sequential_generatedRoot_314.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_314.w_2_0 ctx) (sequential_generatedRoot_314.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_314.outs ctx

theorem sequential_generatedRoot_314.facts_0  {ctx : sequential_generatedRoot_314Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_314.constraints_0 params i_0 (sequential_generatedRoot_314.w_3_0 ctx) (sequential_generatedRoot_314.w_2_0 ctx) (sequential_generatedRoot_314.w_4_0 ctx) (sequential_generatedRoot_314.w_5_0 ctx) (sequential_generatedRoot_314.w_11_0 ctx) outputs) :
    sequential_generatedRoot_314.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_314.Facts  (ctx : sequential_generatedRoot_314Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_314.Facts_0 ctx outputs

theorem sequential_generatedRoot_314.facts  (ctx : sequential_generatedRoot_314Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_314.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_314.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_314.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_316.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_316.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_316Ctx" => (sequential_generatedRoot_316.Ctx params i_0)

def sequential_generatedRoot_316.w_0_0  (ctx : sequential_generatedRoot_316Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_316.w_3_0  (ctx : sequential_generatedRoot_316Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_316.w_1_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  if (sequential_generatedRoot_316.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_316.w_2_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  0

def sequential_generatedRoot_316.w_4_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_316.w_5_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_316.w_6_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  1

def sequential_generatedRoot_316.w_7_0  (ctx : sequential_generatedRoot_316Ctx) : Bool :=
  decide ((sequential_generatedRoot_316.w_5_0 ctx) = (sequential_generatedRoot_316.w_6_0 ctx))

def sequential_generatedRoot_316.w_8_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  if (sequential_generatedRoot_316.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_316.w_9_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  ((sequential_generatedRoot_316.w_1_0 ctx) + (sequential_generatedRoot_316.w_8_0 ctx))

def sequential_generatedRoot_316.w_10_0  (ctx : sequential_generatedRoot_316Ctx) : Int :=
  2

def sequential_generatedRoot_316.w_11_0  (ctx : sequential_generatedRoot_316Ctx) : Bool :=
  decide ((sequential_generatedRoot_316.w_9_0 ctx) = (sequential_generatedRoot_316.w_10_0 ctx))

def sequential_generatedRoot_316.outs  (ctx : sequential_generatedRoot_316Ctx) : Bool :=
  (sequential_generatedRoot_316.w_11_0 ctx)

structure sequential_generatedRoot_316.Facts_0  (ctx : sequential_generatedRoot_316Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_263 params i_0 i (current, ((sequential_generatedRoot_316.w_3_0 ctx), ((sequential_generatedRoot_316.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_316.w_2_0 ctx) (sequential_generatedRoot_316.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_316.outs ctx

theorem sequential_generatedRoot_316.facts_0  {ctx : sequential_generatedRoot_316Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_316.constraints_0 params i_0 (sequential_generatedRoot_316.w_3_0 ctx) (sequential_generatedRoot_316.w_2_0 ctx) (sequential_generatedRoot_316.w_4_0 ctx) (sequential_generatedRoot_316.w_5_0 ctx) (sequential_generatedRoot_316.w_11_0 ctx) outputs) :
    sequential_generatedRoot_316.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_316.Facts  (ctx : sequential_generatedRoot_316Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_316.Facts_0 ctx outputs

theorem sequential_generatedRoot_316.facts  (ctx : sequential_generatedRoot_316Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_316.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_316.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_316.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_318.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_318.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_318Ctx" => (sequential_generatedRoot_318.Ctx params i_0)

def sequential_generatedRoot_318.w_0_0  (ctx : sequential_generatedRoot_318Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_318.w_3_0  (ctx : sequential_generatedRoot_318Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_318.w_1_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  if (sequential_generatedRoot_318.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_318.w_2_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  0

def sequential_generatedRoot_318.w_4_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_318.w_5_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_318.w_6_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  1

def sequential_generatedRoot_318.w_7_0  (ctx : sequential_generatedRoot_318Ctx) : Bool :=
  decide ((sequential_generatedRoot_318.w_5_0 ctx) = (sequential_generatedRoot_318.w_6_0 ctx))

def sequential_generatedRoot_318.w_8_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  if (sequential_generatedRoot_318.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_318.w_9_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  ((sequential_generatedRoot_318.w_1_0 ctx) + (sequential_generatedRoot_318.w_8_0 ctx))

def sequential_generatedRoot_318.w_10_0  (ctx : sequential_generatedRoot_318Ctx) : Int :=
  2

def sequential_generatedRoot_318.w_11_0  (ctx : sequential_generatedRoot_318Ctx) : Bool :=
  decide ((sequential_generatedRoot_318.w_9_0 ctx) = (sequential_generatedRoot_318.w_10_0 ctx))

def sequential_generatedRoot_318.outs  (ctx : sequential_generatedRoot_318Ctx) : Bool :=
  (sequential_generatedRoot_318.w_11_0 ctx)

structure sequential_generatedRoot_318.Facts_0  (ctx : sequential_generatedRoot_318Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_264 params i_0 i (current, ((sequential_generatedRoot_318.w_3_0 ctx), ((sequential_generatedRoot_318.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_318.w_2_0 ctx) (sequential_generatedRoot_318.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_318.outs ctx

theorem sequential_generatedRoot_318.facts_0  {ctx : sequential_generatedRoot_318Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_318.constraints_0 params i_0 (sequential_generatedRoot_318.w_3_0 ctx) (sequential_generatedRoot_318.w_2_0 ctx) (sequential_generatedRoot_318.w_4_0 ctx) (sequential_generatedRoot_318.w_5_0 ctx) (sequential_generatedRoot_318.w_11_0 ctx) outputs) :
    sequential_generatedRoot_318.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_318.Facts  (ctx : sequential_generatedRoot_318Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_318.Facts_0 ctx outputs

theorem sequential_generatedRoot_318.facts  (ctx : sequential_generatedRoot_318Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_318.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_318.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_318.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_320.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_320.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_320Ctx" => (sequential_generatedRoot_320.Ctx params i_0)

def sequential_generatedRoot_320.w_0_0  (ctx : sequential_generatedRoot_320Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_320.w_3_0  (ctx : sequential_generatedRoot_320Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_320.w_1_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  if (sequential_generatedRoot_320.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_320.w_2_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  0

def sequential_generatedRoot_320.w_4_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_320.w_5_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_320.w_6_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  1

def sequential_generatedRoot_320.w_7_0  (ctx : sequential_generatedRoot_320Ctx) : Bool :=
  decide ((sequential_generatedRoot_320.w_5_0 ctx) = (sequential_generatedRoot_320.w_6_0 ctx))

def sequential_generatedRoot_320.w_8_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  if (sequential_generatedRoot_320.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_320.w_9_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  ((sequential_generatedRoot_320.w_1_0 ctx) + (sequential_generatedRoot_320.w_8_0 ctx))

def sequential_generatedRoot_320.w_10_0  (ctx : sequential_generatedRoot_320Ctx) : Int :=
  2

def sequential_generatedRoot_320.w_11_0  (ctx : sequential_generatedRoot_320Ctx) : Bool :=
  decide ((sequential_generatedRoot_320.w_9_0 ctx) = (sequential_generatedRoot_320.w_10_0 ctx))

def sequential_generatedRoot_320.outs  (ctx : sequential_generatedRoot_320Ctx) : Bool :=
  (sequential_generatedRoot_320.w_11_0 ctx)

structure sequential_generatedRoot_320.Facts_0  (ctx : sequential_generatedRoot_320Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_265 params i_0 i (current, ((sequential_generatedRoot_320.w_3_0 ctx), ((sequential_generatedRoot_320.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_320.w_2_0 ctx) (sequential_generatedRoot_320.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_320.outs ctx

theorem sequential_generatedRoot_320.facts_0  {ctx : sequential_generatedRoot_320Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_320.constraints_0 params i_0 (sequential_generatedRoot_320.w_3_0 ctx) (sequential_generatedRoot_320.w_2_0 ctx) (sequential_generatedRoot_320.w_4_0 ctx) (sequential_generatedRoot_320.w_5_0 ctx) (sequential_generatedRoot_320.w_11_0 ctx) outputs) :
    sequential_generatedRoot_320.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_320.Facts  (ctx : sequential_generatedRoot_320Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_320.Facts_0 ctx outputs

theorem sequential_generatedRoot_320.facts  (ctx : sequential_generatedRoot_320Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_320.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_320.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_320.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_322.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_322.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_322Ctx" => (sequential_generatedRoot_322.Ctx params i_0)

def sequential_generatedRoot_322.w_0_0  (ctx : sequential_generatedRoot_322Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_322.w_3_0  (ctx : sequential_generatedRoot_322Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_322.w_1_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  if (sequential_generatedRoot_322.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_322.w_2_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  0

def sequential_generatedRoot_322.w_4_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_322.w_5_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_322.w_6_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  1

def sequential_generatedRoot_322.w_7_0  (ctx : sequential_generatedRoot_322Ctx) : Bool :=
  decide ((sequential_generatedRoot_322.w_5_0 ctx) = (sequential_generatedRoot_322.w_6_0 ctx))

def sequential_generatedRoot_322.w_8_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  if (sequential_generatedRoot_322.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_322.w_9_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  ((sequential_generatedRoot_322.w_1_0 ctx) + (sequential_generatedRoot_322.w_8_0 ctx))

def sequential_generatedRoot_322.w_10_0  (ctx : sequential_generatedRoot_322Ctx) : Int :=
  2

def sequential_generatedRoot_322.w_11_0  (ctx : sequential_generatedRoot_322Ctx) : Bool :=
  decide ((sequential_generatedRoot_322.w_9_0 ctx) = (sequential_generatedRoot_322.w_10_0 ctx))

def sequential_generatedRoot_322.outs  (ctx : sequential_generatedRoot_322Ctx) : Bool :=
  (sequential_generatedRoot_322.w_11_0 ctx)

structure sequential_generatedRoot_322.Facts_0  (ctx : sequential_generatedRoot_322Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_266 params i_0 i (current, ((sequential_generatedRoot_322.w_3_0 ctx), ((sequential_generatedRoot_322.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_322.w_2_0 ctx) (sequential_generatedRoot_322.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_322.outs ctx

theorem sequential_generatedRoot_322.facts_0  {ctx : sequential_generatedRoot_322Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_322.constraints_0 params i_0 (sequential_generatedRoot_322.w_3_0 ctx) (sequential_generatedRoot_322.w_2_0 ctx) (sequential_generatedRoot_322.w_4_0 ctx) (sequential_generatedRoot_322.w_5_0 ctx) (sequential_generatedRoot_322.w_11_0 ctx) outputs) :
    sequential_generatedRoot_322.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_322.Facts  (ctx : sequential_generatedRoot_322Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_322.Facts_0 ctx outputs

theorem sequential_generatedRoot_322.facts  (ctx : sequential_generatedRoot_322Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_322.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_322.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_322.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_324.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_324.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_324Ctx" => (sequential_generatedRoot_324.Ctx params i_0)

def sequential_generatedRoot_324.w_0_0  (ctx : sequential_generatedRoot_324Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_324.w_3_0  (ctx : sequential_generatedRoot_324Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_324.w_1_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  if (sequential_generatedRoot_324.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_324.w_2_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  0

def sequential_generatedRoot_324.w_4_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_324.w_5_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_324.w_6_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  1

def sequential_generatedRoot_324.w_7_0  (ctx : sequential_generatedRoot_324Ctx) : Bool :=
  decide ((sequential_generatedRoot_324.w_5_0 ctx) = (sequential_generatedRoot_324.w_6_0 ctx))

def sequential_generatedRoot_324.w_8_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  if (sequential_generatedRoot_324.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_324.w_9_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  ((sequential_generatedRoot_324.w_1_0 ctx) + (sequential_generatedRoot_324.w_8_0 ctx))

def sequential_generatedRoot_324.w_10_0  (ctx : sequential_generatedRoot_324Ctx) : Int :=
  2

def sequential_generatedRoot_324.w_11_0  (ctx : sequential_generatedRoot_324Ctx) : Bool :=
  decide ((sequential_generatedRoot_324.w_9_0 ctx) = (sequential_generatedRoot_324.w_10_0 ctx))

def sequential_generatedRoot_324.outs  (ctx : sequential_generatedRoot_324Ctx) : Bool :=
  (sequential_generatedRoot_324.w_11_0 ctx)

structure sequential_generatedRoot_324.Facts_0  (ctx : sequential_generatedRoot_324Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_267 params i_0 i (current, ((sequential_generatedRoot_324.w_3_0 ctx), ((sequential_generatedRoot_324.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_324.w_2_0 ctx) (sequential_generatedRoot_324.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_324.outs ctx

theorem sequential_generatedRoot_324.facts_0  {ctx : sequential_generatedRoot_324Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_324.constraints_0 params i_0 (sequential_generatedRoot_324.w_3_0 ctx) (sequential_generatedRoot_324.w_2_0 ctx) (sequential_generatedRoot_324.w_4_0 ctx) (sequential_generatedRoot_324.w_5_0 ctx) (sequential_generatedRoot_324.w_11_0 ctx) outputs) :
    sequential_generatedRoot_324.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_324.Facts  (ctx : sequential_generatedRoot_324Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_324.Facts_0 ctx outputs

theorem sequential_generatedRoot_324.facts  (ctx : sequential_generatedRoot_324Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_324.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_324.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_324.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_326.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_326.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_326Ctx" => (sequential_generatedRoot_326.Ctx params i_0)

def sequential_generatedRoot_326.w_0_0  (ctx : sequential_generatedRoot_326Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_326.w_3_0  (ctx : sequential_generatedRoot_326Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_326.w_1_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  if (sequential_generatedRoot_326.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_326.w_2_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  0

def sequential_generatedRoot_326.w_4_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_326.w_5_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_326.w_6_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  1

def sequential_generatedRoot_326.w_7_0  (ctx : sequential_generatedRoot_326Ctx) : Bool :=
  decide ((sequential_generatedRoot_326.w_5_0 ctx) = (sequential_generatedRoot_326.w_6_0 ctx))

def sequential_generatedRoot_326.w_8_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  if (sequential_generatedRoot_326.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_326.w_9_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  ((sequential_generatedRoot_326.w_1_0 ctx) + (sequential_generatedRoot_326.w_8_0 ctx))

def sequential_generatedRoot_326.w_10_0  (ctx : sequential_generatedRoot_326Ctx) : Int :=
  2

def sequential_generatedRoot_326.w_11_0  (ctx : sequential_generatedRoot_326Ctx) : Bool :=
  decide ((sequential_generatedRoot_326.w_9_0 ctx) = (sequential_generatedRoot_326.w_10_0 ctx))

def sequential_generatedRoot_326.outs  (ctx : sequential_generatedRoot_326Ctx) : Bool :=
  (sequential_generatedRoot_326.w_11_0 ctx)

structure sequential_generatedRoot_326.Facts_0  (ctx : sequential_generatedRoot_326Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_268 params i_0 i (current, ((sequential_generatedRoot_326.w_3_0 ctx), ((sequential_generatedRoot_326.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_326.w_2_0 ctx) (sequential_generatedRoot_326.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_326.outs ctx

theorem sequential_generatedRoot_326.facts_0  {ctx : sequential_generatedRoot_326Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_326.constraints_0 params i_0 (sequential_generatedRoot_326.w_3_0 ctx) (sequential_generatedRoot_326.w_2_0 ctx) (sequential_generatedRoot_326.w_4_0 ctx) (sequential_generatedRoot_326.w_5_0 ctx) (sequential_generatedRoot_326.w_11_0 ctx) outputs) :
    sequential_generatedRoot_326.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_326.Facts  (ctx : sequential_generatedRoot_326Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_326.Facts_0 ctx outputs

theorem sequential_generatedRoot_326.facts  (ctx : sequential_generatedRoot_326Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_326.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_326.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_326.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_328.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_328.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_328Ctx" => (sequential_generatedRoot_328.Ctx params i_0)

def sequential_generatedRoot_328.w_0_0  (ctx : sequential_generatedRoot_328Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_328.w_3_0  (ctx : sequential_generatedRoot_328Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_328.w_1_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  if (sequential_generatedRoot_328.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_328.w_2_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  0

def sequential_generatedRoot_328.w_4_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_328.w_5_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_328.w_6_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  1

def sequential_generatedRoot_328.w_7_0  (ctx : sequential_generatedRoot_328Ctx) : Bool :=
  decide ((sequential_generatedRoot_328.w_5_0 ctx) = (sequential_generatedRoot_328.w_6_0 ctx))

def sequential_generatedRoot_328.w_8_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  if (sequential_generatedRoot_328.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_328.w_9_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  ((sequential_generatedRoot_328.w_1_0 ctx) + (sequential_generatedRoot_328.w_8_0 ctx))

def sequential_generatedRoot_328.w_10_0  (ctx : sequential_generatedRoot_328Ctx) : Int :=
  2

def sequential_generatedRoot_328.w_11_0  (ctx : sequential_generatedRoot_328Ctx) : Bool :=
  decide ((sequential_generatedRoot_328.w_9_0 ctx) = (sequential_generatedRoot_328.w_10_0 ctx))

def sequential_generatedRoot_328.outs  (ctx : sequential_generatedRoot_328Ctx) : Bool :=
  (sequential_generatedRoot_328.w_11_0 ctx)

structure sequential_generatedRoot_328.Facts_0  (ctx : sequential_generatedRoot_328Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_269 params i_0 i (current, ((sequential_generatedRoot_328.w_3_0 ctx), ((sequential_generatedRoot_328.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_328.w_2_0 ctx) (sequential_generatedRoot_328.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_328.outs ctx

theorem sequential_generatedRoot_328.facts_0  {ctx : sequential_generatedRoot_328Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_328.constraints_0 params i_0 (sequential_generatedRoot_328.w_3_0 ctx) (sequential_generatedRoot_328.w_2_0 ctx) (sequential_generatedRoot_328.w_4_0 ctx) (sequential_generatedRoot_328.w_5_0 ctx) (sequential_generatedRoot_328.w_11_0 ctx) outputs) :
    sequential_generatedRoot_328.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_328.Facts  (ctx : sequential_generatedRoot_328Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_328.Facts_0 ctx outputs

theorem sequential_generatedRoot_328.facts  (ctx : sequential_generatedRoot_328Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_328.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_328.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_328.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_330.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_330.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_330Ctx" => (sequential_generatedRoot_330.Ctx params i_0)

def sequential_generatedRoot_330.w_0_0  (ctx : sequential_generatedRoot_330Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_330.w_3_0  (ctx : sequential_generatedRoot_330Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_330.w_1_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  if (sequential_generatedRoot_330.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_330.w_2_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  0

def sequential_generatedRoot_330.w_4_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_330.w_5_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_330.w_6_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  1

def sequential_generatedRoot_330.w_7_0  (ctx : sequential_generatedRoot_330Ctx) : Bool :=
  decide ((sequential_generatedRoot_330.w_5_0 ctx) = (sequential_generatedRoot_330.w_6_0 ctx))

def sequential_generatedRoot_330.w_8_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  if (sequential_generatedRoot_330.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_330.w_9_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  ((sequential_generatedRoot_330.w_1_0 ctx) + (sequential_generatedRoot_330.w_8_0 ctx))

def sequential_generatedRoot_330.w_10_0  (ctx : sequential_generatedRoot_330Ctx) : Int :=
  2

def sequential_generatedRoot_330.w_11_0  (ctx : sequential_generatedRoot_330Ctx) : Bool :=
  decide ((sequential_generatedRoot_330.w_9_0 ctx) = (sequential_generatedRoot_330.w_10_0 ctx))

def sequential_generatedRoot_330.outs  (ctx : sequential_generatedRoot_330Ctx) : Bool :=
  (sequential_generatedRoot_330.w_11_0 ctx)

structure sequential_generatedRoot_330.Facts_0  (ctx : sequential_generatedRoot_330Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_270 params i_0 i (current, ((sequential_generatedRoot_330.w_3_0 ctx), ((sequential_generatedRoot_330.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_330.w_2_0 ctx) (sequential_generatedRoot_330.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_330.outs ctx

theorem sequential_generatedRoot_330.facts_0  {ctx : sequential_generatedRoot_330Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_330.constraints_0 params i_0 (sequential_generatedRoot_330.w_3_0 ctx) (sequential_generatedRoot_330.w_2_0 ctx) (sequential_generatedRoot_330.w_4_0 ctx) (sequential_generatedRoot_330.w_5_0 ctx) (sequential_generatedRoot_330.w_11_0 ctx) outputs) :
    sequential_generatedRoot_330.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_330.Facts  (ctx : sequential_generatedRoot_330Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_330.Facts_0 ctx outputs

theorem sequential_generatedRoot_330.facts  (ctx : sequential_generatedRoot_330Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_330.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_330.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_330.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_332.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_332.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_332Ctx" => (sequential_generatedRoot_332.Ctx params i_0)

def sequential_generatedRoot_332.w_0_0  (ctx : sequential_generatedRoot_332Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_332.w_3_0  (ctx : sequential_generatedRoot_332Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_332.w_1_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  if (sequential_generatedRoot_332.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_332.w_2_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  0

def sequential_generatedRoot_332.w_4_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_332.w_5_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_332.w_6_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  1

def sequential_generatedRoot_332.w_7_0  (ctx : sequential_generatedRoot_332Ctx) : Bool :=
  decide ((sequential_generatedRoot_332.w_5_0 ctx) = (sequential_generatedRoot_332.w_6_0 ctx))

def sequential_generatedRoot_332.w_8_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  if (sequential_generatedRoot_332.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_332.w_9_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  ((sequential_generatedRoot_332.w_1_0 ctx) + (sequential_generatedRoot_332.w_8_0 ctx))

def sequential_generatedRoot_332.w_10_0  (ctx : sequential_generatedRoot_332Ctx) : Int :=
  2

def sequential_generatedRoot_332.w_11_0  (ctx : sequential_generatedRoot_332Ctx) : Bool :=
  decide ((sequential_generatedRoot_332.w_9_0 ctx) = (sequential_generatedRoot_332.w_10_0 ctx))

def sequential_generatedRoot_332.outs  (ctx : sequential_generatedRoot_332Ctx) : Bool :=
  (sequential_generatedRoot_332.w_11_0 ctx)

structure sequential_generatedRoot_332.Facts_0  (ctx : sequential_generatedRoot_332Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_271 params i_0 i (current, ((sequential_generatedRoot_332.w_3_0 ctx), ((sequential_generatedRoot_332.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_332.w_2_0 ctx) (sequential_generatedRoot_332.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_332.outs ctx

theorem sequential_generatedRoot_332.facts_0  {ctx : sequential_generatedRoot_332Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_332.constraints_0 params i_0 (sequential_generatedRoot_332.w_3_0 ctx) (sequential_generatedRoot_332.w_2_0 ctx) (sequential_generatedRoot_332.w_4_0 ctx) (sequential_generatedRoot_332.w_5_0 ctx) (sequential_generatedRoot_332.w_11_0 ctx) outputs) :
    sequential_generatedRoot_332.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_332.Facts  (ctx : sequential_generatedRoot_332Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_332.Facts_0 ctx outputs

theorem sequential_generatedRoot_332.facts  (ctx : sequential_generatedRoot_332Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_332.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_332.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_332.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_334.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_334.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_334Ctx" => (sequential_generatedRoot_334.Ctx params i_0)

def sequential_generatedRoot_334.w_0_0  (ctx : sequential_generatedRoot_334Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_334.w_3_0  (ctx : sequential_generatedRoot_334Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_334.w_1_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  if (sequential_generatedRoot_334.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_334.w_2_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  0

def sequential_generatedRoot_334.w_4_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_334.w_5_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_334.w_6_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  1

def sequential_generatedRoot_334.w_7_0  (ctx : sequential_generatedRoot_334Ctx) : Bool :=
  decide ((sequential_generatedRoot_334.w_5_0 ctx) = (sequential_generatedRoot_334.w_6_0 ctx))

def sequential_generatedRoot_334.w_8_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  if (sequential_generatedRoot_334.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_334.w_9_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  ((sequential_generatedRoot_334.w_1_0 ctx) + (sequential_generatedRoot_334.w_8_0 ctx))

def sequential_generatedRoot_334.w_10_0  (ctx : sequential_generatedRoot_334Ctx) : Int :=
  2

def sequential_generatedRoot_334.w_11_0  (ctx : sequential_generatedRoot_334Ctx) : Bool :=
  decide ((sequential_generatedRoot_334.w_9_0 ctx) = (sequential_generatedRoot_334.w_10_0 ctx))

def sequential_generatedRoot_334.outs  (ctx : sequential_generatedRoot_334Ctx) : Bool :=
  (sequential_generatedRoot_334.w_11_0 ctx)

structure sequential_generatedRoot_334.Facts_0  (ctx : sequential_generatedRoot_334Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_272 params i_0 i (current, ((sequential_generatedRoot_334.w_3_0 ctx), ((sequential_generatedRoot_334.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_334.w_2_0 ctx) (sequential_generatedRoot_334.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_334.outs ctx

theorem sequential_generatedRoot_334.facts_0  {ctx : sequential_generatedRoot_334Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_334.constraints_0 params i_0 (sequential_generatedRoot_334.w_3_0 ctx) (sequential_generatedRoot_334.w_2_0 ctx) (sequential_generatedRoot_334.w_4_0 ctx) (sequential_generatedRoot_334.w_5_0 ctx) (sequential_generatedRoot_334.w_11_0 ctx) outputs) :
    sequential_generatedRoot_334.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_334.Facts  (ctx : sequential_generatedRoot_334Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_334.Facts_0 ctx outputs

theorem sequential_generatedRoot_334.facts  (ctx : sequential_generatedRoot_334Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_334.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_334.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_334.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_336.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_336.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_336Ctx" => (sequential_generatedRoot_336.Ctx params i_0)

def sequential_generatedRoot_336.w_0_0  (ctx : sequential_generatedRoot_336Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_336.w_3_0  (ctx : sequential_generatedRoot_336Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_336.w_1_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  if (sequential_generatedRoot_336.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_336.w_2_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  0

def sequential_generatedRoot_336.w_4_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_336.w_5_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_336.w_6_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  1

def sequential_generatedRoot_336.w_7_0  (ctx : sequential_generatedRoot_336Ctx) : Bool :=
  decide ((sequential_generatedRoot_336.w_5_0 ctx) = (sequential_generatedRoot_336.w_6_0 ctx))

def sequential_generatedRoot_336.w_8_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  if (sequential_generatedRoot_336.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_336.w_9_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  ((sequential_generatedRoot_336.w_1_0 ctx) + (sequential_generatedRoot_336.w_8_0 ctx))

def sequential_generatedRoot_336.w_10_0  (ctx : sequential_generatedRoot_336Ctx) : Int :=
  2

def sequential_generatedRoot_336.w_11_0  (ctx : sequential_generatedRoot_336Ctx) : Bool :=
  decide ((sequential_generatedRoot_336.w_9_0 ctx) = (sequential_generatedRoot_336.w_10_0 ctx))

def sequential_generatedRoot_336.outs  (ctx : sequential_generatedRoot_336Ctx) : Bool :=
  (sequential_generatedRoot_336.w_11_0 ctx)

structure sequential_generatedRoot_336.Facts_0  (ctx : sequential_generatedRoot_336Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_273 params i_0 i (current, ((sequential_generatedRoot_336.w_3_0 ctx), ((sequential_generatedRoot_336.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_336.w_2_0 ctx) (sequential_generatedRoot_336.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_336.outs ctx

theorem sequential_generatedRoot_336.facts_0  {ctx : sequential_generatedRoot_336Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_336.constraints_0 params i_0 (sequential_generatedRoot_336.w_3_0 ctx) (sequential_generatedRoot_336.w_2_0 ctx) (sequential_generatedRoot_336.w_4_0 ctx) (sequential_generatedRoot_336.w_5_0 ctx) (sequential_generatedRoot_336.w_11_0 ctx) outputs) :
    sequential_generatedRoot_336.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_336.Facts  (ctx : sequential_generatedRoot_336Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_336.Facts_0 ctx outputs

theorem sequential_generatedRoot_336.facts  (ctx : sequential_generatedRoot_336Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_336.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_336.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_336.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_338.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_338.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_338Ctx" => (sequential_generatedRoot_338.Ctx params i_0)

def sequential_generatedRoot_338.w_0_0  (ctx : sequential_generatedRoot_338Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_338.w_3_0  (ctx : sequential_generatedRoot_338Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_338.w_1_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  if (sequential_generatedRoot_338.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_338.w_2_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  0

def sequential_generatedRoot_338.w_4_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_338.w_5_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_338.w_6_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  1

def sequential_generatedRoot_338.w_7_0  (ctx : sequential_generatedRoot_338Ctx) : Bool :=
  decide ((sequential_generatedRoot_338.w_5_0 ctx) = (sequential_generatedRoot_338.w_6_0 ctx))

def sequential_generatedRoot_338.w_8_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  if (sequential_generatedRoot_338.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_338.w_9_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  ((sequential_generatedRoot_338.w_1_0 ctx) + (sequential_generatedRoot_338.w_8_0 ctx))

def sequential_generatedRoot_338.w_10_0  (ctx : sequential_generatedRoot_338Ctx) : Int :=
  2

def sequential_generatedRoot_338.w_11_0  (ctx : sequential_generatedRoot_338Ctx) : Bool :=
  decide ((sequential_generatedRoot_338.w_9_0 ctx) = (sequential_generatedRoot_338.w_10_0 ctx))

def sequential_generatedRoot_338.outs  (ctx : sequential_generatedRoot_338Ctx) : Bool :=
  (sequential_generatedRoot_338.w_11_0 ctx)

structure sequential_generatedRoot_338.Facts_0  (ctx : sequential_generatedRoot_338Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_274 params i_0 i (current, ((sequential_generatedRoot_338.w_3_0 ctx), ((sequential_generatedRoot_338.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_338.w_2_0 ctx) (sequential_generatedRoot_338.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_338.outs ctx

theorem sequential_generatedRoot_338.facts_0  {ctx : sequential_generatedRoot_338Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_338.constraints_0 params i_0 (sequential_generatedRoot_338.w_3_0 ctx) (sequential_generatedRoot_338.w_2_0 ctx) (sequential_generatedRoot_338.w_4_0 ctx) (sequential_generatedRoot_338.w_5_0 ctx) (sequential_generatedRoot_338.w_11_0 ctx) outputs) :
    sequential_generatedRoot_338.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_338.Facts  (ctx : sequential_generatedRoot_338Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_338.Facts_0 ctx outputs

theorem sequential_generatedRoot_338.facts  (ctx : sequential_generatedRoot_338Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_338.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_338.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_338.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_340.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_340.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_340Ctx" => (sequential_generatedRoot_340.Ctx params i_0)

def sequential_generatedRoot_340.w_0_0  (ctx : sequential_generatedRoot_340Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_340.w_3_0  (ctx : sequential_generatedRoot_340Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_340.w_1_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  if (sequential_generatedRoot_340.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_340.w_2_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  0

def sequential_generatedRoot_340.w_4_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_340.w_5_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_340.w_6_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  1

def sequential_generatedRoot_340.w_7_0  (ctx : sequential_generatedRoot_340Ctx) : Bool :=
  decide ((sequential_generatedRoot_340.w_5_0 ctx) = (sequential_generatedRoot_340.w_6_0 ctx))

def sequential_generatedRoot_340.w_8_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  if (sequential_generatedRoot_340.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_340.w_9_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  ((sequential_generatedRoot_340.w_1_0 ctx) + (sequential_generatedRoot_340.w_8_0 ctx))

def sequential_generatedRoot_340.w_10_0  (ctx : sequential_generatedRoot_340Ctx) : Int :=
  2

def sequential_generatedRoot_340.w_11_0  (ctx : sequential_generatedRoot_340Ctx) : Bool :=
  decide ((sequential_generatedRoot_340.w_9_0 ctx) = (sequential_generatedRoot_340.w_10_0 ctx))

def sequential_generatedRoot_340.outs  (ctx : sequential_generatedRoot_340Ctx) : Bool :=
  (sequential_generatedRoot_340.w_11_0 ctx)

structure sequential_generatedRoot_340.Facts_0  (ctx : sequential_generatedRoot_340Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_275 params i_0 i (current, ((sequential_generatedRoot_340.w_3_0 ctx), ((sequential_generatedRoot_340.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_340.w_2_0 ctx) (sequential_generatedRoot_340.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_340.outs ctx

theorem sequential_generatedRoot_340.facts_0  {ctx : sequential_generatedRoot_340Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_340.constraints_0 params i_0 (sequential_generatedRoot_340.w_3_0 ctx) (sequential_generatedRoot_340.w_2_0 ctx) (sequential_generatedRoot_340.w_4_0 ctx) (sequential_generatedRoot_340.w_5_0 ctx) (sequential_generatedRoot_340.w_11_0 ctx) outputs) :
    sequential_generatedRoot_340.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_340.Facts  (ctx : sequential_generatedRoot_340Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_340.Facts_0 ctx outputs

theorem sequential_generatedRoot_340.facts  (ctx : sequential_generatedRoot_340Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_340.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_340.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_340.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_342.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_342.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_342Ctx" => (sequential_generatedRoot_342.Ctx params i_0)

def sequential_generatedRoot_342.w_0_0  (ctx : sequential_generatedRoot_342Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_342.w_3_0  (ctx : sequential_generatedRoot_342Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_342.w_1_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  if (sequential_generatedRoot_342.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_342.w_2_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  0

def sequential_generatedRoot_342.w_4_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_342.w_5_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_342.w_6_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  1

def sequential_generatedRoot_342.w_7_0  (ctx : sequential_generatedRoot_342Ctx) : Bool :=
  decide ((sequential_generatedRoot_342.w_5_0 ctx) = (sequential_generatedRoot_342.w_6_0 ctx))

def sequential_generatedRoot_342.w_8_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  if (sequential_generatedRoot_342.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_342.w_9_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  ((sequential_generatedRoot_342.w_1_0 ctx) + (sequential_generatedRoot_342.w_8_0 ctx))

def sequential_generatedRoot_342.w_10_0  (ctx : sequential_generatedRoot_342Ctx) : Int :=
  2

def sequential_generatedRoot_342.w_11_0  (ctx : sequential_generatedRoot_342Ctx) : Bool :=
  decide ((sequential_generatedRoot_342.w_9_0 ctx) = (sequential_generatedRoot_342.w_10_0 ctx))

def sequential_generatedRoot_342.outs  (ctx : sequential_generatedRoot_342Ctx) : Bool :=
  (sequential_generatedRoot_342.w_11_0 ctx)

structure sequential_generatedRoot_342.Facts_0  (ctx : sequential_generatedRoot_342Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_276 params i_0 i (current, ((sequential_generatedRoot_342.w_3_0 ctx), ((sequential_generatedRoot_342.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_342.outs ctx

theorem sequential_generatedRoot_342.facts_0  {ctx : sequential_generatedRoot_342Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_342.constraints_0 params i_0 (sequential_generatedRoot_342.w_3_0 ctx) (sequential_generatedRoot_342.w_2_0 ctx) (sequential_generatedRoot_342.w_4_0 ctx) (sequential_generatedRoot_342.w_5_0 ctx) (sequential_generatedRoot_342.w_11_0 ctx) outputs) :
    sequential_generatedRoot_342.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_342.Facts  (ctx : sequential_generatedRoot_342Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_342.Facts_0 ctx outputs

theorem sequential_generatedRoot_342.facts  (ctx : sequential_generatedRoot_342Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_342.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_342.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_342.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_344.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 2607022080 → Int) × Unit
  witness : sequential_generatedRoot_344.Witness

section
variable {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_344Ctx" => (sequential_generatedRoot_344.Ctx params i_0)

def sequential_generatedRoot_344.w_0_0  (ctx : sequential_generatedRoot_344Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_344.w_3_0  (ctx : sequential_generatedRoot_344Ctx) : Fin 2607022080 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_344.w_1_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  if (sequential_generatedRoot_344.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_344.w_2_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  0

def sequential_generatedRoot_344.w_4_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_344.w_5_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_344.w_6_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  1

def sequential_generatedRoot_344.w_7_0  (ctx : sequential_generatedRoot_344Ctx) : Bool :=
  decide ((sequential_generatedRoot_344.w_5_0 ctx) = (sequential_generatedRoot_344.w_6_0 ctx))

def sequential_generatedRoot_344.w_8_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  if (sequential_generatedRoot_344.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_344.w_9_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  ((sequential_generatedRoot_344.w_1_0 ctx) + (sequential_generatedRoot_344.w_8_0 ctx))

def sequential_generatedRoot_344.w_10_0  (ctx : sequential_generatedRoot_344Ctx) : Int :=
  2

def sequential_generatedRoot_344.w_11_0  (ctx : sequential_generatedRoot_344Ctx) : Bool :=
  decide ((sequential_generatedRoot_344.w_9_0 ctx) = (sequential_generatedRoot_344.w_10_0 ctx))

def sequential_generatedRoot_344.outs  (ctx : sequential_generatedRoot_344Ctx) : Bool :=
  (sequential_generatedRoot_344.w_11_0 ctx)

structure sequential_generatedRoot_344.Facts_0  (ctx : sequential_generatedRoot_344Ctx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (130)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_277 params i_0 i (current, ((sequential_generatedRoot_344.w_3_0 ctx), ((sequential_generatedRoot_344.w_4_0 ctx), ()))) next) (Int.toNat (130)) (sequential_generatedRoot_344.w_2_0 ctx) (sequential_generatedRoot_344.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_344.outs ctx

theorem sequential_generatedRoot_344.facts_0  {ctx : sequential_generatedRoot_344Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_344.constraints_0 params i_0 (sequential_generatedRoot_344.w_3_0 ctx) (sequential_generatedRoot_344.w_2_0 ctx) (sequential_generatedRoot_344.w_4_0 ctx) (sequential_generatedRoot_344.w_5_0 ctx) (sequential_generatedRoot_344.w_11_0 ctx) outputs) :
    sequential_generatedRoot_344.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_344.Facts  (ctx : sequential_generatedRoot_344Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_344.Facts_0 ctx outputs

theorem sequential_generatedRoot_344.facts  (ctx : sequential_generatedRoot_344Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_344.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_344.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_344.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

section
variable {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx params)

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : Fin 8389632 → Int :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 8389632 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : Fin 1025 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_87_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_89_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_91_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_93_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_95_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_97_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_99_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_101_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_103_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_105_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_107_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_109_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_111_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_113_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_115_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_117_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_119_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_121_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_123_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_125_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_127_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_129_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_131_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_133_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_135_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_137_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_139_0  (ctx : generatedRootCtx) : Fin 318898285953889075200 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_141_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_143_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_145_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_147_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_149_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_151_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_153_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_155_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_157_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_159_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_161_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_163_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_165_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_167_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_169_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_171_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_173_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_175_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_177_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_179_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_181_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_183_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_185_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_187_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_189_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_191_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_193_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_195_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_197_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_199_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_201_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_203_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_205_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_207_0  (ctx : generatedRootCtx) : Fin 8519680 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_209_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_211_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_213_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_215_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_217_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_219_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_221_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_223_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_225_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_227_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_229_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_231_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_233_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_235_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_237_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_239_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_241_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_243_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_245_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_247_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_249_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_251_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_253_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_255_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_257_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_259_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_261_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_263_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_265_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_267_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_269_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_271_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_273_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_275_0  (ctx : generatedRootCtx) : Fin 38598194862489600 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_277_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_279_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_281_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_283_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_285_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_287_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_289_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_291_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_293_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_295_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_297_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_299_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_301_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_303_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_305_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_307_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_309_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_311_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_313_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_315_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_317_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_319_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_321_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_323_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_325_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_327_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_329_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_331_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_333_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_335_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_337_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_339_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_341_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_343_0  (ctx : generatedRootCtx) : Fin 2607022080 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_0_0  (ctx : generatedRootCtx) : Bool :=
  true

def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_3_0

def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_6_0 ctx) then 1 else 0

def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_9_0

def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_9_0 ctx) * (generatedRoot.w_9_0 ctx))

def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Int :=
  1

def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_10_0 ctx) = (generatedRoot.w_11_0 ctx))

def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_12_0 ctx) then 1 else 0

def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_15_0

def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_17_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_15_0 ctx) * (generatedRoot.w_16_0 ctx))

def generatedRoot.w_18_0  (ctx : generatedRootCtx) : Int :=
  1

def generatedRoot.w_19_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_18_0 ctx) - (generatedRoot.w_9_0 ctx))

def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Int :=
  65536

def generatedRoot.w_21_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_19_0 ctx) * (generatedRoot.w_20_0 ctx))

def generatedRoot.w_22_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_17_0 ctx) + (generatedRoot.w_21_0 ctx))

def generatedRoot.w_24_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_24_0 ctx) * (generatedRoot.w_25_0 ctx))

def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_26_0 ctx) * (generatedRoot.w_27_0 ctx))

def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_22_0 ctx) = (generatedRoot.w_28_0 ctx))

def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_29_0 ctx) then 1 else 0

def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_13_0 ctx) + (generatedRoot.w_30_0 ctx))

def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_33_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_31_0 ctx) = (generatedRoot.w_32_0 ctx))

def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_33_0 ctx) then 1 else 0

def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_7_0 ctx) + (generatedRoot.w_34_0 ctx))

def generatedRoot.w_36_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_37_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_35_0 ctx) = (generatedRoot.w_36_0 ctx))

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : Fin 8193 → Int :=
  ctx.witness.w_40_0

def generatedRoot.w_40_1  (ctx : generatedRootCtx) : Fin 8193 → Int :=
  ctx.witness.w_40_1

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : Fin 8389632 → Int :=
  ctx.witness.w_41_0

def generatedRoot.w_42_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_42_0

def generatedRoot.w_43_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_42_0 ctx) then 1 else 0

def generatedRoot.w_45_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_45_0

def generatedRoot.w_46_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_45_0 ctx) * (generatedRoot.w_45_0 ctx))

def generatedRoot.w_47_0  (ctx : generatedRootCtx) : Int :=
  1

def generatedRoot.w_48_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_46_0 ctx) = (generatedRoot.w_47_0 ctx))

def generatedRoot.w_49_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_48_0 ctx) then 1 else 0

def generatedRoot.w_51_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_51_0

def generatedRoot.w_52_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_53_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_51_0 ctx) * (generatedRoot.w_52_0 ctx))

def generatedRoot.w_54_0  (ctx : generatedRootCtx) : Int :=
  1

def generatedRoot.w_55_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_54_0 ctx) - (generatedRoot.w_45_0 ctx))

def generatedRoot.w_56_0  (ctx : generatedRootCtx) : Int :=
  65536

def generatedRoot.w_57_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_55_0 ctx) * (generatedRoot.w_56_0 ctx))

def generatedRoot.w_58_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_53_0 ctx) + (generatedRoot.w_57_0 ctx))

def generatedRoot.w_59_0  (ctx : generatedRootCtx) : Int :=
  ctx.witness.w_59_0

def generatedRoot.w_60_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_61_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_59_0 ctx) * (generatedRoot.w_60_0 ctx))

def generatedRoot.w_62_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_63_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_61_0 ctx) * (generatedRoot.w_62_0 ctx))

def generatedRoot.w_64_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_58_0 ctx) = (generatedRoot.w_63_0 ctx))

def generatedRoot.w_65_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_64_0 ctx) then 1 else 0

def generatedRoot.w_66_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_49_0 ctx) + (generatedRoot.w_65_0 ctx))

def generatedRoot.w_67_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_68_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_66_0 ctx) = (generatedRoot.w_67_0 ctx))

def generatedRoot.w_69_0  (ctx : generatedRootCtx) : Int :=
  if (generatedRoot.w_68_0 ctx) then 1 else 0

def generatedRoot.w_70_0  (ctx : generatedRootCtx) : Int :=
  ((generatedRoot.w_43_0 ctx) + (generatedRoot.w_69_0 ctx))

def generatedRoot.w_71_0  (ctx : generatedRootCtx) : Int :=
  2

def generatedRoot.w_72_0  (ctx : generatedRootCtx) : Bool :=
  decide ((generatedRoot.w_70_0 ctx) = (generatedRoot.w_71_0 ctx))

def generatedRoot.w_74_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_74_0

def generatedRoot.w_76_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_76_0

def generatedRoot.w_78_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_78_0

def generatedRoot.w_80_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_80_0

def generatedRoot.w_82_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_82_0

def generatedRoot.w_84_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_84_0

def generatedRoot.w_86_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_86_0

def generatedRoot.w_88_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_88_0

def generatedRoot.w_90_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_90_0

def generatedRoot.w_92_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_92_0

def generatedRoot.w_94_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_94_0

def generatedRoot.w_96_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_96_0

def generatedRoot.w_98_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_98_0

def generatedRoot.w_100_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_100_0

def generatedRoot.w_102_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_102_0

def generatedRoot.w_104_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_104_0

def generatedRoot.w_106_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_106_0

def generatedRoot.w_108_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_108_0

def generatedRoot.w_110_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_110_0

def generatedRoot.w_112_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_112_0

def generatedRoot.w_114_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_114_0

def generatedRoot.w_116_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_116_0

def generatedRoot.w_118_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_118_0

def generatedRoot.w_120_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_120_0

def generatedRoot.w_122_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_122_0

def generatedRoot.w_124_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_124_0

def generatedRoot.w_126_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_126_0

def generatedRoot.w_128_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_128_0

def generatedRoot.w_130_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_130_0

def generatedRoot.w_132_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_132_0

def generatedRoot.w_134_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_134_0

def generatedRoot.w_136_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_136_0

def generatedRoot.w_138_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_138_0

def generatedRoot.w_140_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_140_0

def generatedRoot.w_142_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_142_0

def generatedRoot.w_144_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_144_0

def generatedRoot.w_146_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_146_0

def generatedRoot.w_148_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_148_0

def generatedRoot.w_150_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_150_0

def generatedRoot.w_152_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_152_0

def generatedRoot.w_154_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_154_0

def generatedRoot.w_156_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_156_0

def generatedRoot.w_158_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_158_0

def generatedRoot.w_160_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_160_0

def generatedRoot.w_162_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_162_0

def generatedRoot.w_164_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_164_0

def generatedRoot.w_166_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_166_0

def generatedRoot.w_168_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_168_0

def generatedRoot.w_170_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_170_0

def generatedRoot.w_172_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_172_0

def generatedRoot.w_174_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_174_0

def generatedRoot.w_176_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_176_0

def generatedRoot.w_178_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_178_0

def generatedRoot.w_180_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_180_0

def generatedRoot.w_182_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_182_0

def generatedRoot.w_184_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_184_0

def generatedRoot.w_186_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_186_0

def generatedRoot.w_188_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_188_0

def generatedRoot.w_190_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_190_0

def generatedRoot.w_192_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_192_0

def generatedRoot.w_194_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_194_0

def generatedRoot.w_196_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_196_0

def generatedRoot.w_198_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_198_0

def generatedRoot.w_200_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_200_0

def generatedRoot.w_202_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_202_0

def generatedRoot.w_204_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_204_0

def generatedRoot.w_206_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_206_0

def generatedRoot.w_208_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_208_0

def generatedRoot.w_210_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_210_0

def generatedRoot.w_212_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_212_0

def generatedRoot.w_214_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_214_0

def generatedRoot.w_216_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_216_0

def generatedRoot.w_218_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_218_0

def generatedRoot.w_220_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_220_0

def generatedRoot.w_222_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_222_0

def generatedRoot.w_224_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_224_0

def generatedRoot.w_226_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_226_0

def generatedRoot.w_228_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_228_0

def generatedRoot.w_230_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_230_0

def generatedRoot.w_232_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_232_0

def generatedRoot.w_234_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_234_0

def generatedRoot.w_236_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_236_0

def generatedRoot.w_238_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_238_0

def generatedRoot.w_240_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_240_0

def generatedRoot.w_242_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_242_0

def generatedRoot.w_244_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_244_0

def generatedRoot.w_246_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_246_0

def generatedRoot.w_248_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_248_0

def generatedRoot.w_250_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_250_0

def generatedRoot.w_252_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_252_0

def generatedRoot.w_254_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_254_0

def generatedRoot.w_256_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_256_0

def generatedRoot.w_258_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_258_0

def generatedRoot.w_260_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_260_0

def generatedRoot.w_262_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_262_0

def generatedRoot.w_264_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_264_0

def generatedRoot.w_266_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_266_0

def generatedRoot.w_268_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_268_0

def generatedRoot.w_270_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_270_0

def generatedRoot.w_272_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_272_0

def generatedRoot.w_274_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_274_0

def generatedRoot.w_276_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_276_0

def generatedRoot.w_278_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_278_0

def generatedRoot.w_280_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_280_0

def generatedRoot.w_282_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_282_0

def generatedRoot.w_284_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_284_0

def generatedRoot.w_286_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_286_0

def generatedRoot.w_288_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_288_0

def generatedRoot.w_290_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_290_0

def generatedRoot.w_292_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_292_0

def generatedRoot.w_294_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_294_0

def generatedRoot.w_296_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_296_0

def generatedRoot.w_298_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_298_0

def generatedRoot.w_300_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_300_0

def generatedRoot.w_302_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_302_0

def generatedRoot.w_304_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_304_0

def generatedRoot.w_306_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_306_0

def generatedRoot.w_308_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_308_0

def generatedRoot.w_310_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_310_0

def generatedRoot.w_312_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_312_0

def generatedRoot.w_314_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_314_0

def generatedRoot.w_316_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_316_0

def generatedRoot.w_318_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_318_0

def generatedRoot.w_320_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_320_0

def generatedRoot.w_322_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_322_0

def generatedRoot.w_324_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_324_0

def generatedRoot.w_326_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_326_0

def generatedRoot.w_328_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_328_0

def generatedRoot.w_330_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_330_0

def generatedRoot.w_332_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_332_0

def generatedRoot.w_334_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_334_0

def generatedRoot.w_336_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_336_0

def generatedRoot.w_338_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_338_0

def generatedRoot.w_340_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_340_0

def generatedRoot.w_342_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_342_0

def generatedRoot.w_344_0  (ctx : generatedRootCtx) : Bool :=
  ctx.witness.w_344_0

def generatedRoot.outs  (ctx : generatedRootCtx) : Bool :=
  (generatedRoot.w_344_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) : Prop where
  t0 : 0 ≤ (8389632)
  h3_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_3 params i (current, ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) next) (Int.toNat (8389632)) (generatedRoot.w_0_0 ctx) (generatedRoot.w_3_0 ctx)
  t2 : 0 ≤ (1025)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_2_0 ctx), ())))) next) (Int.toNat (1025)) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (generatedRoot.w_8_0 ctx) (0) (generatedRoot.w_9_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (generatedRoot.w_14_0 ctx) (0) (generatedRoot.w_15_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (0) (generatedRoot.w_24_0 ctx)
  h40_1 : MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 8192 8193 13 (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx)
  t8 : (8389632) = 8389632
  h41_0 : (∀ i : Fin 8389632, parallel_generatedRoot_41 params i ((generatedRoot.w_40_1 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_40_0 ctx), ()))) ((generatedRoot.w_41_0 ctx) i))
  t10 : 0 ≤ (1025)
  h42_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 params i (current, ((generatedRoot.w_38_0 ctx), ((generatedRoot.w_39_0 ctx), ((generatedRoot.w_41_0 ctx), ())))) next) (Int.toNat (1025)) (generatedRoot.w_37_0 ctx) (generatedRoot.w_42_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (generatedRoot.w_44_0 ctx) (0) (generatedRoot.w_45_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (generatedRoot.w_50_0 ctx) (0) (generatedRoot.w_51_0 ctx)
  h59_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (1) (generatedRoot.w_59_0 ctx)
  t15 : 0 ≤ (2453063738106839040)
  h74_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 params i (current, ((generatedRoot.w_73_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx)
  t17 : 0 ≤ (2453063738106839040)
  h76_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 params i (current, ((generatedRoot.w_75_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx)
  t19 : 0 ≤ (2453063738106839040)
  h78_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, ((generatedRoot.w_77_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx)
  t21 : 0 ≤ (2453063738106839040)
  h80_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 params i (current, ((generatedRoot.w_79_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx)
  t23 : 0 ≤ (2453063738106839040)
  h82_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 params i (current, ((generatedRoot.w_81_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx)
  t25 : 0 ≤ (2453063738106839040)
  h84_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 params i (current, ((generatedRoot.w_83_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx)
  t27 : 0 ≤ (2453063738106839040)
  h86_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 params i (current, ((generatedRoot.w_85_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx)
  t29 : 0 ≤ (2453063738106839040)
  h88_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 params i (current, ((generatedRoot.w_87_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx)
  t31 : 0 ≤ (2453063738106839040)

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1  (ctx : generatedRootCtx) : Prop where
  h90_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_90 params i (current, ((generatedRoot.w_89_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx)
  t33 : 0 ≤ (2453063738106839040)
  h92_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_92 params i (current, ((generatedRoot.w_91_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx)
  t35 : 0 ≤ (2453063738106839040)
  h94_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_94 params i (current, ((generatedRoot.w_93_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx)
  t37 : 0 ≤ (2453063738106839040)
  h96_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_96 params i (current, ((generatedRoot.w_95_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx)
  t39 : 0 ≤ (2453063738106839040)
  h98_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_98 params i (current, ((generatedRoot.w_97_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx)
  t41 : 0 ≤ (2453063738106839040)
  h100_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_100 params i (current, ((generatedRoot.w_99_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx)
  t43 : 0 ≤ (2453063738106839040)
  h102_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_102 params i (current, ((generatedRoot.w_101_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx)
  t45 : 0 ≤ (2453063738106839040)
  h104_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_104 params i (current, ((generatedRoot.w_103_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx)
  t47 : 0 ≤ (2453063738106839040)
  h106_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_106 params i (current, ((generatedRoot.w_105_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx)
  t49 : 0 ≤ (2453063738106839040)
  h108_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_108 params i (current, ((generatedRoot.w_107_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx)
  t51 : 0 ≤ (2453063738106839040)
  h110_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_110 params i (current, ((generatedRoot.w_109_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx)
  t53 : 0 ≤ (2453063738106839040)
  h112_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_112 params i (current, ((generatedRoot.w_111_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx)
  t55 : 0 ≤ (2453063738106839040)
  h114_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_114 params i (current, ((generatedRoot.w_113_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx)
  t57 : 0 ≤ (2453063738106839040)
  h116_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_116 params i (current, ((generatedRoot.w_115_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx)
  t59 : 0 ≤ (2453063738106839040)
  h118_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_118 params i (current, ((generatedRoot.w_117_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx)
  t61 : 0 ≤ (2453063738106839040)
  h120_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_120 params i (current, ((generatedRoot.w_119_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx)
  t63 : 0 ≤ (2453063738106839040)

theorem generatedRoot.facts_1  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_1 params (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx)) :
    generatedRoot.Facts_1 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2  (ctx : generatedRootCtx) : Prop where
  h122_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_122 params i (current, ((generatedRoot.w_121_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx)
  t65 : 0 ≤ (2453063738106839040)
  h124_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_124 params i (current, ((generatedRoot.w_123_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx)
  t67 : 0 ≤ (2453063738106839040)
  h126_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_126 params i (current, ((generatedRoot.w_125_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx)
  t69 : 0 ≤ (2453063738106839040)
  h128_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_128 params i (current, ((generatedRoot.w_127_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx)
  t71 : 0 ≤ (2453063738106839040)
  h130_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_130 params i (current, ((generatedRoot.w_129_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx)
  t73 : 0 ≤ (2453063738106839040)
  h132_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_132 params i (current, ((generatedRoot.w_131_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx)
  t75 : 0 ≤ (2453063738106839040)
  h134_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_134 params i (current, ((generatedRoot.w_133_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx)
  t77 : 0 ≤ (2453063738106839040)
  h136_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_136 params i (current, ((generatedRoot.w_135_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx)
  t79 : 0 ≤ (2453063738106839040)
  h138_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_138 params i (current, ((generatedRoot.w_137_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx)
  t81 : 0 ≤ (2453063738106839040)
  h140_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_140 params i (current, ((generatedRoot.w_139_0 ctx), ())) next) (Int.toNat (2453063738106839040)) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx)
  t83 : 0 ≤ (65536)
  h142_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_142 params i (current, ((generatedRoot.w_141_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx)
  t85 : 0 ≤ (65536)
  h144_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_144 params i (current, ((generatedRoot.w_143_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx)
  t87 : 0 ≤ (65536)
  h146_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_146 params i (current, ((generatedRoot.w_145_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx)
  t89 : 0 ≤ (65536)
  h148_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_148 params i (current, ((generatedRoot.w_147_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx)
  t91 : 0 ≤ (65536)
  h150_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_150 params i (current, ((generatedRoot.w_149_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx)
  t93 : 0 ≤ (65536)
  h152_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_152 params i (current, ((generatedRoot.w_151_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx)
  t95 : 0 ≤ (65536)

theorem generatedRoot.facts_2  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_2 params (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx)) :
    generatedRoot.Facts_2 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3  (ctx : generatedRootCtx) : Prop where
  h154_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_154 params i (current, ((generatedRoot.w_153_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx)
  t97 : 0 ≤ (65536)
  h156_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_156 params i (current, ((generatedRoot.w_155_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx)
  t99 : 0 ≤ (65536)
  h158_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_158 params i (current, ((generatedRoot.w_157_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx)
  t101 : 0 ≤ (65536)
  h160_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_160 params i (current, ((generatedRoot.w_159_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx)
  t103 : 0 ≤ (65536)
  h162_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_162 params i (current, ((generatedRoot.w_161_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx)
  t105 : 0 ≤ (65536)
  h164_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_164 params i (current, ((generatedRoot.w_163_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx)
  t107 : 0 ≤ (65536)
  h166_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_166 params i (current, ((generatedRoot.w_165_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx)
  t109 : 0 ≤ (65536)
  h168_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_168 params i (current, ((generatedRoot.w_167_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx)
  t111 : 0 ≤ (65536)
  h170_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_170 params i (current, ((generatedRoot.w_169_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx)
  t113 : 0 ≤ (65536)
  h172_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_172 params i (current, ((generatedRoot.w_171_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx)
  t115 : 0 ≤ (65536)
  h174_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_174 params i (current, ((generatedRoot.w_173_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx)
  t117 : 0 ≤ (65536)
  h176_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_176 params i (current, ((generatedRoot.w_175_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx)
  t119 : 0 ≤ (65536)
  h178_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_178 params i (current, ((generatedRoot.w_177_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx)
  t121 : 0 ≤ (65536)
  h180_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_180 params i (current, ((generatedRoot.w_179_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx)
  t123 : 0 ≤ (65536)
  h182_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_182 params i (current, ((generatedRoot.w_181_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx)
  t125 : 0 ≤ (65536)
  h184_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_184 params i (current, ((generatedRoot.w_183_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx)
  t127 : 0 ≤ (65536)

theorem generatedRoot.facts_3  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_3 params (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx)) :
    generatedRoot.Facts_3 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4  (ctx : generatedRootCtx) : Prop where
  h186_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_186 params i (current, ((generatedRoot.w_185_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx)
  t129 : 0 ≤ (65536)
  h188_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_188 params i (current, ((generatedRoot.w_187_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx)
  t131 : 0 ≤ (65536)
  h190_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_190 params i (current, ((generatedRoot.w_189_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx)
  t133 : 0 ≤ (65536)
  h192_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_192 params i (current, ((generatedRoot.w_191_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx)
  t135 : 0 ≤ (65536)
  h194_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_194 params i (current, ((generatedRoot.w_193_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx)
  t137 : 0 ≤ (65536)
  h196_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_196 params i (current, ((generatedRoot.w_195_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx)
  t139 : 0 ≤ (65536)
  h198_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_198 params i (current, ((generatedRoot.w_197_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx)
  t141 : 0 ≤ (65536)
  h200_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_200 params i (current, ((generatedRoot.w_199_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx)
  t143 : 0 ≤ (65536)
  h202_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_202 params i (current, ((generatedRoot.w_201_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx)
  t145 : 0 ≤ (65536)
  h204_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_204 params i (current, ((generatedRoot.w_203_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx)
  t147 : 0 ≤ (65536)
  h206_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_206 params i (current, ((generatedRoot.w_205_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx)
  t149 : 0 ≤ (65536)
  h208_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_208 params i (current, ((generatedRoot.w_207_0 ctx), ())) next) (Int.toNat (65536)) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx)
  t151 : 0 ≤ (296909191249920)
  h210_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_210 params i (current, ((generatedRoot.w_209_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx)
  t153 : 0 ≤ (296909191249920)
  h212_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_212 params i (current, ((generatedRoot.w_211_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_210_0 ctx) (generatedRoot.w_212_0 ctx)
  t155 : 0 ≤ (296909191249920)
  h214_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_214 params i (current, ((generatedRoot.w_213_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_212_0 ctx) (generatedRoot.w_214_0 ctx)
  t157 : 0 ≤ (296909191249920)
  h216_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_216 params i (current, ((generatedRoot.w_215_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_214_0 ctx) (generatedRoot.w_216_0 ctx)
  t159 : 0 ≤ (296909191249920)

theorem generatedRoot.facts_4  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_4 params (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_216_0 ctx)) :
    generatedRoot.Facts_4 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5  (ctx : generatedRootCtx) : Prop where
  h218_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_218 params i (current, ((generatedRoot.w_217_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_216_0 ctx) (generatedRoot.w_218_0 ctx)
  t161 : 0 ≤ (296909191249920)
  h220_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_220 params i (current, ((generatedRoot.w_219_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_218_0 ctx) (generatedRoot.w_220_0 ctx)
  t163 : 0 ≤ (296909191249920)
  h222_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_222 params i (current, ((generatedRoot.w_221_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx)
  t165 : 0 ≤ (296909191249920)
  h224_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_224 params i (current, ((generatedRoot.w_223_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx)
  t167 : 0 ≤ (296909191249920)
  h226_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_226 params i (current, ((generatedRoot.w_225_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_224_0 ctx) (generatedRoot.w_226_0 ctx)
  t169 : 0 ≤ (296909191249920)
  h228_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_228 params i (current, ((generatedRoot.w_227_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_226_0 ctx) (generatedRoot.w_228_0 ctx)
  t171 : 0 ≤ (296909191249920)
  h230_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_230 params i (current, ((generatedRoot.w_229_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_228_0 ctx) (generatedRoot.w_230_0 ctx)
  t173 : 0 ≤ (296909191249920)
  h232_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_232 params i (current, ((generatedRoot.w_231_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx)
  t175 : 0 ≤ (296909191249920)
  h234_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_234 params i (current, ((generatedRoot.w_233_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx)
  t177 : 0 ≤ (296909191249920)
  h236_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_236 params i (current, ((generatedRoot.w_235_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx)
  t179 : 0 ≤ (296909191249920)
  h238_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_238 params i (current, ((generatedRoot.w_237_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_236_0 ctx) (generatedRoot.w_238_0 ctx)
  t181 : 0 ≤ (296909191249920)
  h240_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_240 params i (current, ((generatedRoot.w_239_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_238_0 ctx) (generatedRoot.w_240_0 ctx)
  t183 : 0 ≤ (296909191249920)
  h242_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_242 params i (current, ((generatedRoot.w_241_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_240_0 ctx) (generatedRoot.w_242_0 ctx)
  t185 : 0 ≤ (296909191249920)
  h244_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_244 params i (current, ((generatedRoot.w_243_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_242_0 ctx) (generatedRoot.w_244_0 ctx)
  t187 : 0 ≤ (296909191249920)
  h246_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_246 params i (current, ((generatedRoot.w_245_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_244_0 ctx) (generatedRoot.w_246_0 ctx)
  t189 : 0 ≤ (296909191249920)
  h248_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_248 params i (current, ((generatedRoot.w_247_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_246_0 ctx) (generatedRoot.w_248_0 ctx)
  t191 : 0 ≤ (296909191249920)

theorem generatedRoot.facts_5  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_5 params (generatedRoot.w_217_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_248_0 ctx)) :
    generatedRoot.Facts_5 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6  (ctx : generatedRootCtx) : Prop where
  h250_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_250 params i (current, ((generatedRoot.w_249_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_248_0 ctx) (generatedRoot.w_250_0 ctx)
  t193 : 0 ≤ (296909191249920)
  h252_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_252 params i (current, ((generatedRoot.w_251_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_250_0 ctx) (generatedRoot.w_252_0 ctx)
  t195 : 0 ≤ (296909191249920)
  h254_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_254 params i (current, ((generatedRoot.w_253_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_252_0 ctx) (generatedRoot.w_254_0 ctx)
  t197 : 0 ≤ (296909191249920)
  h256_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_256 params i (current, ((generatedRoot.w_255_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_254_0 ctx) (generatedRoot.w_256_0 ctx)
  t199 : 0 ≤ (296909191249920)
  h258_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_258 params i (current, ((generatedRoot.w_257_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_256_0 ctx) (generatedRoot.w_258_0 ctx)
  t201 : 0 ≤ (296909191249920)
  h260_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_260 params i (current, ((generatedRoot.w_259_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_258_0 ctx) (generatedRoot.w_260_0 ctx)
  t203 : 0 ≤ (296909191249920)
  h262_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_262 params i (current, ((generatedRoot.w_261_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_260_0 ctx) (generatedRoot.w_262_0 ctx)
  t205 : 0 ≤ (296909191249920)
  h264_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_264 params i (current, ((generatedRoot.w_263_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_262_0 ctx) (generatedRoot.w_264_0 ctx)
  t207 : 0 ≤ (296909191249920)
  h266_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_266 params i (current, ((generatedRoot.w_265_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_264_0 ctx) (generatedRoot.w_266_0 ctx)
  t209 : 0 ≤ (296909191249920)
  h268_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_268 params i (current, ((generatedRoot.w_267_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_266_0 ctx) (generatedRoot.w_268_0 ctx)
  t211 : 0 ≤ (296909191249920)
  h270_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_270 params i (current, ((generatedRoot.w_269_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_268_0 ctx) (generatedRoot.w_270_0 ctx)
  t213 : 0 ≤ (296909191249920)
  h272_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_272 params i (current, ((generatedRoot.w_271_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_270_0 ctx) (generatedRoot.w_272_0 ctx)
  t215 : 0 ≤ (296909191249920)
  h274_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_274 params i (current, ((generatedRoot.w_273_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_272_0 ctx) (generatedRoot.w_274_0 ctx)
  t217 : 0 ≤ (296909191249920)
  h276_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_276 params i (current, ((generatedRoot.w_275_0 ctx), ())) next) (Int.toNat (296909191249920)) (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx)
  t219 : 0 ≤ (20054016)
  h278_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_278 params i (current, ((generatedRoot.w_277_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_276_0 ctx) (generatedRoot.w_278_0 ctx)
  t221 : 0 ≤ (20054016)
  h280_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_280 params i (current, ((generatedRoot.w_279_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_278_0 ctx) (generatedRoot.w_280_0 ctx)
  t223 : 0 ≤ (20054016)

theorem generatedRoot.facts_6  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_6 params (generatedRoot.w_249_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_280_0 ctx)) :
    generatedRoot.Facts_6 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_7  (ctx : generatedRootCtx) : Prop where
  h282_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_282 params i (current, ((generatedRoot.w_281_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_280_0 ctx) (generatedRoot.w_282_0 ctx)
  t225 : 0 ≤ (20054016)
  h284_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_284 params i (current, ((generatedRoot.w_283_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_282_0 ctx) (generatedRoot.w_284_0 ctx)
  t227 : 0 ≤ (20054016)
  h286_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_286 params i (current, ((generatedRoot.w_285_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_284_0 ctx) (generatedRoot.w_286_0 ctx)
  t229 : 0 ≤ (20054016)
  h288_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_288 params i (current, ((generatedRoot.w_287_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_286_0 ctx) (generatedRoot.w_288_0 ctx)
  t231 : 0 ≤ (20054016)
  h290_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_290 params i (current, ((generatedRoot.w_289_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx)
  t233 : 0 ≤ (20054016)
  h292_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_292 params i (current, ((generatedRoot.w_291_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_290_0 ctx) (generatedRoot.w_292_0 ctx)
  t235 : 0 ≤ (20054016)
  h294_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_294 params i (current, ((generatedRoot.w_293_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_292_0 ctx) (generatedRoot.w_294_0 ctx)
  t237 : 0 ≤ (20054016)
  h296_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_296 params i (current, ((generatedRoot.w_295_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_294_0 ctx) (generatedRoot.w_296_0 ctx)
  t239 : 0 ≤ (20054016)
  h298_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_298 params i (current, ((generatedRoot.w_297_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_296_0 ctx) (generatedRoot.w_298_0 ctx)
  t241 : 0 ≤ (20054016)
  h300_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_300 params i (current, ((generatedRoot.w_299_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx)
  t243 : 0 ≤ (20054016)
  h302_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_302 params i (current, ((generatedRoot.w_301_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_300_0 ctx) (generatedRoot.w_302_0 ctx)
  t245 : 0 ≤ (20054016)
  h304_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_304 params i (current, ((generatedRoot.w_303_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_302_0 ctx) (generatedRoot.w_304_0 ctx)
  t247 : 0 ≤ (20054016)
  h306_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_306 params i (current, ((generatedRoot.w_305_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_304_0 ctx) (generatedRoot.w_306_0 ctx)
  t249 : 0 ≤ (20054016)
  h308_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_308 params i (current, ((generatedRoot.w_307_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_306_0 ctx) (generatedRoot.w_308_0 ctx)
  t251 : 0 ≤ (20054016)
  h310_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_310 params i (current, ((generatedRoot.w_309_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_308_0 ctx) (generatedRoot.w_310_0 ctx)
  t253 : 0 ≤ (20054016)
  h312_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_312 params i (current, ((generatedRoot.w_311_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_310_0 ctx) (generatedRoot.w_312_0 ctx)
  t255 : 0 ≤ (20054016)

theorem generatedRoot.facts_7  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_7 params (generatedRoot.w_281_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_312_0 ctx)) :
    generatedRoot.Facts_7 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_8  (ctx : generatedRootCtx) (outputs : Bool) : Prop where
  h314_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_314 params i (current, ((generatedRoot.w_313_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_312_0 ctx) (generatedRoot.w_314_0 ctx)
  t257 : 0 ≤ (20054016)
  h316_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_316 params i (current, ((generatedRoot.w_315_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_314_0 ctx) (generatedRoot.w_316_0 ctx)
  t259 : 0 ≤ (20054016)
  h318_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_318 params i (current, ((generatedRoot.w_317_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_316_0 ctx) (generatedRoot.w_318_0 ctx)
  t261 : 0 ≤ (20054016)
  h320_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_320 params i (current, ((generatedRoot.w_319_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_318_0 ctx) (generatedRoot.w_320_0 ctx)
  t263 : 0 ≤ (20054016)
  h322_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_322 params i (current, ((generatedRoot.w_321_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_320_0 ctx) (generatedRoot.w_322_0 ctx)
  t265 : 0 ≤ (20054016)
  h324_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_324 params i (current, ((generatedRoot.w_323_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_322_0 ctx) (generatedRoot.w_324_0 ctx)
  t267 : 0 ≤ (20054016)
  h326_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_326 params i (current, ((generatedRoot.w_325_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_324_0 ctx) (generatedRoot.w_326_0 ctx)
  t269 : 0 ≤ (20054016)
  h328_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_328 params i (current, ((generatedRoot.w_327_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_326_0 ctx) (generatedRoot.w_328_0 ctx)
  t271 : 0 ≤ (20054016)
  h330_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_330 params i (current, ((generatedRoot.w_329_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_328_0 ctx) (generatedRoot.w_330_0 ctx)
  t273 : 0 ≤ (20054016)
  h332_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_332 params i (current, ((generatedRoot.w_331_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_330_0 ctx) (generatedRoot.w_332_0 ctx)
  t275 : 0 ≤ (20054016)
  h334_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_334 params i (current, ((generatedRoot.w_333_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_332_0 ctx) (generatedRoot.w_334_0 ctx)
  t277 : 0 ≤ (20054016)
  h336_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_336 params i (current, ((generatedRoot.w_335_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_334_0 ctx) (generatedRoot.w_336_0 ctx)
  t279 : 0 ≤ (20054016)
  h338_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_338 params i (current, ((generatedRoot.w_337_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_336_0 ctx) (generatedRoot.w_338_0 ctx)
  t281 : 0 ≤ (20054016)
  h340_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_340 params i (current, ((generatedRoot.w_339_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_338_0 ctx) (generatedRoot.w_340_0 ctx)
  t283 : 0 ≤ (20054016)
  h342_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_342 params i (current, ((generatedRoot.w_341_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_340_0 ctx) (generatedRoot.w_342_0 ctx)
  t285 : 0 ≤ (20054016)
  h344_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_344 params i (current, ((generatedRoot.w_343_0 ctx), ())) next) (Int.toNat (20054016)) (generatedRoot.w_342_0 ctx) (generatedRoot.w_344_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_8  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_8 params (generatedRoot.w_313_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_333_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_337_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_330_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_338_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_344_0 ctx) outputs) :
    generatedRoot.Facts_8 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : Bool) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx
  c7 : generatedRoot.Facts_7 ctx
  c8 : generatedRoot.Facts_8 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : Bool}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1.1
  have t2 := t1.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) t2.1
  have t3 := t2.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) t3.1
  have t4 := t3.2
  have r4 := generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) t4.1
  have t5 := t4.2
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) t5.1
  have t6 := t5.2
  have r6 := generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) t6.1
  have t7 := t6.2
  have r7 := generatedRoot.facts_7 (ctx := ctx) (outputs := outputs) t7.1
  have t8 := t7.2
  have r8 := generatedRoot.facts_8 (ctx := ctx) (outputs := outputs) t8
  ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8⟩

theorem generatedRoot.h41_0_at  {ctx : generatedRootCtx} {outputs : Bool}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 8389632) :
    ∃ w : parallel_generatedRoot_41.Witness, parallel_generatedRoot_41.Facts (⟨((generatedRoot.w_40_1 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_40_0 ctx), ()))), w⟩ : parallel_generatedRoot_41.Ctx params i) ((generatedRoot.w_41_0 ctx) i) :=
  let ⟨w, h⟩ := F.c0.h41_0 i
  ⟨w, parallel_generatedRoot_41.facts _ h⟩

end

end W_Requirement_0

end
