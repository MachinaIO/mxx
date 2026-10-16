import Requirement_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

-- The wire types repeat the ring modulus `X ^ n + 1`; unfolding it while checking a
-- projection against its field type expands it at the ring dimension.
attribute [local irreducible] Mxx.Primitives.negacyclicModulus

noncomputable section

namespace Requirement_0

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_41.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Unit
  witness : parallel_generatedRoot_41.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "parallel_generatedRoot_41Ctx" => (parallel_generatedRoot_41.Ctx tape path params i_0)

def parallel_generatedRoot_41.w_1_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 5 → Int :=
  ctx.inputs.1

def parallel_generatedRoot_41.w_6_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 5 → Int :=
  ctx.inputs.2.1

def parallel_generatedRoot_41.w_9_0  (ctx : parallel_generatedRoot_41Ctx) : Fin 5 → Int :=
  ctx.inputs.2.2.1

def parallel_generatedRoot_41.w_0_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  0

def parallel_generatedRoot_41.w_2_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  (Int.ofNat i_0)

def parallel_generatedRoot_41.w_3_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  5

def parallel_generatedRoot_41.w_4_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_2_0 ctx) % (parallel_generatedRoot_41.w_3_0 ctx))

def parallel_generatedRoot_41.w_5_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_5_0

def parallel_generatedRoot_41.w_7_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  5

def parallel_generatedRoot_41.w_8_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_2_0 ctx) / (parallel_generatedRoot_41.w_7_0 ctx))

def parallel_generatedRoot_41.w_10_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_10_0

def parallel_generatedRoot_41.w_11_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ctx.witness.w_11_0

def parallel_generatedRoot_41.w_12_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  16

def parallel_generatedRoot_41.w_13_0  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  ((parallel_generatedRoot_41.w_11_0 ctx) % (parallel_generatedRoot_41.w_12_0 ctx))

def parallel_generatedRoot_41.outs  (ctx : parallel_generatedRoot_41Ctx) : Int :=
  (parallel_generatedRoot_41.w_13_0 ctx)

structure parallel_generatedRoot_41.Facts_0  (ctx : parallel_generatedRoot_41Ctx) (outputs : Int) : Prop where
  h3_0 : (parallel_generatedRoot_41.w_3_0 ctx) ≠ 0
  h4_0 : 0 ≤ (parallel_generatedRoot_41.w_4_0 ctx)
  h4_0_1 : (parallel_generatedRoot_41.w_4_0 ctx) < 5
  h5_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_41.w_1_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_5_0 ctx)
  h7_0 : (parallel_generatedRoot_41.w_7_0 ctx) ≠ 0
  h4_0_2 : 0 ≤ (parallel_generatedRoot_41.w_4_0 ctx)
  h4_0_3 : (parallel_generatedRoot_41.w_4_0 ctx) < 5
  h10_0 : MxxRuntime.familyGetDynamic (parallel_generatedRoot_41.w_9_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_10_0 ctx)
  t8 : 0 ≤ (4)
  h11_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_parallel_generatedRoot_41_11 tape (path ++ [11, i]) params i_0 i (current, ((parallel_generatedRoot_41.w_5_0 ctx), ((parallel_generatedRoot_41.w_6_0 ctx), ((parallel_generatedRoot_41.w_8_0 ctx), ((parallel_generatedRoot_41.w_10_0 ctx), ()))))) next) (Int.toNat (4)) (parallel_generatedRoot_41.w_0_0 ctx) (parallel_generatedRoot_41.w_11_0 ctx)
  h12_0 : (parallel_generatedRoot_41.w_12_0 ctx) ≠ 0
  hout : outputs = parallel_generatedRoot_41.outs ctx

theorem parallel_generatedRoot_41.facts_0  {ctx : parallel_generatedRoot_41Ctx} {outputs : Int}
    (h : parallel_generatedRoot_41.constraints_0 tape path params i_0 (parallel_generatedRoot_41.w_1_0 ctx) (parallel_generatedRoot_41.w_6_0 ctx) (parallel_generatedRoot_41.w_9_0 ctx) (parallel_generatedRoot_41.w_0_0 ctx) (parallel_generatedRoot_41.w_3_0 ctx) (parallel_generatedRoot_41.w_4_0 ctx) (parallel_generatedRoot_41.w_5_0 ctx) (parallel_generatedRoot_41.w_7_0 ctx) (parallel_generatedRoot_41.w_8_0 ctx) (parallel_generatedRoot_41.w_10_0 ctx) (parallel_generatedRoot_41.w_11_0 ctx) (parallel_generatedRoot_41.w_12_0 ctx) (parallel_generatedRoot_41.w_13_0 ctx) outputs) :
    parallel_generatedRoot_41.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2⟩

structure parallel_generatedRoot_41.Facts  (ctx : parallel_generatedRoot_41Ctx) (outputs : Int) : Prop where
  c0 : parallel_generatedRoot_41.Facts_0 ctx outputs

theorem parallel_generatedRoot_41.facts  (ctx : parallel_generatedRoot_41Ctx) {outputs : Int}
    (h : parallel_generatedRoot_41.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    parallel_generatedRoot_41.Facts ctx outputs :=
  have t0 := h
  have r0 := parallel_generatedRoot_41.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_74.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 72000 → Int) × Unit
  witness : sequential_generatedRoot_74.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_74Ctx" => (sequential_generatedRoot_74.Ctx tape path params i_0)

def sequential_generatedRoot_74.w_0_0  (ctx : sequential_generatedRoot_74Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_74.w_3_0  (ctx : sequential_generatedRoot_74Ctx) : Fin 72000 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_74.w_3_0 ctx), ((sequential_generatedRoot_74.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_74.outs ctx

theorem sequential_generatedRoot_74.facts_0  {ctx : sequential_generatedRoot_74Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_74.constraints_0 tape path params i_0 (sequential_generatedRoot_74.w_3_0 ctx) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_4_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx) (sequential_generatedRoot_74.w_11_0 ctx) outputs) :
    sequential_generatedRoot_74.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_74.Facts  (ctx : sequential_generatedRoot_74Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_74.Facts_0 ctx outputs

theorem sequential_generatedRoot_74.facts  (ctx : sequential_generatedRoot_74Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_74.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_74.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_74.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_76.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 72000 → Int) × Unit
  witness : sequential_generatedRoot_76.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_76Ctx" => (sequential_generatedRoot_76.Ctx tape path params i_0)

def sequential_generatedRoot_76.w_0_0  (ctx : sequential_generatedRoot_76Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_76.w_3_0  (ctx : sequential_generatedRoot_76Ctx) : Fin 72000 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_76.w_3_0 ctx), ((sequential_generatedRoot_76.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_76.outs ctx

theorem sequential_generatedRoot_76.facts_0  {ctx : sequential_generatedRoot_76Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_76.constraints_0 tape path params i_0 (sequential_generatedRoot_76.w_3_0 ctx) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_4_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx) (sequential_generatedRoot_76.w_11_0 ctx) outputs) :
    sequential_generatedRoot_76.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_76.Facts  (ctx : sequential_generatedRoot_76Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_76.Facts_0 ctx outputs

theorem sequential_generatedRoot_76.facts  (ctx : sequential_generatedRoot_76Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_76.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_76.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_76.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_78.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_78.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_78Ctx" => (sequential_generatedRoot_78.Ctx tape path params i_0)

def sequential_generatedRoot_78.w_0_0  (ctx : sequential_generatedRoot_78Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_78.w_3_0  (ctx : sequential_generatedRoot_78Ctx) : Fin 16 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_78.w_3_0 ctx), ((sequential_generatedRoot_78.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_78.outs ctx

theorem sequential_generatedRoot_78.facts_0  {ctx : sequential_generatedRoot_78Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_78.constraints_0 tape path params i_0 (sequential_generatedRoot_78.w_3_0 ctx) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_4_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx) (sequential_generatedRoot_78.w_11_0 ctx) outputs) :
    sequential_generatedRoot_78.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_78.Facts  (ctx : sequential_generatedRoot_78Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_78.Facts_0 ctx outputs

theorem sequential_generatedRoot_78.facts  (ctx : sequential_generatedRoot_78Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_78.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_78.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_78.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_80.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_80.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_80Ctx" => (sequential_generatedRoot_80.Ctx tape path params i_0)

def sequential_generatedRoot_80.w_0_0  (ctx : sequential_generatedRoot_80Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_80.w_3_0  (ctx : sequential_generatedRoot_80Ctx) : Fin 16 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_80.w_3_0 ctx), ((sequential_generatedRoot_80.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_80.outs ctx

theorem sequential_generatedRoot_80.facts_0  {ctx : sequential_generatedRoot_80Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_80.constraints_0 tape path params i_0 (sequential_generatedRoot_80.w_3_0 ctx) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_4_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx) (sequential_generatedRoot_80.w_11_0 ctx) outputs) :
    sequential_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_80.Facts  (ctx : sequential_generatedRoot_80Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_80.Facts_0 ctx outputs

theorem sequential_generatedRoot_80.facts  (ctx : sequential_generatedRoot_80Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_80.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_80.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_82.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 14400 → Int) × Unit
  witness : sequential_generatedRoot_82.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_82Ctx" => (sequential_generatedRoot_82.Ctx tape path params i_0)

def sequential_generatedRoot_82.w_0_0  (ctx : sequential_generatedRoot_82Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_82.w_3_0  (ctx : sequential_generatedRoot_82Ctx) : Fin 14400 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_82.w_3_0 ctx), ((sequential_generatedRoot_82.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_82.outs ctx

theorem sequential_generatedRoot_82.facts_0  {ctx : sequential_generatedRoot_82Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_82.constraints_0 tape path params i_0 (sequential_generatedRoot_82.w_3_0 ctx) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_4_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx) (sequential_generatedRoot_82.w_11_0 ctx) outputs) :
    sequential_generatedRoot_82.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_82.Facts  (ctx : sequential_generatedRoot_82Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_82.Facts_0 ctx outputs

theorem sequential_generatedRoot_82.facts  (ctx : sequential_generatedRoot_82Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_82.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_82.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_82.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_84.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 14400 → Int) × Unit
  witness : sequential_generatedRoot_84.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_84Ctx" => (sequential_generatedRoot_84.Ctx tape path params i_0)

def sequential_generatedRoot_84.w_0_0  (ctx : sequential_generatedRoot_84Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_84.w_3_0  (ctx : sequential_generatedRoot_84Ctx) : Fin 14400 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_84.w_3_0 ctx), ((sequential_generatedRoot_84.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_84.outs ctx

theorem sequential_generatedRoot_84.facts_0  {ctx : sequential_generatedRoot_84Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_84.constraints_0 tape path params i_0 (sequential_generatedRoot_84.w_3_0 ctx) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_4_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx) (sequential_generatedRoot_84.w_11_0 ctx) outputs) :
    sequential_generatedRoot_84.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_84.Facts  (ctx : sequential_generatedRoot_84Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_84.Facts_0 ctx outputs

theorem sequential_generatedRoot_84.facts  (ctx : sequential_generatedRoot_84Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_84.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_84.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_84.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_86.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_86.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_86Ctx" => (sequential_generatedRoot_86.Ctx tape path params i_0)

def sequential_generatedRoot_86.w_0_0  (ctx : sequential_generatedRoot_86Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_86.w_3_0  (ctx : sequential_generatedRoot_86Ctx) : Fin 16 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_86.w_3_0 ctx), ((sequential_generatedRoot_86.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_86.outs ctx

theorem sequential_generatedRoot_86.facts_0  {ctx : sequential_generatedRoot_86Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_86.constraints_0 tape path params i_0 (sequential_generatedRoot_86.w_3_0 ctx) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_4_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx) (sequential_generatedRoot_86.w_11_0 ctx) outputs) :
    sequential_generatedRoot_86.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_86.Facts  (ctx : sequential_generatedRoot_86Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_86.Facts_0 ctx outputs

theorem sequential_generatedRoot_86.facts  (ctx : sequential_generatedRoot_86Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_86.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_86.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_86.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_88.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_88.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {i_0 : Nat}
local notation "sequential_generatedRoot_88Ctx" => (sequential_generatedRoot_88.Ctx tape path params i_0)

def sequential_generatedRoot_88.w_0_0  (ctx : sequential_generatedRoot_88Ctx) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_88.w_3_0  (ctx : sequential_generatedRoot_88Ctx) : Fin 16 → Int :=
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
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 tape (path ++ [5, i]) params i_0 i (current, ((sequential_generatedRoot_88.w_3_0 ctx), ((sequential_generatedRoot_88.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_88.outs ctx

theorem sequential_generatedRoot_88.facts_0  {ctx : sequential_generatedRoot_88Ctx} {outputs : Bool}
    (h : sequential_generatedRoot_88.constraints_0 tape path params i_0 (sequential_generatedRoot_88.w_3_0 ctx) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_4_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx) (sequential_generatedRoot_88.w_11_0 ctx) outputs) :
    sequential_generatedRoot_88.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_88.Facts  (ctx : sequential_generatedRoot_88Ctx) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_88.Facts_0 ctx outputs

theorem sequential_generatedRoot_88.facts  (ctx : sequential_generatedRoot_88Ctx) {outputs : Bool}
    (h : sequential_generatedRoot_88.body tape path params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_88.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_88.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

end

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit
  witness : generatedRoot.Witness

section
variable {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx tape path params)

def generatedRoot.w_1_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.inputs.1

def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_5_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_23_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_38_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0  (ctx : generatedRootCtx) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_44_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_50_0  (ctx : generatedRootCtx) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_73_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_75_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_77_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_79_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_81_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_83_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_85_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_87_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

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
  8

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

def generatedRoot.w_40_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.witness.w_40_0

def generatedRoot.w_40_1  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.witness.w_40_1

def generatedRoot.w_41_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
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
  8

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

def generatedRoot.outs  (ctx : generatedRootCtx) : Bool :=
  (generatedRoot.w_88_0 ctx)

structure generatedRoot.Facts_0  (ctx : generatedRootCtx) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h3_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_3 tape (path ++ [3, i]) params i (current, ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_2_0 ctx), ()))) next) (Int.toNat (5)) (generatedRoot.w_0_0 ctx) (generatedRoot.w_3_0 ctx)
  t2 : 0 ≤ (2)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ((generatedRoot.w_2_0 ctx), ())))) next) (Int.toNat (2)) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (generatedRoot.w_8_0 ctx) (0) (generatedRoot.w_9_0 ctx)
  h15_0 : MxxRuntime.familyGetStatic (generatedRoot.w_14_0 ctx) (0) (generatedRoot.w_15_0 ctx)
  h24_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (0) (generatedRoot.w_24_0 ctx)
  h40_1 : MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx)
  t8 : (5) = 5
  h41_0 : (∀ i : Fin 5, parallel_generatedRoot_41 tape (path ++ [41, i.val]) params i ((generatedRoot.w_40_1 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_40_0 ctx), ()))) ((generatedRoot.w_41_0 ctx) i))
  t10 : 0 ≤ (2)
  h42_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 tape (path ++ [42, i]) params i (current, ((generatedRoot.w_38_0 ctx), ((generatedRoot.w_39_0 ctx), ((generatedRoot.w_41_0 ctx), ())))) next) (Int.toNat (2)) (generatedRoot.w_37_0 ctx) (generatedRoot.w_42_0 ctx)
  h45_0 : MxxRuntime.familyGetStatic (generatedRoot.w_44_0 ctx) (0) (generatedRoot.w_45_0 ctx)
  h51_0 : MxxRuntime.familyGetStatic (generatedRoot.w_50_0 ctx) (0) (generatedRoot.w_51_0 ctx)
  h59_0 : MxxRuntime.familyGetStatic (generatedRoot.w_23_0 ctx) (1) (generatedRoot.w_59_0 ctx)
  t15 : 0 ≤ (36000)
  h74_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 tape (path ++ [74, i]) params i (current, ((generatedRoot.w_73_0 ctx), ())) next) (Int.toNat (36000)) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx)
  t17 : 0 ≤ (36000)
  h76_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 tape (path ++ [76, i]) params i (current, ((generatedRoot.w_75_0 ctx), ())) next) (Int.toNat (36000)) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx)
  t19 : 0 ≤ (8)
  h78_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 tape (path ++ [78, i]) params i (current, ((generatedRoot.w_77_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx)
  t21 : 0 ≤ (8)
  h80_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 tape (path ++ [80, i]) params i (current, ((generatedRoot.w_79_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx)
  t23 : 0 ≤ (7200)
  h82_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 tape (path ++ [82, i]) params i (current, ((generatedRoot.w_81_0 ctx), ())) next) (Int.toNat (7200)) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx)
  t25 : 0 ≤ (7200)
  h84_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 tape (path ++ [84, i]) params i (current, ((generatedRoot.w_83_0 ctx), ())) next) (Int.toNat (7200)) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx)
  t27 : 0 ≤ (8)
  h86_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 tape (path ++ [86, i]) params i (current, ((generatedRoot.w_85_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx)
  t29 : 0 ≤ (8)
  h88_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 tape (path ++ [88, i]) params i (current, ((generatedRoot.w_87_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0  {ctx : generatedRootCtx} {outputs : Bool}
    (h : generatedRoot.constraints_0 tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_0_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_40_1 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts  (ctx : generatedRootCtx) (outputs : Bool) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts  (ctx : generatedRootCtx) {outputs : Bool}
    (h : generatedRoot.body tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

theorem generatedRoot.h41_0_at  {ctx : generatedRootCtx} {outputs : Bool}
    (F : generatedRoot.Facts ctx outputs) (i : Fin 5) :
    ∃ w : parallel_generatedRoot_41.Witness, parallel_generatedRoot_41.Facts (⟨((generatedRoot.w_40_1 ctx), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_40_0 ctx), ()))), w⟩ : parallel_generatedRoot_41.Ctx tape (path ++ [41, i.val]) params i) ((generatedRoot.w_41_0 ctx) i) :=
  let ⟨w, h⟩ := F.c0.h41_0 i
  ⟨w, parallel_generatedRoot_41.facts _ h⟩

end

end Requirement_0

end
