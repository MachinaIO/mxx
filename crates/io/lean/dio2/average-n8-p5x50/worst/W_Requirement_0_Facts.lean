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
structure sequential_generatedRoot_298.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 72000 → Int) × Unit
  witness : sequential_generatedRoot_298.Witness

def sequential_generatedRoot_298.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_298.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Fin 72000 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_298.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_298.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_298.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_298.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_298.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_298.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_298.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_298.w_5_0 ctx) = (sequential_generatedRoot_298.w_6_0 ctx))

def sequential_generatedRoot_298.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_298.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_298.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_298.w_1_0 ctx) + (sequential_generatedRoot_298.w_8_0 ctx))

def sequential_generatedRoot_298.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_298.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_298.w_9_0 ctx) = (sequential_generatedRoot_298.w_10_0 ctx))

def sequential_generatedRoot_298.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_298.w_11_0 ctx)

structure sequential_generatedRoot_298.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_298_5 params i_0 i (current, ((sequential_generatedRoot_298.w_3_0 ctx), ((sequential_generatedRoot_298.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_298.w_2_0 ctx) (sequential_generatedRoot_298.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_298.outs ctx

theorem sequential_generatedRoot_298.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_298.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_298.constraints_0 params i_0 (sequential_generatedRoot_298.w_3_0 ctx) (sequential_generatedRoot_298.w_2_0 ctx) (sequential_generatedRoot_298.w_4_0 ctx) (sequential_generatedRoot_298.w_5_0 ctx) (sequential_generatedRoot_298.w_11_0 ctx) outputs) :
    sequential_generatedRoot_298.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_298.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_298.Facts_0 ctx outputs

theorem sequential_generatedRoot_298.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_298.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_298.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_298.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_298.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_300.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 72000 → Int) × Unit
  witness : sequential_generatedRoot_300.Witness

def sequential_generatedRoot_300.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_300.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Fin 72000 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_300.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_300.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_300.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_300.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_300.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_300.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_300.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_300.w_5_0 ctx) = (sequential_generatedRoot_300.w_6_0 ctx))

def sequential_generatedRoot_300.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_300.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_300.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_300.w_1_0 ctx) + (sequential_generatedRoot_300.w_8_0 ctx))

def sequential_generatedRoot_300.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_300.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_300.w_9_0 ctx) = (sequential_generatedRoot_300.w_10_0 ctx))

def sequential_generatedRoot_300.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_300.w_11_0 ctx)

structure sequential_generatedRoot_300.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_300_5 params i_0 i (current, ((sequential_generatedRoot_300.w_3_0 ctx), ((sequential_generatedRoot_300.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_300.w_2_0 ctx) (sequential_generatedRoot_300.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_300.outs ctx

theorem sequential_generatedRoot_300.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_300.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_300.constraints_0 params i_0 (sequential_generatedRoot_300.w_3_0 ctx) (sequential_generatedRoot_300.w_2_0 ctx) (sequential_generatedRoot_300.w_4_0 ctx) (sequential_generatedRoot_300.w_5_0 ctx) (sequential_generatedRoot_300.w_11_0 ctx) outputs) :
    sequential_generatedRoot_300.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_300.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_300.Facts_0 ctx outputs

theorem sequential_generatedRoot_300.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_300.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_300.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_300.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_300.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_302.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_302.Witness

def sequential_generatedRoot_302.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_302.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Fin 16 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_302.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_302.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_302.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_302.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_302.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_302.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_302.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_302.w_5_0 ctx) = (sequential_generatedRoot_302.w_6_0 ctx))

def sequential_generatedRoot_302.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_302.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_302.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_302.w_1_0 ctx) + (sequential_generatedRoot_302.w_8_0 ctx))

def sequential_generatedRoot_302.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_302.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_302.w_9_0 ctx) = (sequential_generatedRoot_302.w_10_0 ctx))

def sequential_generatedRoot_302.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_302.w_11_0 ctx)

structure sequential_generatedRoot_302.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_302_5 params i_0 i (current, ((sequential_generatedRoot_302.w_3_0 ctx), ((sequential_generatedRoot_302.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_302.w_2_0 ctx) (sequential_generatedRoot_302.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_302.outs ctx

theorem sequential_generatedRoot_302.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_302.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_302.constraints_0 params i_0 (sequential_generatedRoot_302.w_3_0 ctx) (sequential_generatedRoot_302.w_2_0 ctx) (sequential_generatedRoot_302.w_4_0 ctx) (sequential_generatedRoot_302.w_5_0 ctx) (sequential_generatedRoot_302.w_11_0 ctx) outputs) :
    sequential_generatedRoot_302.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_302.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_302.Facts_0 ctx outputs

theorem sequential_generatedRoot_302.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_302.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_302.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_302.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_302.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_304.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_304.Witness

def sequential_generatedRoot_304.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_304.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Fin 16 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_304.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_304.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_304.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_304.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_304.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_304.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_304.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_304.w_5_0 ctx) = (sequential_generatedRoot_304.w_6_0 ctx))

def sequential_generatedRoot_304.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_304.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_304.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_304.w_1_0 ctx) + (sequential_generatedRoot_304.w_8_0 ctx))

def sequential_generatedRoot_304.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_304.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_304.w_9_0 ctx) = (sequential_generatedRoot_304.w_10_0 ctx))

def sequential_generatedRoot_304.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_304.w_11_0 ctx)

structure sequential_generatedRoot_304.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_304_5 params i_0 i (current, ((sequential_generatedRoot_304.w_3_0 ctx), ((sequential_generatedRoot_304.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_304.w_2_0 ctx) (sequential_generatedRoot_304.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_304.outs ctx

theorem sequential_generatedRoot_304.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_304.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_304.constraints_0 params i_0 (sequential_generatedRoot_304.w_3_0 ctx) (sequential_generatedRoot_304.w_2_0 ctx) (sequential_generatedRoot_304.w_4_0 ctx) (sequential_generatedRoot_304.w_5_0 ctx) (sequential_generatedRoot_304.w_11_0 ctx) outputs) :
    sequential_generatedRoot_304.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_304.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_304.Facts_0 ctx outputs

theorem sequential_generatedRoot_304.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_304.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_304.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_304.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_304.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_306.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 14400 → Int) × Unit
  witness : sequential_generatedRoot_306.Witness

def sequential_generatedRoot_306.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_306.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Fin 14400 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_306.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_306.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_306.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_306.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_306.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_306.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_306.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_306.w_5_0 ctx) = (sequential_generatedRoot_306.w_6_0 ctx))

def sequential_generatedRoot_306.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_306.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_306.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_306.w_1_0 ctx) + (sequential_generatedRoot_306.w_8_0 ctx))

def sequential_generatedRoot_306.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_306.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_306.w_9_0 ctx) = (sequential_generatedRoot_306.w_10_0 ctx))

def sequential_generatedRoot_306.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_306.w_11_0 ctx)

structure sequential_generatedRoot_306.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_306_5 params i_0 i (current, ((sequential_generatedRoot_306.w_3_0 ctx), ((sequential_generatedRoot_306.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_306.w_2_0 ctx) (sequential_generatedRoot_306.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_306.outs ctx

theorem sequential_generatedRoot_306.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_306.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_306.constraints_0 params i_0 (sequential_generatedRoot_306.w_3_0 ctx) (sequential_generatedRoot_306.w_2_0 ctx) (sequential_generatedRoot_306.w_4_0 ctx) (sequential_generatedRoot_306.w_5_0 ctx) (sequential_generatedRoot_306.w_11_0 ctx) outputs) :
    sequential_generatedRoot_306.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_306.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_306.Facts_0 ctx outputs

theorem sequential_generatedRoot_306.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_306.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_306.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_306.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_306.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_308.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 14400 → Int) × Unit
  witness : sequential_generatedRoot_308.Witness

def sequential_generatedRoot_308.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_308.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Fin 14400 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_308.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_308.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_308.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_308.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_308.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_308.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_308.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_308.w_5_0 ctx) = (sequential_generatedRoot_308.w_6_0 ctx))

def sequential_generatedRoot_308.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_308.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_308.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_308.w_1_0 ctx) + (sequential_generatedRoot_308.w_8_0 ctx))

def sequential_generatedRoot_308.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_308.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_308.w_9_0 ctx) = (sequential_generatedRoot_308.w_10_0 ctx))

def sequential_generatedRoot_308.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_308.w_11_0 ctx)

structure sequential_generatedRoot_308.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_308_5 params i_0 i (current, ((sequential_generatedRoot_308.w_3_0 ctx), ((sequential_generatedRoot_308.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_308.w_2_0 ctx) (sequential_generatedRoot_308.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_308.outs ctx

theorem sequential_generatedRoot_308.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_308.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_308.constraints_0 params i_0 (sequential_generatedRoot_308.w_3_0 ctx) (sequential_generatedRoot_308.w_2_0 ctx) (sequential_generatedRoot_308.w_4_0 ctx) (sequential_generatedRoot_308.w_5_0 ctx) (sequential_generatedRoot_308.w_11_0 ctx) outputs) :
    sequential_generatedRoot_308.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_308.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_308.Facts_0 ctx outputs

theorem sequential_generatedRoot_308.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_308.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_308.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_308.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_308.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_310.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_310.Witness

def sequential_generatedRoot_310.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_310.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Fin 16 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_310.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_310.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_310.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_310.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_310.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_310.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_310.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_310.w_5_0 ctx) = (sequential_generatedRoot_310.w_6_0 ctx))

def sequential_generatedRoot_310.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_310.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_310.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_310.w_1_0 ctx) + (sequential_generatedRoot_310.w_8_0 ctx))

def sequential_generatedRoot_310.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_310.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_310.w_9_0 ctx) = (sequential_generatedRoot_310.w_10_0 ctx))

def sequential_generatedRoot_310.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_310.w_11_0 ctx)

structure sequential_generatedRoot_310.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_310_5 params i_0 i (current, ((sequential_generatedRoot_310.w_3_0 ctx), ((sequential_generatedRoot_310.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_310.w_2_0 ctx) (sequential_generatedRoot_310.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_310.outs ctx

theorem sequential_generatedRoot_310.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_310.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_310.constraints_0 params i_0 (sequential_generatedRoot_310.w_3_0 ctx) (sequential_generatedRoot_310.w_2_0 ctx) (sequential_generatedRoot_310.w_4_0 ctx) (sequential_generatedRoot_310.w_5_0 ctx) (sequential_generatedRoot_310.w_11_0 ctx) outputs) :
    sequential_generatedRoot_310.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_310.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_310.Facts_0 ctx outputs

theorem sequential_generatedRoot_310.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_310.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_310.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_310.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_310.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_312.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 16 → Int) × Unit
  witness : sequential_generatedRoot_312.Witness

def sequential_generatedRoot_312.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_312.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Fin 16 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_312.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_312.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_312.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_312.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_312.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_312.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_312.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_312.w_5_0 ctx) = (sequential_generatedRoot_312.w_6_0 ctx))

def sequential_generatedRoot_312.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_312.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_312.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_312.w_1_0 ctx) + (sequential_generatedRoot_312.w_8_0 ctx))

def sequential_generatedRoot_312.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_312.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_312.w_9_0 ctx) = (sequential_generatedRoot_312.w_10_0 ctx))

def sequential_generatedRoot_312.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_312.w_11_0 ctx)

structure sequential_generatedRoot_312.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (2)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_312_5 params i_0 i (current, ((sequential_generatedRoot_312.w_3_0 ctx), ((sequential_generatedRoot_312.w_4_0 ctx), ()))) next) (Int.toNat (2)) (sequential_generatedRoot_312.w_2_0 ctx) (sequential_generatedRoot_312.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_312.outs ctx

theorem sequential_generatedRoot_312.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_312.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_312.constraints_0 params i_0 (sequential_generatedRoot_312.w_3_0 ctx) (sequential_generatedRoot_312.w_2_0 ctx) (sequential_generatedRoot_312.w_4_0 ctx) (sequential_generatedRoot_312.w_5_0 ctx) (sequential_generatedRoot_312.w_11_0 ctx) outputs) :
    sequential_generatedRoot_312.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_312.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_312.Facts_0 ctx outputs

theorem sequential_generatedRoot_312.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_312.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_312.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_312.Facts ctx outputs :=
  have t0 := h
  have r0 := sequential_generatedRoot_312.facts_0 (ctx := ctx) (outputs := outputs) t0
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : (Fin 2 → Int) × (Fin 2 → Int) × (Fin 5 → Int) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.1

def generatedRoot.w_8_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_17_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 5 → Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_61_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_67_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 4 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_175_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_181_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_190_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_206_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_212_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 2 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_268_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_274_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 1 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_297_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_299_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_301_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_303_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_305_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_307_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_309_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_311_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  true

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_0_0 ctx) then 1 else 0

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_3_0

def generatedRoot.w_4_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_3_0 ctx) * (generatedRoot.w_3_0 ctx))

def generatedRoot.w_5_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_6_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_4_0 ctx) = (generatedRoot.w_5_0 ctx))

def generatedRoot.w_7_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_6_0 ctx) then 1 else 0

def generatedRoot.w_9_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_9_0

def generatedRoot.w_10_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_11_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_9_0 ctx) * (generatedRoot.w_10_0 ctx))

def generatedRoot.w_12_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_13_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_12_0 ctx) - (generatedRoot.w_3_0 ctx))

def generatedRoot.w_14_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_15_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_13_0 ctx) * (generatedRoot.w_14_0 ctx))

def generatedRoot.w_16_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_11_0 ctx) + (generatedRoot.w_15_0 ctx))

def generatedRoot.w_18_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_18_0

def generatedRoot.w_19_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_20_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_18_0 ctx) * (generatedRoot.w_19_0 ctx))

def generatedRoot.w_21_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_22_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_20_0 ctx) * (generatedRoot.w_21_0 ctx))

def generatedRoot.w_23_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_16_0 ctx) = (generatedRoot.w_22_0 ctx))

def generatedRoot.w_24_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_23_0 ctx) then 1 else 0

def generatedRoot.w_25_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_7_0 ctx) + (generatedRoot.w_24_0 ctx))

def generatedRoot.w_26_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_27_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_25_0 ctx) = (generatedRoot.w_26_0 ctx))

def generatedRoot.w_28_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_27_0 ctx) then 1 else 0

def generatedRoot.w_29_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_1_0 ctx) + (generatedRoot.w_28_0 ctx))

def generatedRoot.w_30_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_31_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_29_0 ctx) = (generatedRoot.w_30_0 ctx))

def generatedRoot.w_32_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_31_0 ctx) then 1 else 0

def generatedRoot.w_33_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_33_0

def generatedRoot.w_34_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_33_0 ctx) * (generatedRoot.w_33_0 ctx))

def generatedRoot.w_35_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_36_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_34_0 ctx) = (generatedRoot.w_35_0 ctx))

def generatedRoot.w_37_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_36_0 ctx) then 1 else 0

def generatedRoot.w_38_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_38_0

def generatedRoot.w_39_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_40_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_38_0 ctx) * (generatedRoot.w_39_0 ctx))

def generatedRoot.w_41_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_42_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_41_0 ctx) - (generatedRoot.w_33_0 ctx))

def generatedRoot.w_43_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_44_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_42_0 ctx) * (generatedRoot.w_43_0 ctx))

def generatedRoot.w_45_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_40_0 ctx) + (generatedRoot.w_44_0 ctx))

def generatedRoot.w_46_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_47_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_48_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_46_0 ctx) * (generatedRoot.w_47_0 ctx))

def generatedRoot.w_49_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_50_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_48_0 ctx) * (generatedRoot.w_49_0 ctx))

def generatedRoot.w_51_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_45_0 ctx) = (generatedRoot.w_50_0 ctx))

def generatedRoot.w_52_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_51_0 ctx) then 1 else 0

def generatedRoot.w_53_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_37_0 ctx) + (generatedRoot.w_52_0 ctx))

def generatedRoot.w_54_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_55_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_53_0 ctx) = (generatedRoot.w_54_0 ctx))

def generatedRoot.w_56_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_55_0 ctx) then 1 else 0

def generatedRoot.w_57_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_32_0 ctx) + (generatedRoot.w_56_0 ctx))

def generatedRoot.w_58_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_59_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_57_0 ctx) = (generatedRoot.w_58_0 ctx))

def generatedRoot.w_60_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_59_0 ctx) then 1 else 0

def generatedRoot.w_62_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_62_0

def generatedRoot.w_63_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_62_0 ctx) * (generatedRoot.w_62_0 ctx))

def generatedRoot.w_64_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_65_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_63_0 ctx) = (generatedRoot.w_64_0 ctx))

def generatedRoot.w_66_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_65_0 ctx) then 1 else 0

def generatedRoot.w_68_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_68_0

def generatedRoot.w_69_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_70_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_68_0 ctx) * (generatedRoot.w_69_0 ctx))

def generatedRoot.w_71_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_72_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_71_0 ctx) - (generatedRoot.w_62_0 ctx))

def generatedRoot.w_73_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_74_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_72_0 ctx) * (generatedRoot.w_73_0 ctx))

def generatedRoot.w_75_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_70_0 ctx) + (generatedRoot.w_74_0 ctx))

def generatedRoot.w_76_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_76_0

def generatedRoot.w_77_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_78_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_76_0 ctx) * (generatedRoot.w_77_0 ctx))

def generatedRoot.w_79_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_80_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_78_0 ctx) * (generatedRoot.w_79_0 ctx))

def generatedRoot.w_81_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_75_0 ctx) = (generatedRoot.w_80_0 ctx))

def generatedRoot.w_82_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_81_0 ctx) then 1 else 0

def generatedRoot.w_83_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_66_0 ctx) + (generatedRoot.w_82_0 ctx))

def generatedRoot.w_84_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_85_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_83_0 ctx) = (generatedRoot.w_84_0 ctx))

def generatedRoot.w_86_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_85_0 ctx) then 1 else 0

def generatedRoot.w_87_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_60_0 ctx) + (generatedRoot.w_86_0 ctx))

def generatedRoot.w_88_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_89_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_87_0 ctx) = (generatedRoot.w_88_0 ctx))

def generatedRoot.w_90_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_89_0 ctx) then 1 else 0

def generatedRoot.w_91_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_91_0

def generatedRoot.w_92_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_91_0 ctx) * (generatedRoot.w_91_0 ctx))

def generatedRoot.w_93_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_94_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_92_0 ctx) = (generatedRoot.w_93_0 ctx))

def generatedRoot.w_95_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_94_0 ctx) then 1 else 0

def generatedRoot.w_96_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_96_0

def generatedRoot.w_97_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_98_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_96_0 ctx) * (generatedRoot.w_97_0 ctx))

def generatedRoot.w_99_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_100_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_99_0 ctx) - (generatedRoot.w_91_0 ctx))

def generatedRoot.w_101_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_102_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_100_0 ctx) * (generatedRoot.w_101_0 ctx))

def generatedRoot.w_103_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_98_0 ctx) + (generatedRoot.w_102_0 ctx))

def generatedRoot.w_104_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_106_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_104_0 ctx) * (generatedRoot.w_105_0 ctx))

def generatedRoot.w_107_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_108_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_106_0 ctx) * (generatedRoot.w_107_0 ctx))

def generatedRoot.w_109_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_103_0 ctx) = (generatedRoot.w_108_0 ctx))

def generatedRoot.w_110_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_109_0 ctx) then 1 else 0

def generatedRoot.w_111_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_95_0 ctx) + (generatedRoot.w_110_0 ctx))

def generatedRoot.w_112_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_113_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_111_0 ctx) = (generatedRoot.w_112_0 ctx))

def generatedRoot.w_114_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_113_0 ctx) then 1 else 0

def generatedRoot.w_115_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_90_0 ctx) + (generatedRoot.w_114_0 ctx))

def generatedRoot.w_116_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_117_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_115_0 ctx) = (generatedRoot.w_116_0 ctx))

def generatedRoot.w_118_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_117_0 ctx) then 1 else 0

def generatedRoot.w_119_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_119_0

def generatedRoot.w_120_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_119_0 ctx) * (generatedRoot.w_119_0 ctx))

def generatedRoot.w_121_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_122_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_120_0 ctx) = (generatedRoot.w_121_0 ctx))

def generatedRoot.w_123_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_122_0 ctx) then 1 else 0

def generatedRoot.w_124_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_124_0

def generatedRoot.w_125_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_126_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_124_0 ctx) * (generatedRoot.w_125_0 ctx))

def generatedRoot.w_127_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_128_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_127_0 ctx) - (generatedRoot.w_119_0 ctx))

def generatedRoot.w_129_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_130_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_128_0 ctx) * (generatedRoot.w_129_0 ctx))

def generatedRoot.w_131_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_126_0 ctx) + (generatedRoot.w_130_0 ctx))

def generatedRoot.w_132_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_132_0

def generatedRoot.w_133_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_134_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_132_0 ctx) * (generatedRoot.w_133_0 ctx))

def generatedRoot.w_135_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_136_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_134_0 ctx) * (generatedRoot.w_135_0 ctx))

def generatedRoot.w_137_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_131_0 ctx) = (generatedRoot.w_136_0 ctx))

def generatedRoot.w_138_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_137_0 ctx) then 1 else 0

def generatedRoot.w_139_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_123_0 ctx) + (generatedRoot.w_138_0 ctx))

def generatedRoot.w_140_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_141_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_139_0 ctx) = (generatedRoot.w_140_0 ctx))

def generatedRoot.w_142_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_141_0 ctx) then 1 else 0

def generatedRoot.w_143_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_118_0 ctx) + (generatedRoot.w_142_0 ctx))

def generatedRoot.w_144_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_145_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_143_0 ctx) = (generatedRoot.w_144_0 ctx))

def generatedRoot.w_146_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_145_0 ctx) then 1 else 0

def generatedRoot.w_147_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_147_0

def generatedRoot.w_148_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_147_0 ctx) * (generatedRoot.w_147_0 ctx))

def generatedRoot.w_149_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_150_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_148_0 ctx) = (generatedRoot.w_149_0 ctx))

def generatedRoot.w_151_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_150_0 ctx) then 1 else 0

def generatedRoot.w_152_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_152_0

def generatedRoot.w_153_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_154_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_152_0 ctx) * (generatedRoot.w_153_0 ctx))

def generatedRoot.w_155_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_156_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_155_0 ctx) - (generatedRoot.w_147_0 ctx))

def generatedRoot.w_157_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_158_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_156_0 ctx) * (generatedRoot.w_157_0 ctx))

def generatedRoot.w_159_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_154_0 ctx) + (generatedRoot.w_158_0 ctx))

def generatedRoot.w_160_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_160_0

def generatedRoot.w_161_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_162_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_160_0 ctx) * (generatedRoot.w_161_0 ctx))

def generatedRoot.w_163_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_164_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_162_0 ctx) * (generatedRoot.w_163_0 ctx))

def generatedRoot.w_165_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_159_0 ctx) = (generatedRoot.w_164_0 ctx))

def generatedRoot.w_166_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_165_0 ctx) then 1 else 0

def generatedRoot.w_167_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_151_0 ctx) + (generatedRoot.w_166_0 ctx))

def generatedRoot.w_168_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_169_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_167_0 ctx) = (generatedRoot.w_168_0 ctx))

def generatedRoot.w_170_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_169_0 ctx) then 1 else 0

def generatedRoot.w_171_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_146_0 ctx) + (generatedRoot.w_170_0 ctx))

def generatedRoot.w_172_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_173_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_171_0 ctx) = (generatedRoot.w_172_0 ctx))

def generatedRoot.w_174_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_173_0 ctx) then 1 else 0

def generatedRoot.w_176_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_176_0

def generatedRoot.w_177_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_176_0 ctx) * (generatedRoot.w_176_0 ctx))

def generatedRoot.w_178_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_179_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_177_0 ctx) = (generatedRoot.w_178_0 ctx))

def generatedRoot.w_180_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_179_0 ctx) then 1 else 0

def generatedRoot.w_182_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_182_0

def generatedRoot.w_183_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_184_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_182_0 ctx) * (generatedRoot.w_183_0 ctx))

def generatedRoot.w_185_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_186_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_185_0 ctx) - (generatedRoot.w_176_0 ctx))

def generatedRoot.w_187_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_188_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_186_0 ctx) * (generatedRoot.w_187_0 ctx))

def generatedRoot.w_189_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_184_0 ctx) + (generatedRoot.w_188_0 ctx))

def generatedRoot.w_191_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_191_0

def generatedRoot.w_192_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_193_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_191_0 ctx) * (generatedRoot.w_192_0 ctx))

def generatedRoot.w_194_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_195_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_193_0 ctx) * (generatedRoot.w_194_0 ctx))

def generatedRoot.w_196_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_189_0 ctx) = (generatedRoot.w_195_0 ctx))

def generatedRoot.w_197_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_196_0 ctx) then 1 else 0

def generatedRoot.w_198_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_180_0 ctx) + (generatedRoot.w_197_0 ctx))

def generatedRoot.w_199_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_200_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_198_0 ctx) = (generatedRoot.w_199_0 ctx))

def generatedRoot.w_201_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_200_0 ctx) then 1 else 0

def generatedRoot.w_202_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_174_0 ctx) + (generatedRoot.w_201_0 ctx))

def generatedRoot.w_203_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_204_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_202_0 ctx) = (generatedRoot.w_203_0 ctx))

def generatedRoot.w_205_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_204_0 ctx) then 1 else 0

def generatedRoot.w_207_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_207_0

def generatedRoot.w_208_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_207_0 ctx) * (generatedRoot.w_207_0 ctx))

def generatedRoot.w_209_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_210_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_208_0 ctx) = (generatedRoot.w_209_0 ctx))

def generatedRoot.w_211_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_210_0 ctx) then 1 else 0

def generatedRoot.w_213_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_213_0

def generatedRoot.w_214_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_215_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_213_0 ctx) * (generatedRoot.w_214_0 ctx))

def generatedRoot.w_216_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_217_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_216_0 ctx) - (generatedRoot.w_207_0 ctx))

def generatedRoot.w_218_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_219_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_217_0 ctx) * (generatedRoot.w_218_0 ctx))

def generatedRoot.w_220_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_215_0 ctx) + (generatedRoot.w_219_0 ctx))

def generatedRoot.w_221_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_222_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_221_0 ctx) + (generatedRoot.w_104_0 ctx))

def generatedRoot.w_223_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_222_0 ctx) + (generatedRoot.w_160_0 ctx))

def generatedRoot.w_224_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  16

def generatedRoot.w_225_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_223_0 ctx) % (generatedRoot.w_224_0 ctx))

def generatedRoot.w_226_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_227_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_225_0 ctx) * (generatedRoot.w_226_0 ctx))

def generatedRoot.w_228_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_229_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_227_0 ctx) * (generatedRoot.w_228_0 ctx))

def generatedRoot.w_230_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_220_0 ctx) = (generatedRoot.w_229_0 ctx))

def generatedRoot.w_231_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_230_0 ctx) then 1 else 0

def generatedRoot.w_232_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_211_0 ctx) + (generatedRoot.w_231_0 ctx))

def generatedRoot.w_233_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_234_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_232_0 ctx) = (generatedRoot.w_233_0 ctx))

def generatedRoot.w_235_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_234_0 ctx) then 1 else 0

def generatedRoot.w_236_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_205_0 ctx) + (generatedRoot.w_235_0 ctx))

def generatedRoot.w_237_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_238_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_236_0 ctx) = (generatedRoot.w_237_0 ctx))

def generatedRoot.w_239_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_238_0 ctx) then 1 else 0

def generatedRoot.w_240_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_240_0

def generatedRoot.w_241_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_240_0 ctx) * (generatedRoot.w_240_0 ctx))

def generatedRoot.w_242_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_243_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_241_0 ctx) = (generatedRoot.w_242_0 ctx))

def generatedRoot.w_244_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_243_0 ctx) then 1 else 0

def generatedRoot.w_245_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_245_0

def generatedRoot.w_246_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_247_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_245_0 ctx) * (generatedRoot.w_246_0 ctx))

def generatedRoot.w_248_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_249_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_248_0 ctx) - (generatedRoot.w_240_0 ctx))

def generatedRoot.w_250_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_251_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_249_0 ctx) * (generatedRoot.w_250_0 ctx))

def generatedRoot.w_252_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_247_0 ctx) + (generatedRoot.w_251_0 ctx))

def generatedRoot.w_253_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_254_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_255_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_253_0 ctx) * (generatedRoot.w_254_0 ctx))

def generatedRoot.w_256_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_257_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_255_0 ctx) * (generatedRoot.w_256_0 ctx))

def generatedRoot.w_258_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_252_0 ctx) = (generatedRoot.w_257_0 ctx))

def generatedRoot.w_259_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_258_0 ctx) then 1 else 0

def generatedRoot.w_260_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_244_0 ctx) + (generatedRoot.w_259_0 ctx))

def generatedRoot.w_261_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_262_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_260_0 ctx) = (generatedRoot.w_261_0 ctx))

def generatedRoot.w_263_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_262_0 ctx) then 1 else 0

def generatedRoot.w_264_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_239_0 ctx) + (generatedRoot.w_263_0 ctx))

def generatedRoot.w_265_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_266_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_264_0 ctx) = (generatedRoot.w_265_0 ctx))

def generatedRoot.w_267_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_266_0 ctx) then 1 else 0

def generatedRoot.w_269_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_269_0

def generatedRoot.w_270_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_269_0 ctx) * (generatedRoot.w_269_0 ctx))

def generatedRoot.w_271_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_272_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_270_0 ctx) = (generatedRoot.w_271_0 ctx))

def generatedRoot.w_273_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_272_0 ctx) then 1 else 0

def generatedRoot.w_275_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_275_0

def generatedRoot.w_276_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_277_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_275_0 ctx) * (generatedRoot.w_276_0 ctx))

def generatedRoot.w_278_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_279_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_278_0 ctx) - (generatedRoot.w_269_0 ctx))

def generatedRoot.w_280_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  8

def generatedRoot.w_281_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_279_0 ctx) * (generatedRoot.w_280_0 ctx))

def generatedRoot.w_282_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_277_0 ctx) + (generatedRoot.w_281_0 ctx))

def generatedRoot.w_283_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.witness.w_283_0

def generatedRoot.w_284_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_285_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_283_0 ctx) * (generatedRoot.w_284_0 ctx))

def generatedRoot.w_286_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_287_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_285_0 ctx) * (generatedRoot.w_286_0 ctx))

def generatedRoot.w_288_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_282_0 ctx) = (generatedRoot.w_287_0 ctx))

def generatedRoot.w_289_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_288_0 ctx) then 1 else 0

def generatedRoot.w_290_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_273_0 ctx) + (generatedRoot.w_289_0 ctx))

def generatedRoot.w_291_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_292_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_290_0 ctx) = (generatedRoot.w_291_0 ctx))

def generatedRoot.w_293_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_292_0 ctx) then 1 else 0

def generatedRoot.w_294_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_267_0 ctx) + (generatedRoot.w_293_0 ctx))

def generatedRoot.w_295_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_296_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_294_0 ctx) = (generatedRoot.w_295_0 ctx))

def generatedRoot.w_298_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_298_0

def generatedRoot.w_300_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_300_0

def generatedRoot.w_302_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_302_0

def generatedRoot.w_304_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_304_0

def generatedRoot.w_306_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_306_0

def generatedRoot.w_308_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_308_0

def generatedRoot.w_310_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_310_0

def generatedRoot.w_312_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_312_0

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  (generatedRoot.w_312_0 ctx)

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  h3_0 : MxxRuntime.familyGetStatic (generatedRoot.w_2_0 ctx) (0) (generatedRoot.w_3_0 ctx)
  h9_0 : MxxRuntime.familyGetStatic (generatedRoot.w_8_0 ctx) (0) (generatedRoot.w_9_0 ctx)
  h18_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (0) (generatedRoot.w_18_0 ctx)
  h33_0 : MxxRuntime.familyGetStatic (generatedRoot.w_2_0 ctx) (1) (generatedRoot.w_33_0 ctx)
  h38_0 : MxxRuntime.familyGetStatic (generatedRoot.w_8_0 ctx) (1) (generatedRoot.w_38_0 ctx)
  h62_0 : MxxRuntime.familyGetStatic (generatedRoot.w_61_0 ctx) (0) (generatedRoot.w_62_0 ctx)
  h68_0 : MxxRuntime.familyGetStatic (generatedRoot.w_67_0 ctx) (0) (generatedRoot.w_68_0 ctx)
  h76_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (1) (generatedRoot.w_76_0 ctx)
  h91_0 : MxxRuntime.familyGetStatic (generatedRoot.w_61_0 ctx) (1) (generatedRoot.w_91_0 ctx)
  h96_0 : MxxRuntime.familyGetStatic (generatedRoot.w_67_0 ctx) (1) (generatedRoot.w_96_0 ctx)
  h104_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (2) (generatedRoot.w_104_0 ctx)
  h119_0 : MxxRuntime.familyGetStatic (generatedRoot.w_61_0 ctx) (2) (generatedRoot.w_119_0 ctx)
  h124_0 : MxxRuntime.familyGetStatic (generatedRoot.w_67_0 ctx) (2) (generatedRoot.w_124_0 ctx)
  h132_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (3) (generatedRoot.w_132_0 ctx)
  h147_0 : MxxRuntime.familyGetStatic (generatedRoot.w_61_0 ctx) (3) (generatedRoot.w_147_0 ctx)
  h152_0 : MxxRuntime.familyGetStatic (generatedRoot.w_67_0 ctx) (3) (generatedRoot.w_152_0 ctx)
  h160_0 : MxxRuntime.familyGetStatic (generatedRoot.w_17_0 ctx) (4) (generatedRoot.w_160_0 ctx)
  h176_0 : MxxRuntime.familyGetStatic (generatedRoot.w_175_0 ctx) (0) (generatedRoot.w_176_0 ctx)
  h182_0 : MxxRuntime.familyGetStatic (generatedRoot.w_181_0 ctx) (0) (generatedRoot.w_182_0 ctx)
  h191_0 : MxxRuntime.familyGetStatic (generatedRoot.w_190_0 ctx) (0) (generatedRoot.w_191_0 ctx)
  h207_0 : MxxRuntime.familyGetStatic (generatedRoot.w_206_0 ctx) (0) (generatedRoot.w_207_0 ctx)
  h213_0 : MxxRuntime.familyGetStatic (generatedRoot.w_212_0 ctx) (0) (generatedRoot.w_213_0 ctx)
  h224_0 : (generatedRoot.w_224_0 ctx) ≠ 0
  h240_0 : MxxRuntime.familyGetStatic (generatedRoot.w_206_0 ctx) (1) (generatedRoot.w_240_0 ctx)
  h245_0 : MxxRuntime.familyGetStatic (generatedRoot.w_212_0 ctx) (1) (generatedRoot.w_245_0 ctx)
  h269_0 : MxxRuntime.familyGetStatic (generatedRoot.w_268_0 ctx) (0) (generatedRoot.w_269_0 ctx)
  h275_0 : MxxRuntime.familyGetStatic (generatedRoot.w_274_0 ctx) (0) (generatedRoot.w_275_0 ctx)
  h283_0 : MxxRuntime.familyGetStatic (generatedRoot.w_190_0 ctx) (1) (generatedRoot.w_283_0 ctx)
  t28 : 0 ≤ (36000)
  h298_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_298 params i (current, ((generatedRoot.w_297_0 ctx), ())) next) (Int.toNat (36000)) (generatedRoot.w_296_0 ctx) (generatedRoot.w_298_0 ctx)
  t30 : 0 ≤ (36000)
  h300_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_300 params i (current, ((generatedRoot.w_299_0 ctx), ())) next) (Int.toNat (36000)) (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx)

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_2_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_213_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_300_0 ctx)) :
    generatedRoot.Facts_0 ctx :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Bool) : Prop where
  t32 : 0 ≤ (8)
  h302_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_302 params i (current, ((generatedRoot.w_301_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_300_0 ctx) (generatedRoot.w_302_0 ctx)
  t34 : 0 ≤ (8)
  h304_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_304 params i (current, ((generatedRoot.w_303_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_302_0 ctx) (generatedRoot.w_304_0 ctx)
  t36 : 0 ≤ (7200)
  h306_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_306 params i (current, ((generatedRoot.w_305_0 ctx), ())) next) (Int.toNat (7200)) (generatedRoot.w_304_0 ctx) (generatedRoot.w_306_0 ctx)
  t38 : 0 ≤ (7200)
  h308_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_308 params i (current, ((generatedRoot.w_307_0 ctx), ())) next) (Int.toNat (7200)) (generatedRoot.w_306_0 ctx) (generatedRoot.w_308_0 ctx)
  t40 : 0 ≤ (8)
  h310_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_310 params i (current, ((generatedRoot.w_309_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_308_0 ctx) (generatedRoot.w_310_0 ctx)
  t42 : 0 ≤ (8)
  h312_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_312 params i (current, ((generatedRoot.w_311_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_310_0 ctx) (generatedRoot.w_312_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_1 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_1 params (generatedRoot.w_301_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_312_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Bool) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : Bool}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have t0 := h
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) t0.1
  have t1 := t0.2
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) t1
  ⟨r0, r1⟩

end W_Requirement_0

end
