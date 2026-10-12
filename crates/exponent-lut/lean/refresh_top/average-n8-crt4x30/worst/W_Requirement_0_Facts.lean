import W_Requirement_0

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace W_Requirement_0

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_8.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 240 → Int) × Unit
  witness : sequential_generatedRoot_8.Witness

def sequential_generatedRoot_8.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_8.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Fin 240 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_8.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_8.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_8.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_8.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_8.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_8.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_8.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_8.w_5_0 ctx) = (sequential_generatedRoot_8.w_6_0 ctx))

def sequential_generatedRoot_8.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_8.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_8.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_8.w_1_0 ctx) + (sequential_generatedRoot_8.w_8_0 ctx))

def sequential_generatedRoot_8.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_8.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_8.w_9_0 ctx) = (sequential_generatedRoot_8.w_10_0 ctx))

def sequential_generatedRoot_8.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_8.w_11_0 ctx)

structure sequential_generatedRoot_8.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_8_5 params i_0 i (current, ((sequential_generatedRoot_8.w_3_0 ctx), ((sequential_generatedRoot_8.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_8.w_2_0 ctx) (sequential_generatedRoot_8.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_8.outs ctx

theorem sequential_generatedRoot_8.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_8.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_8.constraints_0 params i_0 (sequential_generatedRoot_8.w_3_0 ctx) (sequential_generatedRoot_8.w_2_0 ctx) (sequential_generatedRoot_8.w_4_0 ctx) (sequential_generatedRoot_8.w_5_0 ctx) (sequential_generatedRoot_8.w_11_0 ctx) outputs) :
    sequential_generatedRoot_8.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_8.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_8.Facts_0 ctx outputs

theorem sequential_generatedRoot_8.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_8.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_8.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_8.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_8.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_10.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 240 → Int) × Unit
  witness : sequential_generatedRoot_10.Witness

def sequential_generatedRoot_10.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_10.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Fin 240 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_10.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_10.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_10.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_10.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_10.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_10.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_10.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_10.w_5_0 ctx) = (sequential_generatedRoot_10.w_6_0 ctx))

def sequential_generatedRoot_10.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_10.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_10.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_10.w_1_0 ctx) + (sequential_generatedRoot_10.w_8_0 ctx))

def sequential_generatedRoot_10.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_10.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_10.w_9_0 ctx) = (sequential_generatedRoot_10.w_10_0 ctx))

def sequential_generatedRoot_10.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_10.w_11_0 ctx)

structure sequential_generatedRoot_10.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_10_5 params i_0 i (current, ((sequential_generatedRoot_10.w_3_0 ctx), ((sequential_generatedRoot_10.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_10.w_2_0 ctx) (sequential_generatedRoot_10.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_10.outs ctx

theorem sequential_generatedRoot_10.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_10.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_10.constraints_0 params i_0 (sequential_generatedRoot_10.w_3_0 ctx) (sequential_generatedRoot_10.w_2_0 ctx) (sequential_generatedRoot_10.w_4_0 ctx) (sequential_generatedRoot_10.w_5_0 ctx) (sequential_generatedRoot_10.w_11_0 ctx) outputs) :
    sequential_generatedRoot_10.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_10.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_10.Facts_0 ctx outputs

theorem sequential_generatedRoot_10.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_10.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_10.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_10.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_10.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_12.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 240 → Int) × Unit
  witness : sequential_generatedRoot_12.Witness

def sequential_generatedRoot_12.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_12.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Fin 240 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_12.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_12.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_12.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_12.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_12.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_12.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_12.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_12.w_5_0 ctx) = (sequential_generatedRoot_12.w_6_0 ctx))

def sequential_generatedRoot_12.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_12.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_12.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_12.w_1_0 ctx) + (sequential_generatedRoot_12.w_8_0 ctx))

def sequential_generatedRoot_12.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_12.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_12.w_9_0 ctx) = (sequential_generatedRoot_12.w_10_0 ctx))

def sequential_generatedRoot_12.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_12.w_11_0 ctx)

structure sequential_generatedRoot_12.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_12_5 params i_0 i (current, ((sequential_generatedRoot_12.w_3_0 ctx), ((sequential_generatedRoot_12.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_12.w_2_0 ctx) (sequential_generatedRoot_12.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_12.outs ctx

theorem sequential_generatedRoot_12.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_12.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_12.constraints_0 params i_0 (sequential_generatedRoot_12.w_3_0 ctx) (sequential_generatedRoot_12.w_2_0 ctx) (sequential_generatedRoot_12.w_4_0 ctx) (sequential_generatedRoot_12.w_5_0 ctx) (sequential_generatedRoot_12.w_11_0 ctx) outputs) :
    sequential_generatedRoot_12.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_12.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_12.Facts_0 ctx outputs

theorem sequential_generatedRoot_12.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_12.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_12.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_12.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_12.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_14.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 240 → Int) × Unit
  witness : sequential_generatedRoot_14.Witness

def sequential_generatedRoot_14.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_14.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Fin 240 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_14.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_14.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_14.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_14.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_14.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_14.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_14.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_14.w_5_0 ctx) = (sequential_generatedRoot_14.w_6_0 ctx))

def sequential_generatedRoot_14.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_14.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_14.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_14.w_1_0 ctx) + (sequential_generatedRoot_14.w_8_0 ctx))

def sequential_generatedRoot_14.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_14.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_14.w_9_0 ctx) = (sequential_generatedRoot_14.w_10_0 ctx))

def sequential_generatedRoot_14.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_14.w_11_0 ctx)

structure sequential_generatedRoot_14.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_14_5 params i_0 i (current, ((sequential_generatedRoot_14.w_3_0 ctx), ((sequential_generatedRoot_14.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_14.w_2_0 ctx) (sequential_generatedRoot_14.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_14.outs ctx

theorem sequential_generatedRoot_14.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_14.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_14.constraints_0 params i_0 (sequential_generatedRoot_14.w_3_0 ctx) (sequential_generatedRoot_14.w_2_0 ctx) (sequential_generatedRoot_14.w_4_0 ctx) (sequential_generatedRoot_14.w_5_0 ctx) (sequential_generatedRoot_14.w_11_0 ctx) outputs) :
    sequential_generatedRoot_14.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_14.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_14.Facts_0 ctx outputs

theorem sequential_generatedRoot_14.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_14.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_14.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_14.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_14.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_16.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 40 → Int) × Unit
  witness : sequential_generatedRoot_16.Witness

def sequential_generatedRoot_16.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_16.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Fin 40 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_16.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_16.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_16.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_16.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_16.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_16.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_16.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_16.w_5_0 ctx) = (sequential_generatedRoot_16.w_6_0 ctx))

def sequential_generatedRoot_16.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_16.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_16.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_16.w_1_0 ctx) + (sequential_generatedRoot_16.w_8_0 ctx))

def sequential_generatedRoot_16.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_16.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_16.w_9_0 ctx) = (sequential_generatedRoot_16.w_10_0 ctx))

def sequential_generatedRoot_16.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_16.w_11_0 ctx)

structure sequential_generatedRoot_16.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_16_5 params i_0 i (current, ((sequential_generatedRoot_16.w_3_0 ctx), ((sequential_generatedRoot_16.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_16.w_2_0 ctx) (sequential_generatedRoot_16.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_16.outs ctx

theorem sequential_generatedRoot_16.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_16.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_16.constraints_0 params i_0 (sequential_generatedRoot_16.w_3_0 ctx) (sequential_generatedRoot_16.w_2_0 ctx) (sequential_generatedRoot_16.w_4_0 ctx) (sequential_generatedRoot_16.w_5_0 ctx) (sequential_generatedRoot_16.w_11_0 ctx) outputs) :
    sequential_generatedRoot_16.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_16.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_16.Facts_0 ctx outputs

theorem sequential_generatedRoot_16.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_16.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_16.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_16.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_16.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_18.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 40 → Int) × Unit
  witness : sequential_generatedRoot_18.Witness

def sequential_generatedRoot_18.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_18.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Fin 40 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_18.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_18.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_18.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_18.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_18.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_18.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_18.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_18.w_5_0 ctx) = (sequential_generatedRoot_18.w_6_0 ctx))

def sequential_generatedRoot_18.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_18.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_18.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_18.w_1_0 ctx) + (sequential_generatedRoot_18.w_8_0 ctx))

def sequential_generatedRoot_18.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_18.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_18.w_9_0 ctx) = (sequential_generatedRoot_18.w_10_0 ctx))

def sequential_generatedRoot_18.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_18.w_11_0 ctx)

structure sequential_generatedRoot_18.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_18_5 params i_0 i (current, ((sequential_generatedRoot_18.w_3_0 ctx), ((sequential_generatedRoot_18.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_18.w_2_0 ctx) (sequential_generatedRoot_18.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_18.outs ctx

theorem sequential_generatedRoot_18.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_18.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_18.constraints_0 params i_0 (sequential_generatedRoot_18.w_3_0 ctx) (sequential_generatedRoot_18.w_2_0 ctx) (sequential_generatedRoot_18.w_4_0 ctx) (sequential_generatedRoot_18.w_5_0 ctx) (sequential_generatedRoot_18.w_11_0 ctx) outputs) :
    sequential_generatedRoot_18.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_18.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_18.Facts_0 ctx outputs

theorem sequential_generatedRoot_18.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_18.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_18.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_18.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_18.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_20.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_20.Witness

def sequential_generatedRoot_20.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_20.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_20.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_20.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_20.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_20.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_20.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_20.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_20.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_20.w_5_0 ctx) = (sequential_generatedRoot_20.w_6_0 ctx))

def sequential_generatedRoot_20.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_20.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_20.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_20.w_1_0 ctx) + (sequential_generatedRoot_20.w_8_0 ctx))

def sequential_generatedRoot_20.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_20.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_20.w_9_0 ctx) = (sequential_generatedRoot_20.w_10_0 ctx))

def sequential_generatedRoot_20.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_20.w_11_0 ctx)

structure sequential_generatedRoot_20.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_20_5 params i_0 i (current, ((sequential_generatedRoot_20.w_3_0 ctx), ((sequential_generatedRoot_20.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_20.w_2_0 ctx) (sequential_generatedRoot_20.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_20.outs ctx

theorem sequential_generatedRoot_20.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_20.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_20.constraints_0 params i_0 (sequential_generatedRoot_20.w_3_0 ctx) (sequential_generatedRoot_20.w_2_0 ctx) (sequential_generatedRoot_20.w_4_0 ctx) (sequential_generatedRoot_20.w_5_0 ctx) (sequential_generatedRoot_20.w_11_0 ctx) outputs) :
    sequential_generatedRoot_20.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_20.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_20.Facts_0 ctx outputs

theorem sequential_generatedRoot_20.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_20.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_20.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_20.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_20.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_22.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_22.Witness

def sequential_generatedRoot_22.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_22.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_22.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_22.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_22.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_22.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_22.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_22.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_22.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_22.w_5_0 ctx) = (sequential_generatedRoot_22.w_6_0 ctx))

def sequential_generatedRoot_22.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_22.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_22.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_22.w_1_0 ctx) + (sequential_generatedRoot_22.w_8_0 ctx))

def sequential_generatedRoot_22.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_22.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_22.w_9_0 ctx) = (sequential_generatedRoot_22.w_10_0 ctx))

def sequential_generatedRoot_22.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_22.w_11_0 ctx)

structure sequential_generatedRoot_22.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_22_5 params i_0 i (current, ((sequential_generatedRoot_22.w_3_0 ctx), ((sequential_generatedRoot_22.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_22.w_2_0 ctx) (sequential_generatedRoot_22.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_22.outs ctx

theorem sequential_generatedRoot_22.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_22.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_22.constraints_0 params i_0 (sequential_generatedRoot_22.w_3_0 ctx) (sequential_generatedRoot_22.w_2_0 ctx) (sequential_generatedRoot_22.w_4_0 ctx) (sequential_generatedRoot_22.w_5_0 ctx) (sequential_generatedRoot_22.w_11_0 ctx) outputs) :
    sequential_generatedRoot_22.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_22.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_22.Facts_0 ctx outputs

theorem sequential_generatedRoot_22.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_22.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_22.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_22.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_22.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_24.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_24.Witness

def sequential_generatedRoot_24.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_24.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_24.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_24.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_24.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_24.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_24.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_24.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_24.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_24.w_5_0 ctx) = (sequential_generatedRoot_24.w_6_0 ctx))

def sequential_generatedRoot_24.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_24.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_24.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_24.w_1_0 ctx) + (sequential_generatedRoot_24.w_8_0 ctx))

def sequential_generatedRoot_24.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_24.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_24.w_9_0 ctx) = (sequential_generatedRoot_24.w_10_0 ctx))

def sequential_generatedRoot_24.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_24.w_11_0 ctx)

structure sequential_generatedRoot_24.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_24_5 params i_0 i (current, ((sequential_generatedRoot_24.w_3_0 ctx), ((sequential_generatedRoot_24.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_24.w_2_0 ctx) (sequential_generatedRoot_24.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_24.outs ctx

theorem sequential_generatedRoot_24.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_24.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_24.constraints_0 params i_0 (sequential_generatedRoot_24.w_3_0 ctx) (sequential_generatedRoot_24.w_2_0 ctx) (sequential_generatedRoot_24.w_4_0 ctx) (sequential_generatedRoot_24.w_5_0 ctx) (sequential_generatedRoot_24.w_11_0 ctx) outputs) :
    sequential_generatedRoot_24.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_24.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_24.Facts_0 ctx outputs

theorem sequential_generatedRoot_24.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_24.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_24.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_24.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_24.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_26.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_26.Witness

def sequential_generatedRoot_26.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_26.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_26.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_26.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_26.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_26.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_26.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_26.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_26.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_26.w_5_0 ctx) = (sequential_generatedRoot_26.w_6_0 ctx))

def sequential_generatedRoot_26.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_26.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_26.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_26.w_1_0 ctx) + (sequential_generatedRoot_26.w_8_0 ctx))

def sequential_generatedRoot_26.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_26.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_26.w_9_0 ctx) = (sequential_generatedRoot_26.w_10_0 ctx))

def sequential_generatedRoot_26.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_26.w_11_0 ctx)

structure sequential_generatedRoot_26.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_26_5 params i_0 i (current, ((sequential_generatedRoot_26.w_3_0 ctx), ((sequential_generatedRoot_26.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_26.w_2_0 ctx) (sequential_generatedRoot_26.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_26.outs ctx

theorem sequential_generatedRoot_26.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_26.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_26.constraints_0 params i_0 (sequential_generatedRoot_26.w_3_0 ctx) (sequential_generatedRoot_26.w_2_0 ctx) (sequential_generatedRoot_26.w_4_0 ctx) (sequential_generatedRoot_26.w_5_0 ctx) (sequential_generatedRoot_26.w_11_0 ctx) outputs) :
    sequential_generatedRoot_26.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_26.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_26.Facts_0 ctx outputs

theorem sequential_generatedRoot_26.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_26.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_26.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_26.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_26.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_28.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_28.Witness

def sequential_generatedRoot_28.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_28.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_28.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_28.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_28.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_28.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_28.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_28.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_28.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_28.w_5_0 ctx) = (sequential_generatedRoot_28.w_6_0 ctx))

def sequential_generatedRoot_28.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_28.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_28.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_28.w_1_0 ctx) + (sequential_generatedRoot_28.w_8_0 ctx))

def sequential_generatedRoot_28.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_28.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_28.w_9_0 ctx) = (sequential_generatedRoot_28.w_10_0 ctx))

def sequential_generatedRoot_28.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_28.w_11_0 ctx)

structure sequential_generatedRoot_28.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_28_5 params i_0 i (current, ((sequential_generatedRoot_28.w_3_0 ctx), ((sequential_generatedRoot_28.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_28.w_2_0 ctx) (sequential_generatedRoot_28.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_28.outs ctx

theorem sequential_generatedRoot_28.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_28.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_28.constraints_0 params i_0 (sequential_generatedRoot_28.w_3_0 ctx) (sequential_generatedRoot_28.w_2_0 ctx) (sequential_generatedRoot_28.w_4_0 ctx) (sequential_generatedRoot_28.w_5_0 ctx) (sequential_generatedRoot_28.w_11_0 ctx) outputs) :
    sequential_generatedRoot_28.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_28.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_28.Facts_0 ctx outputs

theorem sequential_generatedRoot_28.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_28.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_28.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_28.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_28.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_30.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_30.Witness

def sequential_generatedRoot_30.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_30.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_30.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_30.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_30.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_30.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_30.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_30.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_30.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_30.w_5_0 ctx) = (sequential_generatedRoot_30.w_6_0 ctx))

def sequential_generatedRoot_30.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_30.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_30.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_30.w_1_0 ctx) + (sequential_generatedRoot_30.w_8_0 ctx))

def sequential_generatedRoot_30.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_30.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_30.w_9_0 ctx) = (sequential_generatedRoot_30.w_10_0 ctx))

def sequential_generatedRoot_30.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_30.w_11_0 ctx)

structure sequential_generatedRoot_30.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_30_5 params i_0 i (current, ((sequential_generatedRoot_30.w_3_0 ctx), ((sequential_generatedRoot_30.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_30.w_2_0 ctx) (sequential_generatedRoot_30.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_30.outs ctx

theorem sequential_generatedRoot_30.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_30.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_30.constraints_0 params i_0 (sequential_generatedRoot_30.w_3_0 ctx) (sequential_generatedRoot_30.w_2_0 ctx) (sequential_generatedRoot_30.w_4_0 ctx) (sequential_generatedRoot_30.w_5_0 ctx) (sequential_generatedRoot_30.w_11_0 ctx) outputs) :
    sequential_generatedRoot_30.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_30.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_30.Facts_0 ctx outputs

theorem sequential_generatedRoot_30.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_30.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_30.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_30.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_30.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_32.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_32.Witness

def sequential_generatedRoot_32.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_32.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_32.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_32.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_32.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_32.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_32.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_32.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_32.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_32.w_5_0 ctx) = (sequential_generatedRoot_32.w_6_0 ctx))

def sequential_generatedRoot_32.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_32.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_32.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_32.w_1_0 ctx) + (sequential_generatedRoot_32.w_8_0 ctx))

def sequential_generatedRoot_32.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_32.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_32.w_9_0 ctx) = (sequential_generatedRoot_32.w_10_0 ctx))

def sequential_generatedRoot_32.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_32.w_11_0 ctx)

structure sequential_generatedRoot_32.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_32_5 params i_0 i (current, ((sequential_generatedRoot_32.w_3_0 ctx), ((sequential_generatedRoot_32.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_32.w_2_0 ctx) (sequential_generatedRoot_32.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_32.outs ctx

theorem sequential_generatedRoot_32.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_32.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_32.constraints_0 params i_0 (sequential_generatedRoot_32.w_3_0 ctx) (sequential_generatedRoot_32.w_2_0 ctx) (sequential_generatedRoot_32.w_4_0 ctx) (sequential_generatedRoot_32.w_5_0 ctx) (sequential_generatedRoot_32.w_11_0 ctx) outputs) :
    sequential_generatedRoot_32.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_32.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_32.Facts_0 ctx outputs

theorem sequential_generatedRoot_32.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_32.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_32.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_32.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_32.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_34.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_34.Witness

def sequential_generatedRoot_34.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_34.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_34.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_34.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_34.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_34.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_34.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_34.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_34.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_34.w_5_0 ctx) = (sequential_generatedRoot_34.w_6_0 ctx))

def sequential_generatedRoot_34.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_34.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_34.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_34.w_1_0 ctx) + (sequential_generatedRoot_34.w_8_0 ctx))

def sequential_generatedRoot_34.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_34.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_34.w_9_0 ctx) = (sequential_generatedRoot_34.w_10_0 ctx))

def sequential_generatedRoot_34.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_34.w_11_0 ctx)

structure sequential_generatedRoot_34.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_34_5 params i_0 i (current, ((sequential_generatedRoot_34.w_3_0 ctx), ((sequential_generatedRoot_34.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_34.w_2_0 ctx) (sequential_generatedRoot_34.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_34.outs ctx

theorem sequential_generatedRoot_34.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_34.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_34.constraints_0 params i_0 (sequential_generatedRoot_34.w_3_0 ctx) (sequential_generatedRoot_34.w_2_0 ctx) (sequential_generatedRoot_34.w_4_0 ctx) (sequential_generatedRoot_34.w_5_0 ctx) (sequential_generatedRoot_34.w_11_0 ctx) outputs) :
    sequential_generatedRoot_34.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_34.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_34.Facts_0 ctx outputs

theorem sequential_generatedRoot_34.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_34.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_34.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_34.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_34.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_36.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_36.Witness

def sequential_generatedRoot_36.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_36.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_36.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_36.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_36.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_36.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_36.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_36.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_36.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_36.w_5_0 ctx) = (sequential_generatedRoot_36.w_6_0 ctx))

def sequential_generatedRoot_36.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_36.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_36.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_36.w_1_0 ctx) + (sequential_generatedRoot_36.w_8_0 ctx))

def sequential_generatedRoot_36.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_36.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_36.w_9_0 ctx) = (sequential_generatedRoot_36.w_10_0 ctx))

def sequential_generatedRoot_36.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_36.w_11_0 ctx)

structure sequential_generatedRoot_36.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_36_5 params i_0 i (current, ((sequential_generatedRoot_36.w_3_0 ctx), ((sequential_generatedRoot_36.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_36.w_2_0 ctx) (sequential_generatedRoot_36.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_36.outs ctx

theorem sequential_generatedRoot_36.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_36.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_36.constraints_0 params i_0 (sequential_generatedRoot_36.w_3_0 ctx) (sequential_generatedRoot_36.w_2_0 ctx) (sequential_generatedRoot_36.w_4_0 ctx) (sequential_generatedRoot_36.w_5_0 ctx) (sequential_generatedRoot_36.w_11_0 ctx) outputs) :
    sequential_generatedRoot_36.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_36.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_36.Facts_0 ctx outputs

theorem sequential_generatedRoot_36.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_36.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_36.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_36.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_36.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_38.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_38.Witness

def sequential_generatedRoot_38.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_38.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_38.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_38.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_38.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_38.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_38.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_38.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_38.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_38.w_5_0 ctx) = (sequential_generatedRoot_38.w_6_0 ctx))

def sequential_generatedRoot_38.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_38.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_38.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_38.w_1_0 ctx) + (sequential_generatedRoot_38.w_8_0 ctx))

def sequential_generatedRoot_38.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_38.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_38.w_9_0 ctx) = (sequential_generatedRoot_38.w_10_0 ctx))

def sequential_generatedRoot_38.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_38.w_11_0 ctx)

structure sequential_generatedRoot_38.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_38_5 params i_0 i (current, ((sequential_generatedRoot_38.w_3_0 ctx), ((sequential_generatedRoot_38.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_38.w_2_0 ctx) (sequential_generatedRoot_38.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_38.outs ctx

theorem sequential_generatedRoot_38.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_38.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_38.constraints_0 params i_0 (sequential_generatedRoot_38.w_3_0 ctx) (sequential_generatedRoot_38.w_2_0 ctx) (sequential_generatedRoot_38.w_4_0 ctx) (sequential_generatedRoot_38.w_5_0 ctx) (sequential_generatedRoot_38.w_11_0 ctx) outputs) :
    sequential_generatedRoot_38.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_38.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_38.Facts_0 ctx outputs

theorem sequential_generatedRoot_38.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_38.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_38.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_38.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_38.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_40.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_40.Witness

def sequential_generatedRoot_40.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_40.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_40.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_40.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_40.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_40.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_40.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_40.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_40.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_40.w_5_0 ctx) = (sequential_generatedRoot_40.w_6_0 ctx))

def sequential_generatedRoot_40.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_40.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_40.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_40.w_1_0 ctx) + (sequential_generatedRoot_40.w_8_0 ctx))

def sequential_generatedRoot_40.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_40.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_40.w_9_0 ctx) = (sequential_generatedRoot_40.w_10_0 ctx))

def sequential_generatedRoot_40.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_40.w_11_0 ctx)

structure sequential_generatedRoot_40.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_40_5 params i_0 i (current, ((sequential_generatedRoot_40.w_3_0 ctx), ((sequential_generatedRoot_40.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_40.w_2_0 ctx) (sequential_generatedRoot_40.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_40.outs ctx

theorem sequential_generatedRoot_40.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_40.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_40.constraints_0 params i_0 (sequential_generatedRoot_40.w_3_0 ctx) (sequential_generatedRoot_40.w_2_0 ctx) (sequential_generatedRoot_40.w_4_0 ctx) (sequential_generatedRoot_40.w_5_0 ctx) (sequential_generatedRoot_40.w_11_0 ctx) outputs) :
    sequential_generatedRoot_40.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_40.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_40.Facts_0 ctx outputs

theorem sequential_generatedRoot_40.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_40.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_40.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_40.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_40.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_42.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_42.Witness

def sequential_generatedRoot_42.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_42.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_42.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_42.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_42.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_42.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_42.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_42.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_42.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_42.w_5_0 ctx) = (sequential_generatedRoot_42.w_6_0 ctx))

def sequential_generatedRoot_42.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_42.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_42.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_42.w_1_0 ctx) + (sequential_generatedRoot_42.w_8_0 ctx))

def sequential_generatedRoot_42.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_42.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_42.w_9_0 ctx) = (sequential_generatedRoot_42.w_10_0 ctx))

def sequential_generatedRoot_42.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_42.w_11_0 ctx)

structure sequential_generatedRoot_42.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_42_5 params i_0 i (current, ((sequential_generatedRoot_42.w_3_0 ctx), ((sequential_generatedRoot_42.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_42.w_2_0 ctx) (sequential_generatedRoot_42.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_42.outs ctx

theorem sequential_generatedRoot_42.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_42.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_42.constraints_0 params i_0 (sequential_generatedRoot_42.w_3_0 ctx) (sequential_generatedRoot_42.w_2_0 ctx) (sequential_generatedRoot_42.w_4_0 ctx) (sequential_generatedRoot_42.w_5_0 ctx) (sequential_generatedRoot_42.w_11_0 ctx) outputs) :
    sequential_generatedRoot_42.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_42.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_42.Facts_0 ctx outputs

theorem sequential_generatedRoot_42.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_42.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_42.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_42.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_42.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_44.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_44.Witness

def sequential_generatedRoot_44.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_44.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_44.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_44.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_44.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_44.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_44.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_44.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_44.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_44.w_5_0 ctx) = (sequential_generatedRoot_44.w_6_0 ctx))

def sequential_generatedRoot_44.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_44.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_44.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_44.w_1_0 ctx) + (sequential_generatedRoot_44.w_8_0 ctx))

def sequential_generatedRoot_44.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_44.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_44.w_9_0 ctx) = (sequential_generatedRoot_44.w_10_0 ctx))

def sequential_generatedRoot_44.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_44.w_11_0 ctx)

structure sequential_generatedRoot_44.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_44_5 params i_0 i (current, ((sequential_generatedRoot_44.w_3_0 ctx), ((sequential_generatedRoot_44.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_44.w_2_0 ctx) (sequential_generatedRoot_44.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_44.outs ctx

theorem sequential_generatedRoot_44.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_44.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_44.constraints_0 params i_0 (sequential_generatedRoot_44.w_3_0 ctx) (sequential_generatedRoot_44.w_2_0 ctx) (sequential_generatedRoot_44.w_4_0 ctx) (sequential_generatedRoot_44.w_5_0 ctx) (sequential_generatedRoot_44.w_11_0 ctx) outputs) :
    sequential_generatedRoot_44.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_44.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_44.Facts_0 ctx outputs

theorem sequential_generatedRoot_44.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_44.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_44.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_44.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_44.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_46.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_46.Witness

def sequential_generatedRoot_46.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_46.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_46.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_46.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_46.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_46.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_46.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_46.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_46.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_46.w_5_0 ctx) = (sequential_generatedRoot_46.w_6_0 ctx))

def sequential_generatedRoot_46.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_46.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_46.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_46.w_1_0 ctx) + (sequential_generatedRoot_46.w_8_0 ctx))

def sequential_generatedRoot_46.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_46.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_46.w_9_0 ctx) = (sequential_generatedRoot_46.w_10_0 ctx))

def sequential_generatedRoot_46.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_46.w_11_0 ctx)

structure sequential_generatedRoot_46.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_46_5 params i_0 i (current, ((sequential_generatedRoot_46.w_3_0 ctx), ((sequential_generatedRoot_46.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_46.w_2_0 ctx) (sequential_generatedRoot_46.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_46.outs ctx

theorem sequential_generatedRoot_46.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_46.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_46.constraints_0 params i_0 (sequential_generatedRoot_46.w_3_0 ctx) (sequential_generatedRoot_46.w_2_0 ctx) (sequential_generatedRoot_46.w_4_0 ctx) (sequential_generatedRoot_46.w_5_0 ctx) (sequential_generatedRoot_46.w_11_0 ctx) outputs) :
    sequential_generatedRoot_46.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_46.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_46.Facts_0 ctx outputs

theorem sequential_generatedRoot_46.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_46.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_46.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_46.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_46.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_48.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_48.Witness

def sequential_generatedRoot_48.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_48.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_48.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_48.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_48.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_48.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_48.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_48.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_48.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_48.w_5_0 ctx) = (sequential_generatedRoot_48.w_6_0 ctx))

def sequential_generatedRoot_48.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_48.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_48.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_48.w_1_0 ctx) + (sequential_generatedRoot_48.w_8_0 ctx))

def sequential_generatedRoot_48.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_48.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_48.w_9_0 ctx) = (sequential_generatedRoot_48.w_10_0 ctx))

def sequential_generatedRoot_48.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_48.w_11_0 ctx)

structure sequential_generatedRoot_48.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_48_5 params i_0 i (current, ((sequential_generatedRoot_48.w_3_0 ctx), ((sequential_generatedRoot_48.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_48.w_2_0 ctx) (sequential_generatedRoot_48.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_48.outs ctx

theorem sequential_generatedRoot_48.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_48.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_48.constraints_0 params i_0 (sequential_generatedRoot_48.w_3_0 ctx) (sequential_generatedRoot_48.w_2_0 ctx) (sequential_generatedRoot_48.w_4_0 ctx) (sequential_generatedRoot_48.w_5_0 ctx) (sequential_generatedRoot_48.w_11_0 ctx) outputs) :
    sequential_generatedRoot_48.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_48.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_48.Facts_0 ctx outputs

theorem sequential_generatedRoot_48.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_48.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_48.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_48.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_48.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_50.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_50.Witness

def sequential_generatedRoot_50.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_50.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_50.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_50.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_50.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_50.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_50.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_50.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_50.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_50.w_5_0 ctx) = (sequential_generatedRoot_50.w_6_0 ctx))

def sequential_generatedRoot_50.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_50.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_50.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_50.w_1_0 ctx) + (sequential_generatedRoot_50.w_8_0 ctx))

def sequential_generatedRoot_50.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_50.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_50.w_9_0 ctx) = (sequential_generatedRoot_50.w_10_0 ctx))

def sequential_generatedRoot_50.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_50.w_11_0 ctx)

structure sequential_generatedRoot_50.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_50_5 params i_0 i (current, ((sequential_generatedRoot_50.w_3_0 ctx), ((sequential_generatedRoot_50.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_50.w_2_0 ctx) (sequential_generatedRoot_50.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_50.outs ctx

theorem sequential_generatedRoot_50.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_50.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_50.constraints_0 params i_0 (sequential_generatedRoot_50.w_3_0 ctx) (sequential_generatedRoot_50.w_2_0 ctx) (sequential_generatedRoot_50.w_4_0 ctx) (sequential_generatedRoot_50.w_5_0 ctx) (sequential_generatedRoot_50.w_11_0 ctx) outputs) :
    sequential_generatedRoot_50.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_50.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_50.Facts_0 ctx outputs

theorem sequential_generatedRoot_50.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_50.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_50.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_50.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_50.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_52.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_52.Witness

def sequential_generatedRoot_52.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_52.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_52.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_52.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_52.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_52.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_52.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_52.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_52.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_52.w_5_0 ctx) = (sequential_generatedRoot_52.w_6_0 ctx))

def sequential_generatedRoot_52.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_52.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_52.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_52.w_1_0 ctx) + (sequential_generatedRoot_52.w_8_0 ctx))

def sequential_generatedRoot_52.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_52.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_52.w_9_0 ctx) = (sequential_generatedRoot_52.w_10_0 ctx))

def sequential_generatedRoot_52.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_52.w_11_0 ctx)

structure sequential_generatedRoot_52.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_52_5 params i_0 i (current, ((sequential_generatedRoot_52.w_3_0 ctx), ((sequential_generatedRoot_52.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_52.w_2_0 ctx) (sequential_generatedRoot_52.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_52.outs ctx

theorem sequential_generatedRoot_52.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_52.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_52.constraints_0 params i_0 (sequential_generatedRoot_52.w_3_0 ctx) (sequential_generatedRoot_52.w_2_0 ctx) (sequential_generatedRoot_52.w_4_0 ctx) (sequential_generatedRoot_52.w_5_0 ctx) (sequential_generatedRoot_52.w_11_0 ctx) outputs) :
    sequential_generatedRoot_52.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_52.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_52.Facts_0 ctx outputs

theorem sequential_generatedRoot_52.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_52.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_52.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_52.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_52.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_54.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_54.Witness

def sequential_generatedRoot_54.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_54.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_54.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_54.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_54.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_54.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_54.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_54.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_54.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_54.w_5_0 ctx) = (sequential_generatedRoot_54.w_6_0 ctx))

def sequential_generatedRoot_54.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_54.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_54.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_54.w_1_0 ctx) + (sequential_generatedRoot_54.w_8_0 ctx))

def sequential_generatedRoot_54.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_54.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_54.w_9_0 ctx) = (sequential_generatedRoot_54.w_10_0 ctx))

def sequential_generatedRoot_54.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_54.w_11_0 ctx)

structure sequential_generatedRoot_54.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_54_5 params i_0 i (current, ((sequential_generatedRoot_54.w_3_0 ctx), ((sequential_generatedRoot_54.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_54.w_2_0 ctx) (sequential_generatedRoot_54.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_54.outs ctx

theorem sequential_generatedRoot_54.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_54.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_54.constraints_0 params i_0 (sequential_generatedRoot_54.w_3_0 ctx) (sequential_generatedRoot_54.w_2_0 ctx) (sequential_generatedRoot_54.w_4_0 ctx) (sequential_generatedRoot_54.w_5_0 ctx) (sequential_generatedRoot_54.w_11_0 ctx) outputs) :
    sequential_generatedRoot_54.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_54.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_54.Facts_0 ctx outputs

theorem sequential_generatedRoot_54.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_54.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_54.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_54.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_54.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_56.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_56.Witness

def sequential_generatedRoot_56.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_56.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_56.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_56.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_56.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_56.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_56.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_56.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_56.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_56.w_5_0 ctx) = (sequential_generatedRoot_56.w_6_0 ctx))

def sequential_generatedRoot_56.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_56.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_56.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_56.w_1_0 ctx) + (sequential_generatedRoot_56.w_8_0 ctx))

def sequential_generatedRoot_56.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_56.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_56.w_9_0 ctx) = (sequential_generatedRoot_56.w_10_0 ctx))

def sequential_generatedRoot_56.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_56.w_11_0 ctx)

structure sequential_generatedRoot_56.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_56_5 params i_0 i (current, ((sequential_generatedRoot_56.w_3_0 ctx), ((sequential_generatedRoot_56.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_56.w_2_0 ctx) (sequential_generatedRoot_56.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_56.outs ctx

theorem sequential_generatedRoot_56.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_56.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_56.constraints_0 params i_0 (sequential_generatedRoot_56.w_3_0 ctx) (sequential_generatedRoot_56.w_2_0 ctx) (sequential_generatedRoot_56.w_4_0 ctx) (sequential_generatedRoot_56.w_5_0 ctx) (sequential_generatedRoot_56.w_11_0 ctx) outputs) :
    sequential_generatedRoot_56.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_56.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_56.Facts_0 ctx outputs

theorem sequential_generatedRoot_56.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_56.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_56.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_56.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_56.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_58.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_58.Witness

def sequential_generatedRoot_58.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_58.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_58.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_58.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_58.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_58.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_58.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_58.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_58.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_58.w_5_0 ctx) = (sequential_generatedRoot_58.w_6_0 ctx))

def sequential_generatedRoot_58.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_58.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_58.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_58.w_1_0 ctx) + (sequential_generatedRoot_58.w_8_0 ctx))

def sequential_generatedRoot_58.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_58.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_58.w_9_0 ctx) = (sequential_generatedRoot_58.w_10_0 ctx))

def sequential_generatedRoot_58.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_58.w_11_0 ctx)

structure sequential_generatedRoot_58.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_58_5 params i_0 i (current, ((sequential_generatedRoot_58.w_3_0 ctx), ((sequential_generatedRoot_58.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_58.w_2_0 ctx) (sequential_generatedRoot_58.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_58.outs ctx

theorem sequential_generatedRoot_58.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_58.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_58.constraints_0 params i_0 (sequential_generatedRoot_58.w_3_0 ctx) (sequential_generatedRoot_58.w_2_0 ctx) (sequential_generatedRoot_58.w_4_0 ctx) (sequential_generatedRoot_58.w_5_0 ctx) (sequential_generatedRoot_58.w_11_0 ctx) outputs) :
    sequential_generatedRoot_58.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_58.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_58.Facts_0 ctx outputs

theorem sequential_generatedRoot_58.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_58.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_58.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_58.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_58.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_60.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_60.Witness

def sequential_generatedRoot_60.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_60.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_60.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_60.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_60.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_60.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_60.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_60.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_60.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_60.w_5_0 ctx) = (sequential_generatedRoot_60.w_6_0 ctx))

def sequential_generatedRoot_60.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_60.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_60.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_60.w_1_0 ctx) + (sequential_generatedRoot_60.w_8_0 ctx))

def sequential_generatedRoot_60.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_60.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_60.w_9_0 ctx) = (sequential_generatedRoot_60.w_10_0 ctx))

def sequential_generatedRoot_60.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_60.w_11_0 ctx)

structure sequential_generatedRoot_60.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_60_5 params i_0 i (current, ((sequential_generatedRoot_60.w_3_0 ctx), ((sequential_generatedRoot_60.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_60.w_2_0 ctx) (sequential_generatedRoot_60.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_60.outs ctx

theorem sequential_generatedRoot_60.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_60.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_60.constraints_0 params i_0 (sequential_generatedRoot_60.w_3_0 ctx) (sequential_generatedRoot_60.w_2_0 ctx) (sequential_generatedRoot_60.w_4_0 ctx) (sequential_generatedRoot_60.w_5_0 ctx) (sequential_generatedRoot_60.w_11_0 ctx) outputs) :
    sequential_generatedRoot_60.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_60.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_60.Facts_0 ctx outputs

theorem sequential_generatedRoot_60.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_60.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_60.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_60.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_60.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_62.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_62.Witness

def sequential_generatedRoot_62.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_62.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_62.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_62.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_62.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_62.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_62.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_62.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_62.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_62.w_5_0 ctx) = (sequential_generatedRoot_62.w_6_0 ctx))

def sequential_generatedRoot_62.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_62.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_62.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_62.w_1_0 ctx) + (sequential_generatedRoot_62.w_8_0 ctx))

def sequential_generatedRoot_62.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_62.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_62.w_9_0 ctx) = (sequential_generatedRoot_62.w_10_0 ctx))

def sequential_generatedRoot_62.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_62.w_11_0 ctx)

structure sequential_generatedRoot_62.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_62_5 params i_0 i (current, ((sequential_generatedRoot_62.w_3_0 ctx), ((sequential_generatedRoot_62.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_62.w_2_0 ctx) (sequential_generatedRoot_62.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_62.outs ctx

theorem sequential_generatedRoot_62.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_62.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_62.constraints_0 params i_0 (sequential_generatedRoot_62.w_3_0 ctx) (sequential_generatedRoot_62.w_2_0 ctx) (sequential_generatedRoot_62.w_4_0 ctx) (sequential_generatedRoot_62.w_5_0 ctx) (sequential_generatedRoot_62.w_11_0 ctx) outputs) :
    sequential_generatedRoot_62.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_62.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_62.Facts_0 ctx outputs

theorem sequential_generatedRoot_62.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_62.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_62.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_62.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_62.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_64.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_64.Witness

def sequential_generatedRoot_64.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_64.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_64.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_64.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_64.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_64.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_64.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_64.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_64.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_64.w_5_0 ctx) = (sequential_generatedRoot_64.w_6_0 ctx))

def sequential_generatedRoot_64.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_64.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_64.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_64.w_1_0 ctx) + (sequential_generatedRoot_64.w_8_0 ctx))

def sequential_generatedRoot_64.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_64.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_64.w_9_0 ctx) = (sequential_generatedRoot_64.w_10_0 ctx))

def sequential_generatedRoot_64.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_64.w_11_0 ctx)

structure sequential_generatedRoot_64.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_64_5 params i_0 i (current, ((sequential_generatedRoot_64.w_3_0 ctx), ((sequential_generatedRoot_64.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_64.w_2_0 ctx) (sequential_generatedRoot_64.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_64.outs ctx

theorem sequential_generatedRoot_64.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_64.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_64.constraints_0 params i_0 (sequential_generatedRoot_64.w_3_0 ctx) (sequential_generatedRoot_64.w_2_0 ctx) (sequential_generatedRoot_64.w_4_0 ctx) (sequential_generatedRoot_64.w_5_0 ctx) (sequential_generatedRoot_64.w_11_0 ctx) outputs) :
    sequential_generatedRoot_64.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_64.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_64.Facts_0 ctx outputs

theorem sequential_generatedRoot_64.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_64.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_64.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_64.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_64.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_66.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_66.Witness

def sequential_generatedRoot_66.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_66.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_66.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_66.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_66.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_66.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_66.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_66.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_66.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_66.w_5_0 ctx) = (sequential_generatedRoot_66.w_6_0 ctx))

def sequential_generatedRoot_66.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_66.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_66.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_66.w_1_0 ctx) + (sequential_generatedRoot_66.w_8_0 ctx))

def sequential_generatedRoot_66.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_66.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_66.w_9_0 ctx) = (sequential_generatedRoot_66.w_10_0 ctx))

def sequential_generatedRoot_66.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_66.w_11_0 ctx)

structure sequential_generatedRoot_66.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_66_5 params i_0 i (current, ((sequential_generatedRoot_66.w_3_0 ctx), ((sequential_generatedRoot_66.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_66.w_2_0 ctx) (sequential_generatedRoot_66.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_66.outs ctx

theorem sequential_generatedRoot_66.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_66.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_66.constraints_0 params i_0 (sequential_generatedRoot_66.w_3_0 ctx) (sequential_generatedRoot_66.w_2_0 ctx) (sequential_generatedRoot_66.w_4_0 ctx) (sequential_generatedRoot_66.w_5_0 ctx) (sequential_generatedRoot_66.w_11_0 ctx) outputs) :
    sequential_generatedRoot_66.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_66.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_66.Facts_0 ctx outputs

theorem sequential_generatedRoot_66.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_66.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_66.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_66.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_66.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_68.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_68.Witness

def sequential_generatedRoot_68.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_68.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_68.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_68.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_68.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_68.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_68.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_68.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_68.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_68.w_5_0 ctx) = (sequential_generatedRoot_68.w_6_0 ctx))

def sequential_generatedRoot_68.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_68.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_68.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_68.w_1_0 ctx) + (sequential_generatedRoot_68.w_8_0 ctx))

def sequential_generatedRoot_68.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_68.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_68.w_9_0 ctx) = (sequential_generatedRoot_68.w_10_0 ctx))

def sequential_generatedRoot_68.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_68.w_11_0 ctx)

structure sequential_generatedRoot_68.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_68_5 params i_0 i (current, ((sequential_generatedRoot_68.w_3_0 ctx), ((sequential_generatedRoot_68.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_68.w_2_0 ctx) (sequential_generatedRoot_68.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_68.outs ctx

theorem sequential_generatedRoot_68.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_68.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_68.constraints_0 params i_0 (sequential_generatedRoot_68.w_3_0 ctx) (sequential_generatedRoot_68.w_2_0 ctx) (sequential_generatedRoot_68.w_4_0 ctx) (sequential_generatedRoot_68.w_5_0 ctx) (sequential_generatedRoot_68.w_11_0 ctx) outputs) :
    sequential_generatedRoot_68.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_68.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_68.Facts_0 ctx outputs

theorem sequential_generatedRoot_68.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_68.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_68.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_68.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_68.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_70.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_70.Witness

def sequential_generatedRoot_70.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_70.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_70.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_70.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_70.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_70.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_70.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_70.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_70.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_70.w_5_0 ctx) = (sequential_generatedRoot_70.w_6_0 ctx))

def sequential_generatedRoot_70.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_70.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_70.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_70.w_1_0 ctx) + (sequential_generatedRoot_70.w_8_0 ctx))

def sequential_generatedRoot_70.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_70.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_70.w_9_0 ctx) = (sequential_generatedRoot_70.w_10_0 ctx))

def sequential_generatedRoot_70.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_70.w_11_0 ctx)

structure sequential_generatedRoot_70.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_70_5 params i_0 i (current, ((sequential_generatedRoot_70.w_3_0 ctx), ((sequential_generatedRoot_70.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_70.w_2_0 ctx) (sequential_generatedRoot_70.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_70.outs ctx

theorem sequential_generatedRoot_70.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_70.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_70.constraints_0 params i_0 (sequential_generatedRoot_70.w_3_0 ctx) (sequential_generatedRoot_70.w_2_0 ctx) (sequential_generatedRoot_70.w_4_0 ctx) (sequential_generatedRoot_70.w_5_0 ctx) (sequential_generatedRoot_70.w_11_0 ctx) outputs) :
    sequential_generatedRoot_70.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_70.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_70.Facts_0 ctx outputs

theorem sequential_generatedRoot_70.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_70.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_70.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_70.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_70.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_72.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_72.Witness

def sequential_generatedRoot_72.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_72.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_72.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_72.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_72.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_72.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_72.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_72.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_72.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_72.w_5_0 ctx) = (sequential_generatedRoot_72.w_6_0 ctx))

def sequential_generatedRoot_72.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_72.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_72.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_72.w_1_0 ctx) + (sequential_generatedRoot_72.w_8_0 ctx))

def sequential_generatedRoot_72.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_72.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_72.w_9_0 ctx) = (sequential_generatedRoot_72.w_10_0 ctx))

def sequential_generatedRoot_72.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_72.w_11_0 ctx)

structure sequential_generatedRoot_72.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_72_5 params i_0 i (current, ((sequential_generatedRoot_72.w_3_0 ctx), ((sequential_generatedRoot_72.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_72.w_2_0 ctx) (sequential_generatedRoot_72.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_72.outs ctx

theorem sequential_generatedRoot_72.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_72.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_72.constraints_0 params i_0 (sequential_generatedRoot_72.w_3_0 ctx) (sequential_generatedRoot_72.w_2_0 ctx) (sequential_generatedRoot_72.w_4_0 ctx) (sequential_generatedRoot_72.w_5_0 ctx) (sequential_generatedRoot_72.w_11_0 ctx) outputs) :
    sequential_generatedRoot_72.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_72.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_72.Facts_0 ctx outputs

theorem sequential_generatedRoot_72.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_72.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_72.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_72.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_72.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_74.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_74.Witness

def sequential_generatedRoot_74.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_74.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_74.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_74.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_74.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_74.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_74.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_74.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_74.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_74.w_5_0 ctx) = (sequential_generatedRoot_74.w_6_0 ctx))

def sequential_generatedRoot_74.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_74.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_74.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_74.w_1_0 ctx) + (sequential_generatedRoot_74.w_8_0 ctx))

def sequential_generatedRoot_74.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_74.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_74.w_9_0 ctx) = (sequential_generatedRoot_74.w_10_0 ctx))

def sequential_generatedRoot_74.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_74.w_11_0 ctx)

structure sequential_generatedRoot_74.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 params i_0 i (current, ((sequential_generatedRoot_74.w_3_0 ctx), ((sequential_generatedRoot_74.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_74.outs ctx

theorem sequential_generatedRoot_74.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_74.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_74.constraints_0 params i_0 (sequential_generatedRoot_74.w_3_0 ctx) (sequential_generatedRoot_74.w_2_0 ctx) (sequential_generatedRoot_74.w_4_0 ctx) (sequential_generatedRoot_74.w_5_0 ctx) (sequential_generatedRoot_74.w_11_0 ctx) outputs) :
    sequential_generatedRoot_74.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_74.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_74.Facts_0 ctx outputs

theorem sequential_generatedRoot_74.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_74.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_74.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_74.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_74.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_76.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_76.Witness

def sequential_generatedRoot_76.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_76.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_76.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_76.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_76.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_76.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_76.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_76.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_76.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_76.w_5_0 ctx) = (sequential_generatedRoot_76.w_6_0 ctx))

def sequential_generatedRoot_76.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_76.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_76.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_76.w_1_0 ctx) + (sequential_generatedRoot_76.w_8_0 ctx))

def sequential_generatedRoot_76.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_76.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_76.w_9_0 ctx) = (sequential_generatedRoot_76.w_10_0 ctx))

def sequential_generatedRoot_76.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_76.w_11_0 ctx)

structure sequential_generatedRoot_76.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 params i_0 i (current, ((sequential_generatedRoot_76.w_3_0 ctx), ((sequential_generatedRoot_76.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_76.outs ctx

theorem sequential_generatedRoot_76.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_76.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_76.constraints_0 params i_0 (sequential_generatedRoot_76.w_3_0 ctx) (sequential_generatedRoot_76.w_2_0 ctx) (sequential_generatedRoot_76.w_4_0 ctx) (sequential_generatedRoot_76.w_5_0 ctx) (sequential_generatedRoot_76.w_11_0 ctx) outputs) :
    sequential_generatedRoot_76.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_76.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_76.Facts_0 ctx outputs

theorem sequential_generatedRoot_76.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_76.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_76.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_76.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_76.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_78.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_78.Witness

def sequential_generatedRoot_78.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_78.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_78.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_78.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_78.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_78.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_78.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_78.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_78.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_78.w_5_0 ctx) = (sequential_generatedRoot_78.w_6_0 ctx))

def sequential_generatedRoot_78.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_78.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_78.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_78.w_1_0 ctx) + (sequential_generatedRoot_78.w_8_0 ctx))

def sequential_generatedRoot_78.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_78.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_78.w_9_0 ctx) = (sequential_generatedRoot_78.w_10_0 ctx))

def sequential_generatedRoot_78.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_78.w_11_0 ctx)

structure sequential_generatedRoot_78.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 params i_0 i (current, ((sequential_generatedRoot_78.w_3_0 ctx), ((sequential_generatedRoot_78.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_78.outs ctx

theorem sequential_generatedRoot_78.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_78.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_78.constraints_0 params i_0 (sequential_generatedRoot_78.w_3_0 ctx) (sequential_generatedRoot_78.w_2_0 ctx) (sequential_generatedRoot_78.w_4_0 ctx) (sequential_generatedRoot_78.w_5_0 ctx) (sequential_generatedRoot_78.w_11_0 ctx) outputs) :
    sequential_generatedRoot_78.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_78.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_78.Facts_0 ctx outputs

theorem sequential_generatedRoot_78.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_78.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_78.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_78.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_78.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_80.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_80.Witness

def sequential_generatedRoot_80.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_80.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_80.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_80.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_80.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_80.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_80.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_80.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_80.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_80.w_5_0 ctx) = (sequential_generatedRoot_80.w_6_0 ctx))

def sequential_generatedRoot_80.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_80.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_80.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_80.w_1_0 ctx) + (sequential_generatedRoot_80.w_8_0 ctx))

def sequential_generatedRoot_80.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_80.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_80.w_9_0 ctx) = (sequential_generatedRoot_80.w_10_0 ctx))

def sequential_generatedRoot_80.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_80.w_11_0 ctx)

structure sequential_generatedRoot_80.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 params i_0 i (current, ((sequential_generatedRoot_80.w_3_0 ctx), ((sequential_generatedRoot_80.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_80.outs ctx

theorem sequential_generatedRoot_80.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_80.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_80.constraints_0 params i_0 (sequential_generatedRoot_80.w_3_0 ctx) (sequential_generatedRoot_80.w_2_0 ctx) (sequential_generatedRoot_80.w_4_0 ctx) (sequential_generatedRoot_80.w_5_0 ctx) (sequential_generatedRoot_80.w_11_0 ctx) outputs) :
    sequential_generatedRoot_80.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_80.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_80.Facts_0 ctx outputs

theorem sequential_generatedRoot_80.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_80.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_80.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_80.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_80.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_82.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_82.Witness

def sequential_generatedRoot_82.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_82.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_82.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_82.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_82.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_82.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_82.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_82.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_82.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_82.w_5_0 ctx) = (sequential_generatedRoot_82.w_6_0 ctx))

def sequential_generatedRoot_82.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_82.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_82.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_82.w_1_0 ctx) + (sequential_generatedRoot_82.w_8_0 ctx))

def sequential_generatedRoot_82.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_82.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_82.w_9_0 ctx) = (sequential_generatedRoot_82.w_10_0 ctx))

def sequential_generatedRoot_82.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_82.w_11_0 ctx)

structure sequential_generatedRoot_82.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 params i_0 i (current, ((sequential_generatedRoot_82.w_3_0 ctx), ((sequential_generatedRoot_82.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_82.outs ctx

theorem sequential_generatedRoot_82.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_82.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_82.constraints_0 params i_0 (sequential_generatedRoot_82.w_3_0 ctx) (sequential_generatedRoot_82.w_2_0 ctx) (sequential_generatedRoot_82.w_4_0 ctx) (sequential_generatedRoot_82.w_5_0 ctx) (sequential_generatedRoot_82.w_11_0 ctx) outputs) :
    sequential_generatedRoot_82.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_82.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_82.Facts_0 ctx outputs

theorem sequential_generatedRoot_82.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_82.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_82.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_82.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_82.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_84.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_84.Witness

def sequential_generatedRoot_84.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_84.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_84.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_84.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_84.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_84.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_84.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_84.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_84.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_84.w_5_0 ctx) = (sequential_generatedRoot_84.w_6_0 ctx))

def sequential_generatedRoot_84.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_84.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_84.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_84.w_1_0 ctx) + (sequential_generatedRoot_84.w_8_0 ctx))

def sequential_generatedRoot_84.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_84.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_84.w_9_0 ctx) = (sequential_generatedRoot_84.w_10_0 ctx))

def sequential_generatedRoot_84.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_84.w_11_0 ctx)

structure sequential_generatedRoot_84.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 params i_0 i (current, ((sequential_generatedRoot_84.w_3_0 ctx), ((sequential_generatedRoot_84.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_84.outs ctx

theorem sequential_generatedRoot_84.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_84.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_84.constraints_0 params i_0 (sequential_generatedRoot_84.w_3_0 ctx) (sequential_generatedRoot_84.w_2_0 ctx) (sequential_generatedRoot_84.w_4_0 ctx) (sequential_generatedRoot_84.w_5_0 ctx) (sequential_generatedRoot_84.w_11_0 ctx) outputs) :
    sequential_generatedRoot_84.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_84.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_84.Facts_0 ctx outputs

theorem sequential_generatedRoot_84.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_84.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_84.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_84.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_84.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_86.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_86.Witness

def sequential_generatedRoot_86.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_86.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_86.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_86.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_86.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_86.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_86.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_86.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_86.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_86.w_5_0 ctx) = (sequential_generatedRoot_86.w_6_0 ctx))

def sequential_generatedRoot_86.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_86.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_86.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_86.w_1_0 ctx) + (sequential_generatedRoot_86.w_8_0 ctx))

def sequential_generatedRoot_86.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_86.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_86.w_9_0 ctx) = (sequential_generatedRoot_86.w_10_0 ctx))

def sequential_generatedRoot_86.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_86.w_11_0 ctx)

structure sequential_generatedRoot_86.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 params i_0 i (current, ((sequential_generatedRoot_86.w_3_0 ctx), ((sequential_generatedRoot_86.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_86.outs ctx

theorem sequential_generatedRoot_86.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_86.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_86.constraints_0 params i_0 (sequential_generatedRoot_86.w_3_0 ctx) (sequential_generatedRoot_86.w_2_0 ctx) (sequential_generatedRoot_86.w_4_0 ctx) (sequential_generatedRoot_86.w_5_0 ctx) (sequential_generatedRoot_86.w_11_0 ctx) outputs) :
    sequential_generatedRoot_86.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_86.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_86.Facts_0 ctx outputs

theorem sequential_generatedRoot_86.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_86.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_86.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_86.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_86.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_88.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_88.Witness

def sequential_generatedRoot_88.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_88.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_88.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_88.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_88.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_88.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_88.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_88.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_88.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_88.w_5_0 ctx) = (sequential_generatedRoot_88.w_6_0 ctx))

def sequential_generatedRoot_88.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_88.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_88.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_88.w_1_0 ctx) + (sequential_generatedRoot_88.w_8_0 ctx))

def sequential_generatedRoot_88.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_88.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_88.w_9_0 ctx) = (sequential_generatedRoot_88.w_10_0 ctx))

def sequential_generatedRoot_88.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_88.w_11_0 ctx)

structure sequential_generatedRoot_88.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 params i_0 i (current, ((sequential_generatedRoot_88.w_3_0 ctx), ((sequential_generatedRoot_88.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_88.outs ctx

theorem sequential_generatedRoot_88.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_88.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_88.constraints_0 params i_0 (sequential_generatedRoot_88.w_3_0 ctx) (sequential_generatedRoot_88.w_2_0 ctx) (sequential_generatedRoot_88.w_4_0 ctx) (sequential_generatedRoot_88.w_5_0 ctx) (sequential_generatedRoot_88.w_11_0 ctx) outputs) :
    sequential_generatedRoot_88.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_88.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_88.Facts_0 ctx outputs

theorem sequential_generatedRoot_88.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_88.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_88.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_88.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_88.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_90.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_90.Witness

def sequential_generatedRoot_90.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_90.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_90.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_90.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_90.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_90.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_90.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_90.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_90.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_90.w_5_0 ctx) = (sequential_generatedRoot_90.w_6_0 ctx))

def sequential_generatedRoot_90.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_90.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_90.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_90.w_1_0 ctx) + (sequential_generatedRoot_90.w_8_0 ctx))

def sequential_generatedRoot_90.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_90.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_90.w_9_0 ctx) = (sequential_generatedRoot_90.w_10_0 ctx))

def sequential_generatedRoot_90.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_90.w_11_0 ctx)

structure sequential_generatedRoot_90.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_90_5 params i_0 i (current, ((sequential_generatedRoot_90.w_3_0 ctx), ((sequential_generatedRoot_90.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_90.w_2_0 ctx) (sequential_generatedRoot_90.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_90.outs ctx

theorem sequential_generatedRoot_90.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_90.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_90.constraints_0 params i_0 (sequential_generatedRoot_90.w_3_0 ctx) (sequential_generatedRoot_90.w_2_0 ctx) (sequential_generatedRoot_90.w_4_0 ctx) (sequential_generatedRoot_90.w_5_0 ctx) (sequential_generatedRoot_90.w_11_0 ctx) outputs) :
    sequential_generatedRoot_90.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_90.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_90.Facts_0 ctx outputs

theorem sequential_generatedRoot_90.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_90.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_90.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_90.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_90.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_92.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_92.Witness

def sequential_generatedRoot_92.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_92.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_92.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_92.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_92.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_92.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_92.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_92.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_92.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_92.w_5_0 ctx) = (sequential_generatedRoot_92.w_6_0 ctx))

def sequential_generatedRoot_92.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_92.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_92.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_92.w_1_0 ctx) + (sequential_generatedRoot_92.w_8_0 ctx))

def sequential_generatedRoot_92.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_92.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_92.w_9_0 ctx) = (sequential_generatedRoot_92.w_10_0 ctx))

def sequential_generatedRoot_92.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_92.w_11_0 ctx)

structure sequential_generatedRoot_92.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_92_5 params i_0 i (current, ((sequential_generatedRoot_92.w_3_0 ctx), ((sequential_generatedRoot_92.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_92.w_2_0 ctx) (sequential_generatedRoot_92.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_92.outs ctx

theorem sequential_generatedRoot_92.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_92.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_92.constraints_0 params i_0 (sequential_generatedRoot_92.w_3_0 ctx) (sequential_generatedRoot_92.w_2_0 ctx) (sequential_generatedRoot_92.w_4_0 ctx) (sequential_generatedRoot_92.w_5_0 ctx) (sequential_generatedRoot_92.w_11_0 ctx) outputs) :
    sequential_generatedRoot_92.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_92.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_92.Facts_0 ctx outputs

theorem sequential_generatedRoot_92.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_92.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_92.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_92.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_92.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_94.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_94.Witness

def sequential_generatedRoot_94.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_94.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_94.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_94.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_94.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_94.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_94.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_94.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_94.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_94.w_5_0 ctx) = (sequential_generatedRoot_94.w_6_0 ctx))

def sequential_generatedRoot_94.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_94.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_94.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_94.w_1_0 ctx) + (sequential_generatedRoot_94.w_8_0 ctx))

def sequential_generatedRoot_94.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_94.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_94.w_9_0 ctx) = (sequential_generatedRoot_94.w_10_0 ctx))

def sequential_generatedRoot_94.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_94.w_11_0 ctx)

structure sequential_generatedRoot_94.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_94_5 params i_0 i (current, ((sequential_generatedRoot_94.w_3_0 ctx), ((sequential_generatedRoot_94.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_94.w_2_0 ctx) (sequential_generatedRoot_94.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_94.outs ctx

theorem sequential_generatedRoot_94.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_94.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_94.constraints_0 params i_0 (sequential_generatedRoot_94.w_3_0 ctx) (sequential_generatedRoot_94.w_2_0 ctx) (sequential_generatedRoot_94.w_4_0 ctx) (sequential_generatedRoot_94.w_5_0 ctx) (sequential_generatedRoot_94.w_11_0 ctx) outputs) :
    sequential_generatedRoot_94.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_94.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_94.Facts_0 ctx outputs

theorem sequential_generatedRoot_94.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_94.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_94.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_94.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_94.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_96.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_96.Witness

def sequential_generatedRoot_96.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_96.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_96.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_96.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_96.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_96.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_96.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_96.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_96.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_96.w_5_0 ctx) = (sequential_generatedRoot_96.w_6_0 ctx))

def sequential_generatedRoot_96.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_96.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_96.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_96.w_1_0 ctx) + (sequential_generatedRoot_96.w_8_0 ctx))

def sequential_generatedRoot_96.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_96.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_96.w_9_0 ctx) = (sequential_generatedRoot_96.w_10_0 ctx))

def sequential_generatedRoot_96.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_96.w_11_0 ctx)

structure sequential_generatedRoot_96.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_96_5 params i_0 i (current, ((sequential_generatedRoot_96.w_3_0 ctx), ((sequential_generatedRoot_96.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_96.w_2_0 ctx) (sequential_generatedRoot_96.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_96.outs ctx

theorem sequential_generatedRoot_96.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_96.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_96.constraints_0 params i_0 (sequential_generatedRoot_96.w_3_0 ctx) (sequential_generatedRoot_96.w_2_0 ctx) (sequential_generatedRoot_96.w_4_0 ctx) (sequential_generatedRoot_96.w_5_0 ctx) (sequential_generatedRoot_96.w_11_0 ctx) outputs) :
    sequential_generatedRoot_96.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_96.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_96.Facts_0 ctx outputs

theorem sequential_generatedRoot_96.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_96.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_96.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_96.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_96.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_98.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_98.Witness

def sequential_generatedRoot_98.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_98.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_98.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_98.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_98.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_98.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_98.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_98.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_98.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_98.w_5_0 ctx) = (sequential_generatedRoot_98.w_6_0 ctx))

def sequential_generatedRoot_98.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_98.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_98.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_98.w_1_0 ctx) + (sequential_generatedRoot_98.w_8_0 ctx))

def sequential_generatedRoot_98.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_98.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_98.w_9_0 ctx) = (sequential_generatedRoot_98.w_10_0 ctx))

def sequential_generatedRoot_98.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_98.w_11_0 ctx)

structure sequential_generatedRoot_98.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_98_5 params i_0 i (current, ((sequential_generatedRoot_98.w_3_0 ctx), ((sequential_generatedRoot_98.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_98.w_2_0 ctx) (sequential_generatedRoot_98.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_98.outs ctx

theorem sequential_generatedRoot_98.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_98.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_98.constraints_0 params i_0 (sequential_generatedRoot_98.w_3_0 ctx) (sequential_generatedRoot_98.w_2_0 ctx) (sequential_generatedRoot_98.w_4_0 ctx) (sequential_generatedRoot_98.w_5_0 ctx) (sequential_generatedRoot_98.w_11_0 ctx) outputs) :
    sequential_generatedRoot_98.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_98.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_98.Facts_0 ctx outputs

theorem sequential_generatedRoot_98.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_98.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_98.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_98.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_98.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_100.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_100.Witness

def sequential_generatedRoot_100.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_100.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_100.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_100.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_100.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_100.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_100.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_100.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_100.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_100.w_5_0 ctx) = (sequential_generatedRoot_100.w_6_0 ctx))

def sequential_generatedRoot_100.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_100.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_100.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_100.w_1_0 ctx) + (sequential_generatedRoot_100.w_8_0 ctx))

def sequential_generatedRoot_100.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_100.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_100.w_9_0 ctx) = (sequential_generatedRoot_100.w_10_0 ctx))

def sequential_generatedRoot_100.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_100.w_11_0 ctx)

structure sequential_generatedRoot_100.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_100_5 params i_0 i (current, ((sequential_generatedRoot_100.w_3_0 ctx), ((sequential_generatedRoot_100.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_100.w_2_0 ctx) (sequential_generatedRoot_100.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_100.outs ctx

theorem sequential_generatedRoot_100.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_100.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_100.constraints_0 params i_0 (sequential_generatedRoot_100.w_3_0 ctx) (sequential_generatedRoot_100.w_2_0 ctx) (sequential_generatedRoot_100.w_4_0 ctx) (sequential_generatedRoot_100.w_5_0 ctx) (sequential_generatedRoot_100.w_11_0 ctx) outputs) :
    sequential_generatedRoot_100.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_100.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_100.Facts_0 ctx outputs

theorem sequential_generatedRoot_100.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_100.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_100.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_100.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_100.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_102.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_102.Witness

def sequential_generatedRoot_102.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_102.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_102.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_102.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_102.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_102.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_102.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_102.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_102.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_102.w_5_0 ctx) = (sequential_generatedRoot_102.w_6_0 ctx))

def sequential_generatedRoot_102.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_102.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_102.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_102.w_1_0 ctx) + (sequential_generatedRoot_102.w_8_0 ctx))

def sequential_generatedRoot_102.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_102.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_102.w_9_0 ctx) = (sequential_generatedRoot_102.w_10_0 ctx))

def sequential_generatedRoot_102.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_102.w_11_0 ctx)

structure sequential_generatedRoot_102.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_102_5 params i_0 i (current, ((sequential_generatedRoot_102.w_3_0 ctx), ((sequential_generatedRoot_102.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_102.w_2_0 ctx) (sequential_generatedRoot_102.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_102.outs ctx

theorem sequential_generatedRoot_102.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_102.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_102.constraints_0 params i_0 (sequential_generatedRoot_102.w_3_0 ctx) (sequential_generatedRoot_102.w_2_0 ctx) (sequential_generatedRoot_102.w_4_0 ctx) (sequential_generatedRoot_102.w_5_0 ctx) (sequential_generatedRoot_102.w_11_0 ctx) outputs) :
    sequential_generatedRoot_102.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_102.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_102.Facts_0 ctx outputs

theorem sequential_generatedRoot_102.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_102.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_102.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_102.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_102.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_104.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_104.Witness

def sequential_generatedRoot_104.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_104.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_104.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_104.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_104.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_104.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_104.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_104.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_104.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_104.w_5_0 ctx) = (sequential_generatedRoot_104.w_6_0 ctx))

def sequential_generatedRoot_104.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_104.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_104.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_104.w_1_0 ctx) + (sequential_generatedRoot_104.w_8_0 ctx))

def sequential_generatedRoot_104.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_104.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_104.w_9_0 ctx) = (sequential_generatedRoot_104.w_10_0 ctx))

def sequential_generatedRoot_104.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_104.w_11_0 ctx)

structure sequential_generatedRoot_104.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_104_5 params i_0 i (current, ((sequential_generatedRoot_104.w_3_0 ctx), ((sequential_generatedRoot_104.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_104.w_2_0 ctx) (sequential_generatedRoot_104.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_104.outs ctx

theorem sequential_generatedRoot_104.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_104.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_104.constraints_0 params i_0 (sequential_generatedRoot_104.w_3_0 ctx) (sequential_generatedRoot_104.w_2_0 ctx) (sequential_generatedRoot_104.w_4_0 ctx) (sequential_generatedRoot_104.w_5_0 ctx) (sequential_generatedRoot_104.w_11_0 ctx) outputs) :
    sequential_generatedRoot_104.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_104.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_104.Facts_0 ctx outputs

theorem sequential_generatedRoot_104.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_104.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_104.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_104.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_104.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_106.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_106.Witness

def sequential_generatedRoot_106.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_106.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_106.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_106.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_106.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_106.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_106.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_106.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_106.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_106.w_5_0 ctx) = (sequential_generatedRoot_106.w_6_0 ctx))

def sequential_generatedRoot_106.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_106.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_106.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_106.w_1_0 ctx) + (sequential_generatedRoot_106.w_8_0 ctx))

def sequential_generatedRoot_106.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_106.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_106.w_9_0 ctx) = (sequential_generatedRoot_106.w_10_0 ctx))

def sequential_generatedRoot_106.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_106.w_11_0 ctx)

structure sequential_generatedRoot_106.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_106_5 params i_0 i (current, ((sequential_generatedRoot_106.w_3_0 ctx), ((sequential_generatedRoot_106.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_106.w_2_0 ctx) (sequential_generatedRoot_106.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_106.outs ctx

theorem sequential_generatedRoot_106.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_106.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_106.constraints_0 params i_0 (sequential_generatedRoot_106.w_3_0 ctx) (sequential_generatedRoot_106.w_2_0 ctx) (sequential_generatedRoot_106.w_4_0 ctx) (sequential_generatedRoot_106.w_5_0 ctx) (sequential_generatedRoot_106.w_11_0 ctx) outputs) :
    sequential_generatedRoot_106.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_106.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_106.Facts_0 ctx outputs

theorem sequential_generatedRoot_106.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_106.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_106.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_106.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_106.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_108.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_108.Witness

def sequential_generatedRoot_108.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_108.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_108.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_108.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_108.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_108.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_108.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_108.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_108.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_108.w_5_0 ctx) = (sequential_generatedRoot_108.w_6_0 ctx))

def sequential_generatedRoot_108.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_108.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_108.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_108.w_1_0 ctx) + (sequential_generatedRoot_108.w_8_0 ctx))

def sequential_generatedRoot_108.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_108.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_108.w_9_0 ctx) = (sequential_generatedRoot_108.w_10_0 ctx))

def sequential_generatedRoot_108.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_108.w_11_0 ctx)

structure sequential_generatedRoot_108.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_108_5 params i_0 i (current, ((sequential_generatedRoot_108.w_3_0 ctx), ((sequential_generatedRoot_108.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_108.w_2_0 ctx) (sequential_generatedRoot_108.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_108.outs ctx

theorem sequential_generatedRoot_108.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_108.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_108.constraints_0 params i_0 (sequential_generatedRoot_108.w_3_0 ctx) (sequential_generatedRoot_108.w_2_0 ctx) (sequential_generatedRoot_108.w_4_0 ctx) (sequential_generatedRoot_108.w_5_0 ctx) (sequential_generatedRoot_108.w_11_0 ctx) outputs) :
    sequential_generatedRoot_108.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_108.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_108.Facts_0 ctx outputs

theorem sequential_generatedRoot_108.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_108.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_108.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_108.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_108.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_110.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_110.Witness

def sequential_generatedRoot_110.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_110.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_110.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_110.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_110.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_110.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_110.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_110.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_110.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_110.w_5_0 ctx) = (sequential_generatedRoot_110.w_6_0 ctx))

def sequential_generatedRoot_110.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_110.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_110.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_110.w_1_0 ctx) + (sequential_generatedRoot_110.w_8_0 ctx))

def sequential_generatedRoot_110.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_110.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_110.w_9_0 ctx) = (sequential_generatedRoot_110.w_10_0 ctx))

def sequential_generatedRoot_110.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_110.w_11_0 ctx)

structure sequential_generatedRoot_110.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_110_5 params i_0 i (current, ((sequential_generatedRoot_110.w_3_0 ctx), ((sequential_generatedRoot_110.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_110.w_2_0 ctx) (sequential_generatedRoot_110.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_110.outs ctx

theorem sequential_generatedRoot_110.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_110.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_110.constraints_0 params i_0 (sequential_generatedRoot_110.w_3_0 ctx) (sequential_generatedRoot_110.w_2_0 ctx) (sequential_generatedRoot_110.w_4_0 ctx) (sequential_generatedRoot_110.w_5_0 ctx) (sequential_generatedRoot_110.w_11_0 ctx) outputs) :
    sequential_generatedRoot_110.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_110.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_110.Facts_0 ctx outputs

theorem sequential_generatedRoot_110.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_110.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_110.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_110.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_110.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_112.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_112.Witness

def sequential_generatedRoot_112.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_112.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_112.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_112.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_112.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_112.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_112.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_112.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_112.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_112.w_5_0 ctx) = (sequential_generatedRoot_112.w_6_0 ctx))

def sequential_generatedRoot_112.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_112.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_112.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_112.w_1_0 ctx) + (sequential_generatedRoot_112.w_8_0 ctx))

def sequential_generatedRoot_112.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_112.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_112.w_9_0 ctx) = (sequential_generatedRoot_112.w_10_0 ctx))

def sequential_generatedRoot_112.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_112.w_11_0 ctx)

structure sequential_generatedRoot_112.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_112_5 params i_0 i (current, ((sequential_generatedRoot_112.w_3_0 ctx), ((sequential_generatedRoot_112.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_112.w_2_0 ctx) (sequential_generatedRoot_112.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_112.outs ctx

theorem sequential_generatedRoot_112.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_112.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_112.constraints_0 params i_0 (sequential_generatedRoot_112.w_3_0 ctx) (sequential_generatedRoot_112.w_2_0 ctx) (sequential_generatedRoot_112.w_4_0 ctx) (sequential_generatedRoot_112.w_5_0 ctx) (sequential_generatedRoot_112.w_11_0 ctx) outputs) :
    sequential_generatedRoot_112.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_112.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_112.Facts_0 ctx outputs

theorem sequential_generatedRoot_112.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_112.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_112.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_112.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_112.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_114.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_114.Witness

def sequential_generatedRoot_114.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_114.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_114.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_114.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_114.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_114.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_114.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_114.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_114.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_114.w_5_0 ctx) = (sequential_generatedRoot_114.w_6_0 ctx))

def sequential_generatedRoot_114.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_114.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_114.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_114.w_1_0 ctx) + (sequential_generatedRoot_114.w_8_0 ctx))

def sequential_generatedRoot_114.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_114.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_114.w_9_0 ctx) = (sequential_generatedRoot_114.w_10_0 ctx))

def sequential_generatedRoot_114.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_114.w_11_0 ctx)

structure sequential_generatedRoot_114.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_114_5 params i_0 i (current, ((sequential_generatedRoot_114.w_3_0 ctx), ((sequential_generatedRoot_114.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_114.w_2_0 ctx) (sequential_generatedRoot_114.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_114.outs ctx

theorem sequential_generatedRoot_114.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_114.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_114.constraints_0 params i_0 (sequential_generatedRoot_114.w_3_0 ctx) (sequential_generatedRoot_114.w_2_0 ctx) (sequential_generatedRoot_114.w_4_0 ctx) (sequential_generatedRoot_114.w_5_0 ctx) (sequential_generatedRoot_114.w_11_0 ctx) outputs) :
    sequential_generatedRoot_114.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_114.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_114.Facts_0 ctx outputs

theorem sequential_generatedRoot_114.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_114.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_114.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_114.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_114.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_116.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_116.Witness

def sequential_generatedRoot_116.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_116.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_116.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_116.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_116.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_116.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_116.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_116.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_116.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_116.w_5_0 ctx) = (sequential_generatedRoot_116.w_6_0 ctx))

def sequential_generatedRoot_116.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_116.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_116.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_116.w_1_0 ctx) + (sequential_generatedRoot_116.w_8_0 ctx))

def sequential_generatedRoot_116.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_116.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_116.w_9_0 ctx) = (sequential_generatedRoot_116.w_10_0 ctx))

def sequential_generatedRoot_116.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_116.w_11_0 ctx)

structure sequential_generatedRoot_116.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_116_5 params i_0 i (current, ((sequential_generatedRoot_116.w_3_0 ctx), ((sequential_generatedRoot_116.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_116.w_2_0 ctx) (sequential_generatedRoot_116.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_116.outs ctx

theorem sequential_generatedRoot_116.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_116.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_116.constraints_0 params i_0 (sequential_generatedRoot_116.w_3_0 ctx) (sequential_generatedRoot_116.w_2_0 ctx) (sequential_generatedRoot_116.w_4_0 ctx) (sequential_generatedRoot_116.w_5_0 ctx) (sequential_generatedRoot_116.w_11_0 ctx) outputs) :
    sequential_generatedRoot_116.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_116.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_116.Facts_0 ctx outputs

theorem sequential_generatedRoot_116.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_116.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_116.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_116.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_116.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_118.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_118.Witness

def sequential_generatedRoot_118.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_118.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_118.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_118.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_118.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_118.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_118.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_118.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_118.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_118.w_5_0 ctx) = (sequential_generatedRoot_118.w_6_0 ctx))

def sequential_generatedRoot_118.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_118.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_118.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_118.w_1_0 ctx) + (sequential_generatedRoot_118.w_8_0 ctx))

def sequential_generatedRoot_118.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_118.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_118.w_9_0 ctx) = (sequential_generatedRoot_118.w_10_0 ctx))

def sequential_generatedRoot_118.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_118.w_11_0 ctx)

structure sequential_generatedRoot_118.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_118_5 params i_0 i (current, ((sequential_generatedRoot_118.w_3_0 ctx), ((sequential_generatedRoot_118.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_118.w_2_0 ctx) (sequential_generatedRoot_118.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_118.outs ctx

theorem sequential_generatedRoot_118.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_118.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_118.constraints_0 params i_0 (sequential_generatedRoot_118.w_3_0 ctx) (sequential_generatedRoot_118.w_2_0 ctx) (sequential_generatedRoot_118.w_4_0 ctx) (sequential_generatedRoot_118.w_5_0 ctx) (sequential_generatedRoot_118.w_11_0 ctx) outputs) :
    sequential_generatedRoot_118.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_118.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_118.Facts_0 ctx outputs

theorem sequential_generatedRoot_118.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_118.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_118.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_118.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_118.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_120.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_120.Witness

def sequential_generatedRoot_120.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_120.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_120.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_120.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_120.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_120.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_120.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_120.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_120.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_120.w_5_0 ctx) = (sequential_generatedRoot_120.w_6_0 ctx))

def sequential_generatedRoot_120.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_120.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_120.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_120.w_1_0 ctx) + (sequential_generatedRoot_120.w_8_0 ctx))

def sequential_generatedRoot_120.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_120.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_120.w_9_0 ctx) = (sequential_generatedRoot_120.w_10_0 ctx))

def sequential_generatedRoot_120.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_120.w_11_0 ctx)

structure sequential_generatedRoot_120.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_120_5 params i_0 i (current, ((sequential_generatedRoot_120.w_3_0 ctx), ((sequential_generatedRoot_120.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_120.w_2_0 ctx) (sequential_generatedRoot_120.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_120.outs ctx

theorem sequential_generatedRoot_120.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_120.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_120.constraints_0 params i_0 (sequential_generatedRoot_120.w_3_0 ctx) (sequential_generatedRoot_120.w_2_0 ctx) (sequential_generatedRoot_120.w_4_0 ctx) (sequential_generatedRoot_120.w_5_0 ctx) (sequential_generatedRoot_120.w_11_0 ctx) outputs) :
    sequential_generatedRoot_120.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_120.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_120.Facts_0 ctx outputs

theorem sequential_generatedRoot_120.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_120.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_120.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_120.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_120.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_122.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_122.Witness

def sequential_generatedRoot_122.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_122.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_122.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_122.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_122.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_122.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_122.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_122.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_122.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_122.w_5_0 ctx) = (sequential_generatedRoot_122.w_6_0 ctx))

def sequential_generatedRoot_122.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_122.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_122.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_122.w_1_0 ctx) + (sequential_generatedRoot_122.w_8_0 ctx))

def sequential_generatedRoot_122.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_122.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_122.w_9_0 ctx) = (sequential_generatedRoot_122.w_10_0 ctx))

def sequential_generatedRoot_122.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_122.w_11_0 ctx)

structure sequential_generatedRoot_122.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_122_5 params i_0 i (current, ((sequential_generatedRoot_122.w_3_0 ctx), ((sequential_generatedRoot_122.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_122.w_2_0 ctx) (sequential_generatedRoot_122.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_122.outs ctx

theorem sequential_generatedRoot_122.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_122.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_122.constraints_0 params i_0 (sequential_generatedRoot_122.w_3_0 ctx) (sequential_generatedRoot_122.w_2_0 ctx) (sequential_generatedRoot_122.w_4_0 ctx) (sequential_generatedRoot_122.w_5_0 ctx) (sequential_generatedRoot_122.w_11_0 ctx) outputs) :
    sequential_generatedRoot_122.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_122.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_122.Facts_0 ctx outputs

theorem sequential_generatedRoot_122.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_122.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_122.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_122.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_122.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_124.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_124.Witness

def sequential_generatedRoot_124.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_124.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_124.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_124.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_124.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_124.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_124.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_124.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_124.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_124.w_5_0 ctx) = (sequential_generatedRoot_124.w_6_0 ctx))

def sequential_generatedRoot_124.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_124.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_124.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_124.w_1_0 ctx) + (sequential_generatedRoot_124.w_8_0 ctx))

def sequential_generatedRoot_124.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_124.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_124.w_9_0 ctx) = (sequential_generatedRoot_124.w_10_0 ctx))

def sequential_generatedRoot_124.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_124.w_11_0 ctx)

structure sequential_generatedRoot_124.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_124_5 params i_0 i (current, ((sequential_generatedRoot_124.w_3_0 ctx), ((sequential_generatedRoot_124.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_124.w_2_0 ctx) (sequential_generatedRoot_124.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_124.outs ctx

theorem sequential_generatedRoot_124.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_124.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_124.constraints_0 params i_0 (sequential_generatedRoot_124.w_3_0 ctx) (sequential_generatedRoot_124.w_2_0 ctx) (sequential_generatedRoot_124.w_4_0 ctx) (sequential_generatedRoot_124.w_5_0 ctx) (sequential_generatedRoot_124.w_11_0 ctx) outputs) :
    sequential_generatedRoot_124.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_124.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_124.Facts_0 ctx outputs

theorem sequential_generatedRoot_124.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_124.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_124.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_124.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_124.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_126.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_126.Witness

def sequential_generatedRoot_126.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_126.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_126.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_126.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_126.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_126.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_126.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_126.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_126.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_126.w_5_0 ctx) = (sequential_generatedRoot_126.w_6_0 ctx))

def sequential_generatedRoot_126.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_126.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_126.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_126.w_1_0 ctx) + (sequential_generatedRoot_126.w_8_0 ctx))

def sequential_generatedRoot_126.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_126.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_126.w_9_0 ctx) = (sequential_generatedRoot_126.w_10_0 ctx))

def sequential_generatedRoot_126.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_126.w_11_0 ctx)

structure sequential_generatedRoot_126.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_126_5 params i_0 i (current, ((sequential_generatedRoot_126.w_3_0 ctx), ((sequential_generatedRoot_126.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_126.w_2_0 ctx) (sequential_generatedRoot_126.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_126.outs ctx

theorem sequential_generatedRoot_126.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_126.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_126.constraints_0 params i_0 (sequential_generatedRoot_126.w_3_0 ctx) (sequential_generatedRoot_126.w_2_0 ctx) (sequential_generatedRoot_126.w_4_0 ctx) (sequential_generatedRoot_126.w_5_0 ctx) (sequential_generatedRoot_126.w_11_0 ctx) outputs) :
    sequential_generatedRoot_126.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_126.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_126.Facts_0 ctx outputs

theorem sequential_generatedRoot_126.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_126.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_126.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_126.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_126.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_128.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_128.Witness

def sequential_generatedRoot_128.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_128.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_128.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_128.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_128.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_128.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_128.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_128.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_128.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_128.w_5_0 ctx) = (sequential_generatedRoot_128.w_6_0 ctx))

def sequential_generatedRoot_128.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_128.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_128.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_128.w_1_0 ctx) + (sequential_generatedRoot_128.w_8_0 ctx))

def sequential_generatedRoot_128.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_128.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_128.w_9_0 ctx) = (sequential_generatedRoot_128.w_10_0 ctx))

def sequential_generatedRoot_128.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_128.w_11_0 ctx)

structure sequential_generatedRoot_128.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_128_5 params i_0 i (current, ((sequential_generatedRoot_128.w_3_0 ctx), ((sequential_generatedRoot_128.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_128.w_2_0 ctx) (sequential_generatedRoot_128.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_128.outs ctx

theorem sequential_generatedRoot_128.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_128.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_128.constraints_0 params i_0 (sequential_generatedRoot_128.w_3_0 ctx) (sequential_generatedRoot_128.w_2_0 ctx) (sequential_generatedRoot_128.w_4_0 ctx) (sequential_generatedRoot_128.w_5_0 ctx) (sequential_generatedRoot_128.w_11_0 ctx) outputs) :
    sequential_generatedRoot_128.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_128.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_128.Facts_0 ctx outputs

theorem sequential_generatedRoot_128.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_128.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_128.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_128.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_128.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_130.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_130.Witness

def sequential_generatedRoot_130.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_130.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_130.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_130.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_130.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_130.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_130.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_130.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_130.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_130.w_5_0 ctx) = (sequential_generatedRoot_130.w_6_0 ctx))

def sequential_generatedRoot_130.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_130.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_130.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_130.w_1_0 ctx) + (sequential_generatedRoot_130.w_8_0 ctx))

def sequential_generatedRoot_130.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_130.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_130.w_9_0 ctx) = (sequential_generatedRoot_130.w_10_0 ctx))

def sequential_generatedRoot_130.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_130.w_11_0 ctx)

structure sequential_generatedRoot_130.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_130_5 params i_0 i (current, ((sequential_generatedRoot_130.w_3_0 ctx), ((sequential_generatedRoot_130.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_130.w_2_0 ctx) (sequential_generatedRoot_130.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_130.outs ctx

theorem sequential_generatedRoot_130.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_130.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_130.constraints_0 params i_0 (sequential_generatedRoot_130.w_3_0 ctx) (sequential_generatedRoot_130.w_2_0 ctx) (sequential_generatedRoot_130.w_4_0 ctx) (sequential_generatedRoot_130.w_5_0 ctx) (sequential_generatedRoot_130.w_11_0 ctx) outputs) :
    sequential_generatedRoot_130.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_130.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_130.Facts_0 ctx outputs

theorem sequential_generatedRoot_130.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_130.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_130.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_130.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_130.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_132.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_132.Witness

def sequential_generatedRoot_132.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_132.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_132.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_132.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_132.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_132.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_132.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_132.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_132.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_132.w_5_0 ctx) = (sequential_generatedRoot_132.w_6_0 ctx))

def sequential_generatedRoot_132.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_132.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_132.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_132.w_1_0 ctx) + (sequential_generatedRoot_132.w_8_0 ctx))

def sequential_generatedRoot_132.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_132.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_132.w_9_0 ctx) = (sequential_generatedRoot_132.w_10_0 ctx))

def sequential_generatedRoot_132.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_132.w_11_0 ctx)

structure sequential_generatedRoot_132.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_132_5 params i_0 i (current, ((sequential_generatedRoot_132.w_3_0 ctx), ((sequential_generatedRoot_132.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_132.w_2_0 ctx) (sequential_generatedRoot_132.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_132.outs ctx

theorem sequential_generatedRoot_132.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_132.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_132.constraints_0 params i_0 (sequential_generatedRoot_132.w_3_0 ctx) (sequential_generatedRoot_132.w_2_0 ctx) (sequential_generatedRoot_132.w_4_0 ctx) (sequential_generatedRoot_132.w_5_0 ctx) (sequential_generatedRoot_132.w_11_0 ctx) outputs) :
    sequential_generatedRoot_132.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_132.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_132.Facts_0 ctx outputs

theorem sequential_generatedRoot_132.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_132.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_132.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_132.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_132.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_134.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_134.Witness

def sequential_generatedRoot_134.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_134.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_134.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_134.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_134.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_134.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_134.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_134.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_134.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_134.w_5_0 ctx) = (sequential_generatedRoot_134.w_6_0 ctx))

def sequential_generatedRoot_134.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_134.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_134.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_134.w_1_0 ctx) + (sequential_generatedRoot_134.w_8_0 ctx))

def sequential_generatedRoot_134.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_134.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_134.w_9_0 ctx) = (sequential_generatedRoot_134.w_10_0 ctx))

def sequential_generatedRoot_134.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_134.w_11_0 ctx)

structure sequential_generatedRoot_134.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_134_5 params i_0 i (current, ((sequential_generatedRoot_134.w_3_0 ctx), ((sequential_generatedRoot_134.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_134.w_2_0 ctx) (sequential_generatedRoot_134.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_134.outs ctx

theorem sequential_generatedRoot_134.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_134.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_134.constraints_0 params i_0 (sequential_generatedRoot_134.w_3_0 ctx) (sequential_generatedRoot_134.w_2_0 ctx) (sequential_generatedRoot_134.w_4_0 ctx) (sequential_generatedRoot_134.w_5_0 ctx) (sequential_generatedRoot_134.w_11_0 ctx) outputs) :
    sequential_generatedRoot_134.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_134.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_134.Facts_0 ctx outputs

theorem sequential_generatedRoot_134.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_134.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_134.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_134.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_134.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_136.Witness

def sequential_generatedRoot_136.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_136.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_136.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_136.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_136.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_136.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_136.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_136.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_136.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_136.w_5_0 ctx) = (sequential_generatedRoot_136.w_6_0 ctx))

def sequential_generatedRoot_136.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_136.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_136.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_136.w_1_0 ctx) + (sequential_generatedRoot_136.w_8_0 ctx))

def sequential_generatedRoot_136.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_136.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_136.w_9_0 ctx) = (sequential_generatedRoot_136.w_10_0 ctx))

def sequential_generatedRoot_136.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_136.w_11_0 ctx)

structure sequential_generatedRoot_136.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_136_5 params i_0 i (current, ((sequential_generatedRoot_136.w_3_0 ctx), ((sequential_generatedRoot_136.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_136.w_2_0 ctx) (sequential_generatedRoot_136.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_136.outs ctx

theorem sequential_generatedRoot_136.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_136.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_136.constraints_0 params i_0 (sequential_generatedRoot_136.w_3_0 ctx) (sequential_generatedRoot_136.w_2_0 ctx) (sequential_generatedRoot_136.w_4_0 ctx) (sequential_generatedRoot_136.w_5_0 ctx) (sequential_generatedRoot_136.w_11_0 ctx) outputs) :
    sequential_generatedRoot_136.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_136.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_136.Facts_0 ctx outputs

theorem sequential_generatedRoot_136.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_136.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_136.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_136.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_136.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_138.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_138.Witness

def sequential_generatedRoot_138.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_138.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_138.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_138.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_138.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_138.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_138.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_138.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_138.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_138.w_5_0 ctx) = (sequential_generatedRoot_138.w_6_0 ctx))

def sequential_generatedRoot_138.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_138.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_138.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_138.w_1_0 ctx) + (sequential_generatedRoot_138.w_8_0 ctx))

def sequential_generatedRoot_138.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_138.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_138.w_9_0 ctx) = (sequential_generatedRoot_138.w_10_0 ctx))

def sequential_generatedRoot_138.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_138.w_11_0 ctx)

structure sequential_generatedRoot_138.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_138_5 params i_0 i (current, ((sequential_generatedRoot_138.w_3_0 ctx), ((sequential_generatedRoot_138.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_138.w_2_0 ctx) (sequential_generatedRoot_138.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_138.outs ctx

theorem sequential_generatedRoot_138.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_138.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_138.constraints_0 params i_0 (sequential_generatedRoot_138.w_3_0 ctx) (sequential_generatedRoot_138.w_2_0 ctx) (sequential_generatedRoot_138.w_4_0 ctx) (sequential_generatedRoot_138.w_5_0 ctx) (sequential_generatedRoot_138.w_11_0 ctx) outputs) :
    sequential_generatedRoot_138.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_138.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_138.Facts_0 ctx outputs

theorem sequential_generatedRoot_138.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_138.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_138.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_138.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_138.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_140.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_140.Witness

def sequential_generatedRoot_140.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_140.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_140.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_140.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_140.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_140.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_140.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_140.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_140.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_140.w_5_0 ctx) = (sequential_generatedRoot_140.w_6_0 ctx))

def sequential_generatedRoot_140.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_140.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_140.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_140.w_1_0 ctx) + (sequential_generatedRoot_140.w_8_0 ctx))

def sequential_generatedRoot_140.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_140.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_140.w_9_0 ctx) = (sequential_generatedRoot_140.w_10_0 ctx))

def sequential_generatedRoot_140.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_140.w_11_0 ctx)

structure sequential_generatedRoot_140.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_140_5 params i_0 i (current, ((sequential_generatedRoot_140.w_3_0 ctx), ((sequential_generatedRoot_140.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_140.w_2_0 ctx) (sequential_generatedRoot_140.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_140.outs ctx

theorem sequential_generatedRoot_140.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_140.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_140.constraints_0 params i_0 (sequential_generatedRoot_140.w_3_0 ctx) (sequential_generatedRoot_140.w_2_0 ctx) (sequential_generatedRoot_140.w_4_0 ctx) (sequential_generatedRoot_140.w_5_0 ctx) (sequential_generatedRoot_140.w_11_0 ctx) outputs) :
    sequential_generatedRoot_140.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_140.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_140.Facts_0 ctx outputs

theorem sequential_generatedRoot_140.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_140.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_140.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_140.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_140.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_142.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_142.Witness

def sequential_generatedRoot_142.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_142.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_142.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_142.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_142.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_142.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_142.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_142.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_142.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_142.w_5_0 ctx) = (sequential_generatedRoot_142.w_6_0 ctx))

def sequential_generatedRoot_142.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_142.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_142.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_142.w_1_0 ctx) + (sequential_generatedRoot_142.w_8_0 ctx))

def sequential_generatedRoot_142.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_142.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_142.w_9_0 ctx) = (sequential_generatedRoot_142.w_10_0 ctx))

def sequential_generatedRoot_142.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_142.w_11_0 ctx)

structure sequential_generatedRoot_142.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_142_5 params i_0 i (current, ((sequential_generatedRoot_142.w_3_0 ctx), ((sequential_generatedRoot_142.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_142.w_2_0 ctx) (sequential_generatedRoot_142.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_142.outs ctx

theorem sequential_generatedRoot_142.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_142.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_142.constraints_0 params i_0 (sequential_generatedRoot_142.w_3_0 ctx) (sequential_generatedRoot_142.w_2_0 ctx) (sequential_generatedRoot_142.w_4_0 ctx) (sequential_generatedRoot_142.w_5_0 ctx) (sequential_generatedRoot_142.w_11_0 ctx) outputs) :
    sequential_generatedRoot_142.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_142.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_142.Facts_0 ctx outputs

theorem sequential_generatedRoot_142.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_142.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_142.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_142.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_142.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_144.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_144.Witness

def sequential_generatedRoot_144.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_144.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_144.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_144.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_144.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_144.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_144.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_144.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_144.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_144.w_5_0 ctx) = (sequential_generatedRoot_144.w_6_0 ctx))

def sequential_generatedRoot_144.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_144.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_144.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_144.w_1_0 ctx) + (sequential_generatedRoot_144.w_8_0 ctx))

def sequential_generatedRoot_144.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_144.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_144.w_9_0 ctx) = (sequential_generatedRoot_144.w_10_0 ctx))

def sequential_generatedRoot_144.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_144.w_11_0 ctx)

structure sequential_generatedRoot_144.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_144_5 params i_0 i (current, ((sequential_generatedRoot_144.w_3_0 ctx), ((sequential_generatedRoot_144.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_144.w_2_0 ctx) (sequential_generatedRoot_144.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_144.outs ctx

theorem sequential_generatedRoot_144.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_144.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_144.constraints_0 params i_0 (sequential_generatedRoot_144.w_3_0 ctx) (sequential_generatedRoot_144.w_2_0 ctx) (sequential_generatedRoot_144.w_4_0 ctx) (sequential_generatedRoot_144.w_5_0 ctx) (sequential_generatedRoot_144.w_11_0 ctx) outputs) :
    sequential_generatedRoot_144.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_144.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_144.Facts_0 ctx outputs

theorem sequential_generatedRoot_144.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_144.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_144.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_144.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_144.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_146.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_146.Witness

def sequential_generatedRoot_146.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_146.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_146.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_146.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_146.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_146.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_146.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_146.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_146.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_146.w_5_0 ctx) = (sequential_generatedRoot_146.w_6_0 ctx))

def sequential_generatedRoot_146.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_146.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_146.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_146.w_1_0 ctx) + (sequential_generatedRoot_146.w_8_0 ctx))

def sequential_generatedRoot_146.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_146.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_146.w_9_0 ctx) = (sequential_generatedRoot_146.w_10_0 ctx))

def sequential_generatedRoot_146.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_146.w_11_0 ctx)

structure sequential_generatedRoot_146.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_146_5 params i_0 i (current, ((sequential_generatedRoot_146.w_3_0 ctx), ((sequential_generatedRoot_146.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_146.w_2_0 ctx) (sequential_generatedRoot_146.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_146.outs ctx

theorem sequential_generatedRoot_146.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_146.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_146.constraints_0 params i_0 (sequential_generatedRoot_146.w_3_0 ctx) (sequential_generatedRoot_146.w_2_0 ctx) (sequential_generatedRoot_146.w_4_0 ctx) (sequential_generatedRoot_146.w_5_0 ctx) (sequential_generatedRoot_146.w_11_0 ctx) outputs) :
    sequential_generatedRoot_146.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_146.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_146.Facts_0 ctx outputs

theorem sequential_generatedRoot_146.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_146.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_146.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_146.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_146.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_148.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_148.Witness

def sequential_generatedRoot_148.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_148.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_148.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_148.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_148.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_148.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_148.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_148.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_148.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_148.w_5_0 ctx) = (sequential_generatedRoot_148.w_6_0 ctx))

def sequential_generatedRoot_148.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_148.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_148.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_148.w_1_0 ctx) + (sequential_generatedRoot_148.w_8_0 ctx))

def sequential_generatedRoot_148.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_148.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_148.w_9_0 ctx) = (sequential_generatedRoot_148.w_10_0 ctx))

def sequential_generatedRoot_148.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_148.w_11_0 ctx)

structure sequential_generatedRoot_148.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_148_5 params i_0 i (current, ((sequential_generatedRoot_148.w_3_0 ctx), ((sequential_generatedRoot_148.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_148.w_2_0 ctx) (sequential_generatedRoot_148.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_148.outs ctx

theorem sequential_generatedRoot_148.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_148.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_148.constraints_0 params i_0 (sequential_generatedRoot_148.w_3_0 ctx) (sequential_generatedRoot_148.w_2_0 ctx) (sequential_generatedRoot_148.w_4_0 ctx) (sequential_generatedRoot_148.w_5_0 ctx) (sequential_generatedRoot_148.w_11_0 ctx) outputs) :
    sequential_generatedRoot_148.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_148.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_148.Facts_0 ctx outputs

theorem sequential_generatedRoot_148.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_148.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_148.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_148.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_148.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_150.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_150.Witness

def sequential_generatedRoot_150.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_150.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_150.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_150.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_150.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_150.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_150.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_150.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_150.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_150.w_5_0 ctx) = (sequential_generatedRoot_150.w_6_0 ctx))

def sequential_generatedRoot_150.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_150.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_150.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_150.w_1_0 ctx) + (sequential_generatedRoot_150.w_8_0 ctx))

def sequential_generatedRoot_150.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_150.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_150.w_9_0 ctx) = (sequential_generatedRoot_150.w_10_0 ctx))

def sequential_generatedRoot_150.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_150.w_11_0 ctx)

structure sequential_generatedRoot_150.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_150_5 params i_0 i (current, ((sequential_generatedRoot_150.w_3_0 ctx), ((sequential_generatedRoot_150.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_150.w_2_0 ctx) (sequential_generatedRoot_150.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_150.outs ctx

theorem sequential_generatedRoot_150.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_150.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_150.constraints_0 params i_0 (sequential_generatedRoot_150.w_3_0 ctx) (sequential_generatedRoot_150.w_2_0 ctx) (sequential_generatedRoot_150.w_4_0 ctx) (sequential_generatedRoot_150.w_5_0 ctx) (sequential_generatedRoot_150.w_11_0 ctx) outputs) :
    sequential_generatedRoot_150.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_150.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_150.Facts_0 ctx outputs

theorem sequential_generatedRoot_150.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_150.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_150.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_150.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_150.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_152.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_152.Witness

def sequential_generatedRoot_152.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_152.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_152.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_152.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_152.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_152.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_152.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_152.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_152.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_152.w_5_0 ctx) = (sequential_generatedRoot_152.w_6_0 ctx))

def sequential_generatedRoot_152.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_152.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_152.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_152.w_1_0 ctx) + (sequential_generatedRoot_152.w_8_0 ctx))

def sequential_generatedRoot_152.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_152.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_152.w_9_0 ctx) = (sequential_generatedRoot_152.w_10_0 ctx))

def sequential_generatedRoot_152.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_152.w_11_0 ctx)

structure sequential_generatedRoot_152.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_152_5 params i_0 i (current, ((sequential_generatedRoot_152.w_3_0 ctx), ((sequential_generatedRoot_152.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_152.w_2_0 ctx) (sequential_generatedRoot_152.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_152.outs ctx

theorem sequential_generatedRoot_152.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_152.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_152.constraints_0 params i_0 (sequential_generatedRoot_152.w_3_0 ctx) (sequential_generatedRoot_152.w_2_0 ctx) (sequential_generatedRoot_152.w_4_0 ctx) (sequential_generatedRoot_152.w_5_0 ctx) (sequential_generatedRoot_152.w_11_0 ctx) outputs) :
    sequential_generatedRoot_152.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_152.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_152.Facts_0 ctx outputs

theorem sequential_generatedRoot_152.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_152.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_152.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_152.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_152.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_154.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_154.Witness

def sequential_generatedRoot_154.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_154.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_154.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_154.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_154.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_154.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_154.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_154.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_154.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_154.w_5_0 ctx) = (sequential_generatedRoot_154.w_6_0 ctx))

def sequential_generatedRoot_154.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_154.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_154.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_154.w_1_0 ctx) + (sequential_generatedRoot_154.w_8_0 ctx))

def sequential_generatedRoot_154.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_154.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_154.w_9_0 ctx) = (sequential_generatedRoot_154.w_10_0 ctx))

def sequential_generatedRoot_154.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_154.w_11_0 ctx)

structure sequential_generatedRoot_154.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_154_5 params i_0 i (current, ((sequential_generatedRoot_154.w_3_0 ctx), ((sequential_generatedRoot_154.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_154.w_2_0 ctx) (sequential_generatedRoot_154.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_154.outs ctx

theorem sequential_generatedRoot_154.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_154.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_154.constraints_0 params i_0 (sequential_generatedRoot_154.w_3_0 ctx) (sequential_generatedRoot_154.w_2_0 ctx) (sequential_generatedRoot_154.w_4_0 ctx) (sequential_generatedRoot_154.w_5_0 ctx) (sequential_generatedRoot_154.w_11_0 ctx) outputs) :
    sequential_generatedRoot_154.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_154.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_154.Facts_0 ctx outputs

theorem sequential_generatedRoot_154.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_154.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_154.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_154.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_154.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_156.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_156.Witness

def sequential_generatedRoot_156.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_156.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_156.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_156.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_156.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_156.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_156.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_156.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_156.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_156.w_5_0 ctx) = (sequential_generatedRoot_156.w_6_0 ctx))

def sequential_generatedRoot_156.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_156.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_156.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_156.w_1_0 ctx) + (sequential_generatedRoot_156.w_8_0 ctx))

def sequential_generatedRoot_156.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_156.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_156.w_9_0 ctx) = (sequential_generatedRoot_156.w_10_0 ctx))

def sequential_generatedRoot_156.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_156.w_11_0 ctx)

structure sequential_generatedRoot_156.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_156_5 params i_0 i (current, ((sequential_generatedRoot_156.w_3_0 ctx), ((sequential_generatedRoot_156.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_156.w_2_0 ctx) (sequential_generatedRoot_156.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_156.outs ctx

theorem sequential_generatedRoot_156.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_156.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_156.constraints_0 params i_0 (sequential_generatedRoot_156.w_3_0 ctx) (sequential_generatedRoot_156.w_2_0 ctx) (sequential_generatedRoot_156.w_4_0 ctx) (sequential_generatedRoot_156.w_5_0 ctx) (sequential_generatedRoot_156.w_11_0 ctx) outputs) :
    sequential_generatedRoot_156.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_156.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_156.Facts_0 ctx outputs

theorem sequential_generatedRoot_156.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_156.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_156.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_156.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_156.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_158.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_158.Witness

def sequential_generatedRoot_158.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_158.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_158.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_158.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_158.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_158.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_158.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_158.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_158.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_158.w_5_0 ctx) = (sequential_generatedRoot_158.w_6_0 ctx))

def sequential_generatedRoot_158.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_158.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_158.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_158.w_1_0 ctx) + (sequential_generatedRoot_158.w_8_0 ctx))

def sequential_generatedRoot_158.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_158.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_158.w_9_0 ctx) = (sequential_generatedRoot_158.w_10_0 ctx))

def sequential_generatedRoot_158.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_158.w_11_0 ctx)

structure sequential_generatedRoot_158.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_158_5 params i_0 i (current, ((sequential_generatedRoot_158.w_3_0 ctx), ((sequential_generatedRoot_158.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_158.w_2_0 ctx) (sequential_generatedRoot_158.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_158.outs ctx

theorem sequential_generatedRoot_158.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_158.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_158.constraints_0 params i_0 (sequential_generatedRoot_158.w_3_0 ctx) (sequential_generatedRoot_158.w_2_0 ctx) (sequential_generatedRoot_158.w_4_0 ctx) (sequential_generatedRoot_158.w_5_0 ctx) (sequential_generatedRoot_158.w_11_0 ctx) outputs) :
    sequential_generatedRoot_158.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_158.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_158.Facts_0 ctx outputs

theorem sequential_generatedRoot_158.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_158.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_158.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_158.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_158.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_160.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_160.Witness

def sequential_generatedRoot_160.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_160.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_160.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_160.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_160.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_160.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_160.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_160.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_160.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_160.w_5_0 ctx) = (sequential_generatedRoot_160.w_6_0 ctx))

def sequential_generatedRoot_160.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_160.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_160.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_160.w_1_0 ctx) + (sequential_generatedRoot_160.w_8_0 ctx))

def sequential_generatedRoot_160.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_160.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_160.w_9_0 ctx) = (sequential_generatedRoot_160.w_10_0 ctx))

def sequential_generatedRoot_160.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_160.w_11_0 ctx)

structure sequential_generatedRoot_160.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_160_5 params i_0 i (current, ((sequential_generatedRoot_160.w_3_0 ctx), ((sequential_generatedRoot_160.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_160.w_2_0 ctx) (sequential_generatedRoot_160.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_160.outs ctx

theorem sequential_generatedRoot_160.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_160.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_160.constraints_0 params i_0 (sequential_generatedRoot_160.w_3_0 ctx) (sequential_generatedRoot_160.w_2_0 ctx) (sequential_generatedRoot_160.w_4_0 ctx) (sequential_generatedRoot_160.w_5_0 ctx) (sequential_generatedRoot_160.w_11_0 ctx) outputs) :
    sequential_generatedRoot_160.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_160.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_160.Facts_0 ctx outputs

theorem sequential_generatedRoot_160.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_160.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_160.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_160.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_160.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_162.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_162.Witness

def sequential_generatedRoot_162.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_162.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_162.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_162.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_162.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_162.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_162.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_162.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_162.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_162.w_5_0 ctx) = (sequential_generatedRoot_162.w_6_0 ctx))

def sequential_generatedRoot_162.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_162.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_162.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_162.w_1_0 ctx) + (sequential_generatedRoot_162.w_8_0 ctx))

def sequential_generatedRoot_162.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_162.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_162.w_9_0 ctx) = (sequential_generatedRoot_162.w_10_0 ctx))

def sequential_generatedRoot_162.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_162.w_11_0 ctx)

structure sequential_generatedRoot_162.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_162_5 params i_0 i (current, ((sequential_generatedRoot_162.w_3_0 ctx), ((sequential_generatedRoot_162.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_162.w_2_0 ctx) (sequential_generatedRoot_162.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_162.outs ctx

theorem sequential_generatedRoot_162.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_162.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_162.constraints_0 params i_0 (sequential_generatedRoot_162.w_3_0 ctx) (sequential_generatedRoot_162.w_2_0 ctx) (sequential_generatedRoot_162.w_4_0 ctx) (sequential_generatedRoot_162.w_5_0 ctx) (sequential_generatedRoot_162.w_11_0 ctx) outputs) :
    sequential_generatedRoot_162.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_162.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_162.Facts_0 ctx outputs

theorem sequential_generatedRoot_162.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_162.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_162.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_162.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_162.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_164.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_164.Witness

def sequential_generatedRoot_164.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_164.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_164.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_164.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_164.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_164.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_164.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_164.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_164.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_164.w_5_0 ctx) = (sequential_generatedRoot_164.w_6_0 ctx))

def sequential_generatedRoot_164.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_164.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_164.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_164.w_1_0 ctx) + (sequential_generatedRoot_164.w_8_0 ctx))

def sequential_generatedRoot_164.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_164.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_164.w_9_0 ctx) = (sequential_generatedRoot_164.w_10_0 ctx))

def sequential_generatedRoot_164.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_164.w_11_0 ctx)

structure sequential_generatedRoot_164.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_164_5 params i_0 i (current, ((sequential_generatedRoot_164.w_3_0 ctx), ((sequential_generatedRoot_164.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_164.w_2_0 ctx) (sequential_generatedRoot_164.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_164.outs ctx

theorem sequential_generatedRoot_164.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_164.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_164.constraints_0 params i_0 (sequential_generatedRoot_164.w_3_0 ctx) (sequential_generatedRoot_164.w_2_0 ctx) (sequential_generatedRoot_164.w_4_0 ctx) (sequential_generatedRoot_164.w_5_0 ctx) (sequential_generatedRoot_164.w_11_0 ctx) outputs) :
    sequential_generatedRoot_164.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_164.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_164.Facts_0 ctx outputs

theorem sequential_generatedRoot_164.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_164.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_164.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_164.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_164.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_166.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_166.Witness

def sequential_generatedRoot_166.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_166.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_166.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_166.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_166.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_166.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_166.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_166.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_166.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_166.w_5_0 ctx) = (sequential_generatedRoot_166.w_6_0 ctx))

def sequential_generatedRoot_166.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_166.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_166.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_166.w_1_0 ctx) + (sequential_generatedRoot_166.w_8_0 ctx))

def sequential_generatedRoot_166.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_166.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_166.w_9_0 ctx) = (sequential_generatedRoot_166.w_10_0 ctx))

def sequential_generatedRoot_166.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_166.w_11_0 ctx)

structure sequential_generatedRoot_166.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_166_5 params i_0 i (current, ((sequential_generatedRoot_166.w_3_0 ctx), ((sequential_generatedRoot_166.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_166.w_2_0 ctx) (sequential_generatedRoot_166.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_166.outs ctx

theorem sequential_generatedRoot_166.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_166.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_166.constraints_0 params i_0 (sequential_generatedRoot_166.w_3_0 ctx) (sequential_generatedRoot_166.w_2_0 ctx) (sequential_generatedRoot_166.w_4_0 ctx) (sequential_generatedRoot_166.w_5_0 ctx) (sequential_generatedRoot_166.w_11_0 ctx) outputs) :
    sequential_generatedRoot_166.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_166.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_166.Facts_0 ctx outputs

theorem sequential_generatedRoot_166.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_166.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_166.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_166.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_166.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_168.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_168.Witness

def sequential_generatedRoot_168.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_168.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_168.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_168.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_168.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_168.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_168.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_168.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_168.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_168.w_5_0 ctx) = (sequential_generatedRoot_168.w_6_0 ctx))

def sequential_generatedRoot_168.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_168.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_168.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_168.w_1_0 ctx) + (sequential_generatedRoot_168.w_8_0 ctx))

def sequential_generatedRoot_168.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_168.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_168.w_9_0 ctx) = (sequential_generatedRoot_168.w_10_0 ctx))

def sequential_generatedRoot_168.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_168.w_11_0 ctx)

structure sequential_generatedRoot_168.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_168_5 params i_0 i (current, ((sequential_generatedRoot_168.w_3_0 ctx), ((sequential_generatedRoot_168.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_168.w_2_0 ctx) (sequential_generatedRoot_168.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_168.outs ctx

theorem sequential_generatedRoot_168.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_168.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_168.constraints_0 params i_0 (sequential_generatedRoot_168.w_3_0 ctx) (sequential_generatedRoot_168.w_2_0 ctx) (sequential_generatedRoot_168.w_4_0 ctx) (sequential_generatedRoot_168.w_5_0 ctx) (sequential_generatedRoot_168.w_11_0 ctx) outputs) :
    sequential_generatedRoot_168.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_168.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_168.Facts_0 ctx outputs

theorem sequential_generatedRoot_168.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_168.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_168.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_168.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_168.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_170.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_170.Witness

def sequential_generatedRoot_170.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_170.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_170.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_170.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_170.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_170.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_170.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_170.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_170.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_170.w_5_0 ctx) = (sequential_generatedRoot_170.w_6_0 ctx))

def sequential_generatedRoot_170.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_170.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_170.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_170.w_1_0 ctx) + (sequential_generatedRoot_170.w_8_0 ctx))

def sequential_generatedRoot_170.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_170.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_170.w_9_0 ctx) = (sequential_generatedRoot_170.w_10_0 ctx))

def sequential_generatedRoot_170.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_170.w_11_0 ctx)

structure sequential_generatedRoot_170.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_170_5 params i_0 i (current, ((sequential_generatedRoot_170.w_3_0 ctx), ((sequential_generatedRoot_170.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_170.w_2_0 ctx) (sequential_generatedRoot_170.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_170.outs ctx

theorem sequential_generatedRoot_170.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_170.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_170.constraints_0 params i_0 (sequential_generatedRoot_170.w_3_0 ctx) (sequential_generatedRoot_170.w_2_0 ctx) (sequential_generatedRoot_170.w_4_0 ctx) (sequential_generatedRoot_170.w_5_0 ctx) (sequential_generatedRoot_170.w_11_0 ctx) outputs) :
    sequential_generatedRoot_170.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_170.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_170.Facts_0 ctx outputs

theorem sequential_generatedRoot_170.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_170.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_170.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_170.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_170.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_172.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_172.Witness

def sequential_generatedRoot_172.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_172.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_172.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_172.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_172.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_172.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_172.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_172.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_172.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_172.w_5_0 ctx) = (sequential_generatedRoot_172.w_6_0 ctx))

def sequential_generatedRoot_172.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_172.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_172.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_172.w_1_0 ctx) + (sequential_generatedRoot_172.w_8_0 ctx))

def sequential_generatedRoot_172.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_172.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_172.w_9_0 ctx) = (sequential_generatedRoot_172.w_10_0 ctx))

def sequential_generatedRoot_172.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_172.w_11_0 ctx)

structure sequential_generatedRoot_172.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_172_5 params i_0 i (current, ((sequential_generatedRoot_172.w_3_0 ctx), ((sequential_generatedRoot_172.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_172.w_2_0 ctx) (sequential_generatedRoot_172.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_172.outs ctx

theorem sequential_generatedRoot_172.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_172.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_172.constraints_0 params i_0 (sequential_generatedRoot_172.w_3_0 ctx) (sequential_generatedRoot_172.w_2_0 ctx) (sequential_generatedRoot_172.w_4_0 ctx) (sequential_generatedRoot_172.w_5_0 ctx) (sequential_generatedRoot_172.w_11_0 ctx) outputs) :
    sequential_generatedRoot_172.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_172.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_172.Facts_0 ctx outputs

theorem sequential_generatedRoot_172.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_172.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_172.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_172.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_172.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_174.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_174.Witness

def sequential_generatedRoot_174.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_174.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_174.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_174.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_174.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_174.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_174.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_174.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_174.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_174.w_5_0 ctx) = (sequential_generatedRoot_174.w_6_0 ctx))

def sequential_generatedRoot_174.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_174.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_174.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_174.w_1_0 ctx) + (sequential_generatedRoot_174.w_8_0 ctx))

def sequential_generatedRoot_174.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_174.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_174.w_9_0 ctx) = (sequential_generatedRoot_174.w_10_0 ctx))

def sequential_generatedRoot_174.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_174.w_11_0 ctx)

structure sequential_generatedRoot_174.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_174_5 params i_0 i (current, ((sequential_generatedRoot_174.w_3_0 ctx), ((sequential_generatedRoot_174.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_174.w_2_0 ctx) (sequential_generatedRoot_174.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_174.outs ctx

theorem sequential_generatedRoot_174.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_174.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_174.constraints_0 params i_0 (sequential_generatedRoot_174.w_3_0 ctx) (sequential_generatedRoot_174.w_2_0 ctx) (sequential_generatedRoot_174.w_4_0 ctx) (sequential_generatedRoot_174.w_5_0 ctx) (sequential_generatedRoot_174.w_11_0 ctx) outputs) :
    sequential_generatedRoot_174.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_174.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_174.Facts_0 ctx outputs

theorem sequential_generatedRoot_174.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_174.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_174.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_174.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_174.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_176.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_176.Witness

def sequential_generatedRoot_176.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_176.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_176.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_176.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_176.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_176.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_176.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_176.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_176.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_176.w_5_0 ctx) = (sequential_generatedRoot_176.w_6_0 ctx))

def sequential_generatedRoot_176.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_176.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_176.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_176.w_1_0 ctx) + (sequential_generatedRoot_176.w_8_0 ctx))

def sequential_generatedRoot_176.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_176.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_176.w_9_0 ctx) = (sequential_generatedRoot_176.w_10_0 ctx))

def sequential_generatedRoot_176.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_176.w_11_0 ctx)

structure sequential_generatedRoot_176.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_176_5 params i_0 i (current, ((sequential_generatedRoot_176.w_3_0 ctx), ((sequential_generatedRoot_176.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_176.w_2_0 ctx) (sequential_generatedRoot_176.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_176.outs ctx

theorem sequential_generatedRoot_176.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_176.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_176.constraints_0 params i_0 (sequential_generatedRoot_176.w_3_0 ctx) (sequential_generatedRoot_176.w_2_0 ctx) (sequential_generatedRoot_176.w_4_0 ctx) (sequential_generatedRoot_176.w_5_0 ctx) (sequential_generatedRoot_176.w_11_0 ctx) outputs) :
    sequential_generatedRoot_176.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_176.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_176.Facts_0 ctx outputs

theorem sequential_generatedRoot_176.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_176.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_176.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_176.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_176.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_178.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_178.Witness

def sequential_generatedRoot_178.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_178.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_178.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_178.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_178.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_178.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_178.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_178.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_178.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_178.w_5_0 ctx) = (sequential_generatedRoot_178.w_6_0 ctx))

def sequential_generatedRoot_178.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_178.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_178.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_178.w_1_0 ctx) + (sequential_generatedRoot_178.w_8_0 ctx))

def sequential_generatedRoot_178.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_178.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_178.w_9_0 ctx) = (sequential_generatedRoot_178.w_10_0 ctx))

def sequential_generatedRoot_178.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_178.w_11_0 ctx)

structure sequential_generatedRoot_178.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_178_5 params i_0 i (current, ((sequential_generatedRoot_178.w_3_0 ctx), ((sequential_generatedRoot_178.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_178.w_2_0 ctx) (sequential_generatedRoot_178.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_178.outs ctx

theorem sequential_generatedRoot_178.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_178.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_178.constraints_0 params i_0 (sequential_generatedRoot_178.w_3_0 ctx) (sequential_generatedRoot_178.w_2_0 ctx) (sequential_generatedRoot_178.w_4_0 ctx) (sequential_generatedRoot_178.w_5_0 ctx) (sequential_generatedRoot_178.w_11_0 ctx) outputs) :
    sequential_generatedRoot_178.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_178.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_178.Facts_0 ctx outputs

theorem sequential_generatedRoot_178.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_178.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_178.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_178.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_178.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_180.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_180.Witness

def sequential_generatedRoot_180.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_180.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_180.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_180.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_180.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_180.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_180.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_180.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_180.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_180.w_5_0 ctx) = (sequential_generatedRoot_180.w_6_0 ctx))

def sequential_generatedRoot_180.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_180.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_180.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_180.w_1_0 ctx) + (sequential_generatedRoot_180.w_8_0 ctx))

def sequential_generatedRoot_180.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_180.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_180.w_9_0 ctx) = (sequential_generatedRoot_180.w_10_0 ctx))

def sequential_generatedRoot_180.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_180.w_11_0 ctx)

structure sequential_generatedRoot_180.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_180_5 params i_0 i (current, ((sequential_generatedRoot_180.w_3_0 ctx), ((sequential_generatedRoot_180.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_180.w_2_0 ctx) (sequential_generatedRoot_180.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_180.outs ctx

theorem sequential_generatedRoot_180.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_180.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_180.constraints_0 params i_0 (sequential_generatedRoot_180.w_3_0 ctx) (sequential_generatedRoot_180.w_2_0 ctx) (sequential_generatedRoot_180.w_4_0 ctx) (sequential_generatedRoot_180.w_5_0 ctx) (sequential_generatedRoot_180.w_11_0 ctx) outputs) :
    sequential_generatedRoot_180.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_180.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_180.Facts_0 ctx outputs

theorem sequential_generatedRoot_180.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_180.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_180.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_180.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_180.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_182.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_182.Witness

def sequential_generatedRoot_182.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_182.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_182.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_182.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_182.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_182.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_182.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_182.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_182.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_182.w_5_0 ctx) = (sequential_generatedRoot_182.w_6_0 ctx))

def sequential_generatedRoot_182.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_182.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_182.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_182.w_1_0 ctx) + (sequential_generatedRoot_182.w_8_0 ctx))

def sequential_generatedRoot_182.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_182.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_182.w_9_0 ctx) = (sequential_generatedRoot_182.w_10_0 ctx))

def sequential_generatedRoot_182.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_182.w_11_0 ctx)

structure sequential_generatedRoot_182.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_182_5 params i_0 i (current, ((sequential_generatedRoot_182.w_3_0 ctx), ((sequential_generatedRoot_182.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_182.w_2_0 ctx) (sequential_generatedRoot_182.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_182.outs ctx

theorem sequential_generatedRoot_182.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_182.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_182.constraints_0 params i_0 (sequential_generatedRoot_182.w_3_0 ctx) (sequential_generatedRoot_182.w_2_0 ctx) (sequential_generatedRoot_182.w_4_0 ctx) (sequential_generatedRoot_182.w_5_0 ctx) (sequential_generatedRoot_182.w_11_0 ctx) outputs) :
    sequential_generatedRoot_182.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_182.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_182.Facts_0 ctx outputs

theorem sequential_generatedRoot_182.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_182.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_182.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_182.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_182.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_184.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_184.Witness

def sequential_generatedRoot_184.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_184.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_184.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_184.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_184.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_184.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_184.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_184.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_184.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_184.w_5_0 ctx) = (sequential_generatedRoot_184.w_6_0 ctx))

def sequential_generatedRoot_184.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_184.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_184.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_184.w_1_0 ctx) + (sequential_generatedRoot_184.w_8_0 ctx))

def sequential_generatedRoot_184.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_184.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_184.w_9_0 ctx) = (sequential_generatedRoot_184.w_10_0 ctx))

def sequential_generatedRoot_184.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_184.w_11_0 ctx)

structure sequential_generatedRoot_184.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_184_5 params i_0 i (current, ((sequential_generatedRoot_184.w_3_0 ctx), ((sequential_generatedRoot_184.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_184.w_2_0 ctx) (sequential_generatedRoot_184.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_184.outs ctx

theorem sequential_generatedRoot_184.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_184.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_184.constraints_0 params i_0 (sequential_generatedRoot_184.w_3_0 ctx) (sequential_generatedRoot_184.w_2_0 ctx) (sequential_generatedRoot_184.w_4_0 ctx) (sequential_generatedRoot_184.w_5_0 ctx) (sequential_generatedRoot_184.w_11_0 ctx) outputs) :
    sequential_generatedRoot_184.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_184.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_184.Facts_0 ctx outputs

theorem sequential_generatedRoot_184.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_184.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_184.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_184.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_184.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_186.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_186.Witness

def sequential_generatedRoot_186.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_186.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_186.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_186.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_186.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_186.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_186.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_186.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_186.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_186.w_5_0 ctx) = (sequential_generatedRoot_186.w_6_0 ctx))

def sequential_generatedRoot_186.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_186.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_186.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_186.w_1_0 ctx) + (sequential_generatedRoot_186.w_8_0 ctx))

def sequential_generatedRoot_186.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_186.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_186.w_9_0 ctx) = (sequential_generatedRoot_186.w_10_0 ctx))

def sequential_generatedRoot_186.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_186.w_11_0 ctx)

structure sequential_generatedRoot_186.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_186_5 params i_0 i (current, ((sequential_generatedRoot_186.w_3_0 ctx), ((sequential_generatedRoot_186.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_186.w_2_0 ctx) (sequential_generatedRoot_186.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_186.outs ctx

theorem sequential_generatedRoot_186.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_186.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_186.constraints_0 params i_0 (sequential_generatedRoot_186.w_3_0 ctx) (sequential_generatedRoot_186.w_2_0 ctx) (sequential_generatedRoot_186.w_4_0 ctx) (sequential_generatedRoot_186.w_5_0 ctx) (sequential_generatedRoot_186.w_11_0 ctx) outputs) :
    sequential_generatedRoot_186.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_186.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_186.Facts_0 ctx outputs

theorem sequential_generatedRoot_186.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_186.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_186.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_186.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_186.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_188.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_188.Witness

def sequential_generatedRoot_188.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_188.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_188.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_188.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_188.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_188.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_188.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_188.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_188.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_188.w_5_0 ctx) = (sequential_generatedRoot_188.w_6_0 ctx))

def sequential_generatedRoot_188.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_188.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_188.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_188.w_1_0 ctx) + (sequential_generatedRoot_188.w_8_0 ctx))

def sequential_generatedRoot_188.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_188.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_188.w_9_0 ctx) = (sequential_generatedRoot_188.w_10_0 ctx))

def sequential_generatedRoot_188.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_188.w_11_0 ctx)

structure sequential_generatedRoot_188.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_188_5 params i_0 i (current, ((sequential_generatedRoot_188.w_3_0 ctx), ((sequential_generatedRoot_188.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_188.w_2_0 ctx) (sequential_generatedRoot_188.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_188.outs ctx

theorem sequential_generatedRoot_188.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_188.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_188.constraints_0 params i_0 (sequential_generatedRoot_188.w_3_0 ctx) (sequential_generatedRoot_188.w_2_0 ctx) (sequential_generatedRoot_188.w_4_0 ctx) (sequential_generatedRoot_188.w_5_0 ctx) (sequential_generatedRoot_188.w_11_0 ctx) outputs) :
    sequential_generatedRoot_188.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_188.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_188.Facts_0 ctx outputs

theorem sequential_generatedRoot_188.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_188.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_188.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_188.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_188.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_190.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_190.Witness

def sequential_generatedRoot_190.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_190.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_190.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_190.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_190.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_190.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_190.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_190.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_190.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_190.w_5_0 ctx) = (sequential_generatedRoot_190.w_6_0 ctx))

def sequential_generatedRoot_190.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_190.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_190.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_190.w_1_0 ctx) + (sequential_generatedRoot_190.w_8_0 ctx))

def sequential_generatedRoot_190.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_190.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_190.w_9_0 ctx) = (sequential_generatedRoot_190.w_10_0 ctx))

def sequential_generatedRoot_190.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_190.w_11_0 ctx)

structure sequential_generatedRoot_190.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_190_5 params i_0 i (current, ((sequential_generatedRoot_190.w_3_0 ctx), ((sequential_generatedRoot_190.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_190.w_2_0 ctx) (sequential_generatedRoot_190.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_190.outs ctx

theorem sequential_generatedRoot_190.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_190.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_190.constraints_0 params i_0 (sequential_generatedRoot_190.w_3_0 ctx) (sequential_generatedRoot_190.w_2_0 ctx) (sequential_generatedRoot_190.w_4_0 ctx) (sequential_generatedRoot_190.w_5_0 ctx) (sequential_generatedRoot_190.w_11_0 ctx) outputs) :
    sequential_generatedRoot_190.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_190.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_190.Facts_0 ctx outputs

theorem sequential_generatedRoot_190.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_190.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_190.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_190.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_190.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_192.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_192.Witness

def sequential_generatedRoot_192.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_192.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_192.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_192.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_192.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_192.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_192.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_192.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_192.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_192.w_5_0 ctx) = (sequential_generatedRoot_192.w_6_0 ctx))

def sequential_generatedRoot_192.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_192.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_192.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_192.w_1_0 ctx) + (sequential_generatedRoot_192.w_8_0 ctx))

def sequential_generatedRoot_192.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_192.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_192.w_9_0 ctx) = (sequential_generatedRoot_192.w_10_0 ctx))

def sequential_generatedRoot_192.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_192.w_11_0 ctx)

structure sequential_generatedRoot_192.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_192_5 params i_0 i (current, ((sequential_generatedRoot_192.w_3_0 ctx), ((sequential_generatedRoot_192.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_192.w_2_0 ctx) (sequential_generatedRoot_192.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_192.outs ctx

theorem sequential_generatedRoot_192.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_192.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_192.constraints_0 params i_0 (sequential_generatedRoot_192.w_3_0 ctx) (sequential_generatedRoot_192.w_2_0 ctx) (sequential_generatedRoot_192.w_4_0 ctx) (sequential_generatedRoot_192.w_5_0 ctx) (sequential_generatedRoot_192.w_11_0 ctx) outputs) :
    sequential_generatedRoot_192.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_192.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_192.Facts_0 ctx outputs

theorem sequential_generatedRoot_192.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_192.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_192.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_192.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_192.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_194.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_194.Witness

def sequential_generatedRoot_194.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_194.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_194.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_194.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_194.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_194.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_194.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_194.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_194.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_194.w_5_0 ctx) = (sequential_generatedRoot_194.w_6_0 ctx))

def sequential_generatedRoot_194.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_194.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_194.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_194.w_1_0 ctx) + (sequential_generatedRoot_194.w_8_0 ctx))

def sequential_generatedRoot_194.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_194.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_194.w_9_0 ctx) = (sequential_generatedRoot_194.w_10_0 ctx))

def sequential_generatedRoot_194.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_194.w_11_0 ctx)

structure sequential_generatedRoot_194.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_194_5 params i_0 i (current, ((sequential_generatedRoot_194.w_3_0 ctx), ((sequential_generatedRoot_194.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_194.w_2_0 ctx) (sequential_generatedRoot_194.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_194.outs ctx

theorem sequential_generatedRoot_194.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_194.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_194.constraints_0 params i_0 (sequential_generatedRoot_194.w_3_0 ctx) (sequential_generatedRoot_194.w_2_0 ctx) (sequential_generatedRoot_194.w_4_0 ctx) (sequential_generatedRoot_194.w_5_0 ctx) (sequential_generatedRoot_194.w_11_0 ctx) outputs) :
    sequential_generatedRoot_194.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_194.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_194.Facts_0 ctx outputs

theorem sequential_generatedRoot_194.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_194.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_194.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_194.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_194.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_196.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_196.Witness

def sequential_generatedRoot_196.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_196.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_196.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_196.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_196.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_196.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_196.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_196.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_196.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_196.w_5_0 ctx) = (sequential_generatedRoot_196.w_6_0 ctx))

def sequential_generatedRoot_196.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_196.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_196.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_196.w_1_0 ctx) + (sequential_generatedRoot_196.w_8_0 ctx))

def sequential_generatedRoot_196.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_196.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_196.w_9_0 ctx) = (sequential_generatedRoot_196.w_10_0 ctx))

def sequential_generatedRoot_196.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_196.w_11_0 ctx)

structure sequential_generatedRoot_196.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_196_5 params i_0 i (current, ((sequential_generatedRoot_196.w_3_0 ctx), ((sequential_generatedRoot_196.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_196.w_2_0 ctx) (sequential_generatedRoot_196.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_196.outs ctx

theorem sequential_generatedRoot_196.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_196.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_196.constraints_0 params i_0 (sequential_generatedRoot_196.w_3_0 ctx) (sequential_generatedRoot_196.w_2_0 ctx) (sequential_generatedRoot_196.w_4_0 ctx) (sequential_generatedRoot_196.w_5_0 ctx) (sequential_generatedRoot_196.w_11_0 ctx) outputs) :
    sequential_generatedRoot_196.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_196.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_196.Facts_0 ctx outputs

theorem sequential_generatedRoot_196.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_196.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_196.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_196.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_196.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_198.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_198.Witness

def sequential_generatedRoot_198.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_198.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_198.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_198.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_198.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_198.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_198.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_198.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_198.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_198.w_5_0 ctx) = (sequential_generatedRoot_198.w_6_0 ctx))

def sequential_generatedRoot_198.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_198.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_198.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_198.w_1_0 ctx) + (sequential_generatedRoot_198.w_8_0 ctx))

def sequential_generatedRoot_198.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_198.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_198.w_9_0 ctx) = (sequential_generatedRoot_198.w_10_0 ctx))

def sequential_generatedRoot_198.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_198.w_11_0 ctx)

structure sequential_generatedRoot_198.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_198_5 params i_0 i (current, ((sequential_generatedRoot_198.w_3_0 ctx), ((sequential_generatedRoot_198.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_198.w_2_0 ctx) (sequential_generatedRoot_198.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_198.outs ctx

theorem sequential_generatedRoot_198.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_198.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_198.constraints_0 params i_0 (sequential_generatedRoot_198.w_3_0 ctx) (sequential_generatedRoot_198.w_2_0 ctx) (sequential_generatedRoot_198.w_4_0 ctx) (sequential_generatedRoot_198.w_5_0 ctx) (sequential_generatedRoot_198.w_11_0 ctx) outputs) :
    sequential_generatedRoot_198.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_198.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_198.Facts_0 ctx outputs

theorem sequential_generatedRoot_198.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_198.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_198.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_198.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_198.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_200.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_200.Witness

def sequential_generatedRoot_200.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_200.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_200.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_200.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_200.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_200.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_200.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_200.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_200.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_200.w_5_0 ctx) = (sequential_generatedRoot_200.w_6_0 ctx))

def sequential_generatedRoot_200.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_200.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_200.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_200.w_1_0 ctx) + (sequential_generatedRoot_200.w_8_0 ctx))

def sequential_generatedRoot_200.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_200.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_200.w_9_0 ctx) = (sequential_generatedRoot_200.w_10_0 ctx))

def sequential_generatedRoot_200.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_200.w_11_0 ctx)

structure sequential_generatedRoot_200.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_200_5 params i_0 i (current, ((sequential_generatedRoot_200.w_3_0 ctx), ((sequential_generatedRoot_200.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_200.w_2_0 ctx) (sequential_generatedRoot_200.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_200.outs ctx

theorem sequential_generatedRoot_200.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_200.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_200.constraints_0 params i_0 (sequential_generatedRoot_200.w_3_0 ctx) (sequential_generatedRoot_200.w_2_0 ctx) (sequential_generatedRoot_200.w_4_0 ctx) (sequential_generatedRoot_200.w_5_0 ctx) (sequential_generatedRoot_200.w_11_0 ctx) outputs) :
    sequential_generatedRoot_200.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_200.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_200.Facts_0 ctx outputs

theorem sequential_generatedRoot_200.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_200.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_200.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_200.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_200.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_202.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_202.Witness

def sequential_generatedRoot_202.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_202.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_202.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_202.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_202.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_202.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_202.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_202.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_202.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_202.w_5_0 ctx) = (sequential_generatedRoot_202.w_6_0 ctx))

def sequential_generatedRoot_202.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_202.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_202.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_202.w_1_0 ctx) + (sequential_generatedRoot_202.w_8_0 ctx))

def sequential_generatedRoot_202.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_202.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_202.w_9_0 ctx) = (sequential_generatedRoot_202.w_10_0 ctx))

def sequential_generatedRoot_202.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_202.w_11_0 ctx)

structure sequential_generatedRoot_202.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_202_5 params i_0 i (current, ((sequential_generatedRoot_202.w_3_0 ctx), ((sequential_generatedRoot_202.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_202.w_2_0 ctx) (sequential_generatedRoot_202.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_202.outs ctx

theorem sequential_generatedRoot_202.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_202.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_202.constraints_0 params i_0 (sequential_generatedRoot_202.w_3_0 ctx) (sequential_generatedRoot_202.w_2_0 ctx) (sequential_generatedRoot_202.w_4_0 ctx) (sequential_generatedRoot_202.w_5_0 ctx) (sequential_generatedRoot_202.w_11_0 ctx) outputs) :
    sequential_generatedRoot_202.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_202.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_202.Facts_0 ctx outputs

theorem sequential_generatedRoot_202.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_202.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_202.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_202.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_202.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_204.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_204.Witness

def sequential_generatedRoot_204.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_204.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_204.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_204.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_204.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_204.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_204.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_204.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_204.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_204.w_5_0 ctx) = (sequential_generatedRoot_204.w_6_0 ctx))

def sequential_generatedRoot_204.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_204.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_204.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_204.w_1_0 ctx) + (sequential_generatedRoot_204.w_8_0 ctx))

def sequential_generatedRoot_204.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_204.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_204.w_9_0 ctx) = (sequential_generatedRoot_204.w_10_0 ctx))

def sequential_generatedRoot_204.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_204.w_11_0 ctx)

structure sequential_generatedRoot_204.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_204_5 params i_0 i (current, ((sequential_generatedRoot_204.w_3_0 ctx), ((sequential_generatedRoot_204.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_204.w_2_0 ctx) (sequential_generatedRoot_204.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_204.outs ctx

theorem sequential_generatedRoot_204.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_204.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_204.constraints_0 params i_0 (sequential_generatedRoot_204.w_3_0 ctx) (sequential_generatedRoot_204.w_2_0 ctx) (sequential_generatedRoot_204.w_4_0 ctx) (sequential_generatedRoot_204.w_5_0 ctx) (sequential_generatedRoot_204.w_11_0 ctx) outputs) :
    sequential_generatedRoot_204.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_204.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_204.Facts_0 ctx outputs

theorem sequential_generatedRoot_204.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_204.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_204.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_204.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_204.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_206.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_206.Witness

def sequential_generatedRoot_206.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_206.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_206.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_206.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_206.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_206.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_206.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_206.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_206.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_206.w_5_0 ctx) = (sequential_generatedRoot_206.w_6_0 ctx))

def sequential_generatedRoot_206.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_206.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_206.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_206.w_1_0 ctx) + (sequential_generatedRoot_206.w_8_0 ctx))

def sequential_generatedRoot_206.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_206.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_206.w_9_0 ctx) = (sequential_generatedRoot_206.w_10_0 ctx))

def sequential_generatedRoot_206.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_206.w_11_0 ctx)

structure sequential_generatedRoot_206.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_206_5 params i_0 i (current, ((sequential_generatedRoot_206.w_3_0 ctx), ((sequential_generatedRoot_206.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_206.w_2_0 ctx) (sequential_generatedRoot_206.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_206.outs ctx

theorem sequential_generatedRoot_206.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_206.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_206.constraints_0 params i_0 (sequential_generatedRoot_206.w_3_0 ctx) (sequential_generatedRoot_206.w_2_0 ctx) (sequential_generatedRoot_206.w_4_0 ctx) (sequential_generatedRoot_206.w_5_0 ctx) (sequential_generatedRoot_206.w_11_0 ctx) outputs) :
    sequential_generatedRoot_206.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_206.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_206.Facts_0 ctx outputs

theorem sequential_generatedRoot_206.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_206.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_206.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_206.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_206.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_208.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_208.Witness

def sequential_generatedRoot_208.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_208.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_208.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_208.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_208.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_208.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_208.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_208.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_208.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_208.w_5_0 ctx) = (sequential_generatedRoot_208.w_6_0 ctx))

def sequential_generatedRoot_208.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_208.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_208.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_208.w_1_0 ctx) + (sequential_generatedRoot_208.w_8_0 ctx))

def sequential_generatedRoot_208.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_208.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_208.w_9_0 ctx) = (sequential_generatedRoot_208.w_10_0 ctx))

def sequential_generatedRoot_208.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_208.w_11_0 ctx)

structure sequential_generatedRoot_208.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_208_5 params i_0 i (current, ((sequential_generatedRoot_208.w_3_0 ctx), ((sequential_generatedRoot_208.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_208.w_2_0 ctx) (sequential_generatedRoot_208.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_208.outs ctx

theorem sequential_generatedRoot_208.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_208.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_208.constraints_0 params i_0 (sequential_generatedRoot_208.w_3_0 ctx) (sequential_generatedRoot_208.w_2_0 ctx) (sequential_generatedRoot_208.w_4_0 ctx) (sequential_generatedRoot_208.w_5_0 ctx) (sequential_generatedRoot_208.w_11_0 ctx) outputs) :
    sequential_generatedRoot_208.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_208.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_208.Facts_0 ctx outputs

theorem sequential_generatedRoot_208.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_208.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_208.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_208.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_208.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_210.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 800 → Int) × Unit
  witness : sequential_generatedRoot_210.Witness

def sequential_generatedRoot_210.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_210.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Fin 800 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_210.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_210.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_210.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_210.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_210.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  ctx.witness.w_5_0

def sequential_generatedRoot_210.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_210.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_210.w_5_0 ctx) = (sequential_generatedRoot_210.w_6_0 ctx))

def sequential_generatedRoot_210.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_210.w_7_0 ctx) then 1 else 0

def sequential_generatedRoot_210.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_210.w_1_0 ctx) + (sequential_generatedRoot_210.w_8_0 ctx))

def sequential_generatedRoot_210.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_210.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_210.w_9_0 ctx) = (sequential_generatedRoot_210.w_10_0 ctx))

def sequential_generatedRoot_210.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_210.w_11_0 ctx)

structure sequential_generatedRoot_210.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) (outputs : Bool) : Prop where
  t0 : 0 ≤ (5)
  h5_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_210_5 params i_0 i (current, ((sequential_generatedRoot_210.w_3_0 ctx), ((sequential_generatedRoot_210.w_4_0 ctx), ()))) next) (Int.toNat (5)) (sequential_generatedRoot_210.w_2_0 ctx) (sequential_generatedRoot_210.w_5_0 ctx)
  hout : outputs = sequential_generatedRoot_210.outs ctx

theorem sequential_generatedRoot_210.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_210.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_210.constraints_0 params i_0 (sequential_generatedRoot_210.w_3_0 ctx) (sequential_generatedRoot_210.w_2_0 ctx) (sequential_generatedRoot_210.w_4_0 ctx) (sequential_generatedRoot_210.w_5_0 ctx) (sequential_generatedRoot_210.w_11_0 ctx) outputs) :
    sequential_generatedRoot_210.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2⟩

structure sequential_generatedRoot_210.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_210.Facts_0 ctx outputs

theorem sequential_generatedRoot_210.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_210.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_210.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_210.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_210.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_222.Ctx (params : Params) (i_0 : Nat) where
  inputs : Bool × (Fin 8 → Int) × (Fin 40 → Int) × (Fin 5 → Int) × (Fin 40 → Int) × Unit
  witness : sequential_generatedRoot_222.Witness

def sequential_generatedRoot_222.w_0_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Bool :=
  ctx.inputs.1

def sequential_generatedRoot_222.w_2_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Fin 8 → Int :=
  ctx.inputs.2.1

def sequential_generatedRoot_222.w_8_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Fin 40 → Int :=
  ctx.inputs.2.2.1

def sequential_generatedRoot_222.w_9_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Fin 5 → Int :=
  ctx.inputs.2.2.2.1

def sequential_generatedRoot_222.w_13_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Fin 40 → Int :=
  ctx.inputs.2.2.2.2.1

def sequential_generatedRoot_222.w_1_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_222.w_0_0 ctx) then 1 else 0

def sequential_generatedRoot_222.w_3_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  (Int.ofNat i_0)

def sequential_generatedRoot_222.w_4_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ctx.witness.w_4_0

def sequential_generatedRoot_222.w_5_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  1

def sequential_generatedRoot_222.w_6_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_222.w_7_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_222.w_10_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ctx.witness.w_10_0

def sequential_generatedRoot_222.w_11_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_6_0 ctx) + (sequential_generatedRoot_222.w_10_0 ctx))

def sequential_generatedRoot_222.w_12_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  0

def sequential_generatedRoot_222.w_14_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ctx.witness.w_14_0

def sequential_generatedRoot_222.w_15_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_11_0 ctx) + (sequential_generatedRoot_222.w_14_0 ctx))

def sequential_generatedRoot_222.w_16_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  8

def sequential_generatedRoot_222.w_17_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_15_0 ctx) % (sequential_generatedRoot_222.w_16_0 ctx))

def sequential_generatedRoot_222.w_18_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  4

def sequential_generatedRoot_222.w_19_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_17_0 ctx) / (sequential_generatedRoot_222.w_18_0 ctx))

def sequential_generatedRoot_222.w_20_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_5_0 ctx) - (sequential_generatedRoot_222.w_19_0 ctx))

def sequential_generatedRoot_222.w_21_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_222.w_4_0 ctx) = (sequential_generatedRoot_222.w_20_0 ctx))

def sequential_generatedRoot_222.w_22_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  if (sequential_generatedRoot_222.w_21_0 ctx) then 1 else 0

def sequential_generatedRoot_222.w_23_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  ((sequential_generatedRoot_222.w_1_0 ctx) + (sequential_generatedRoot_222.w_22_0 ctx))

def sequential_generatedRoot_222.w_24_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Int :=
  2

def sequential_generatedRoot_222.w_25_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Bool :=
  decide ((sequential_generatedRoot_222.w_23_0 ctx) = (sequential_generatedRoot_222.w_24_0 ctx))

def sequential_generatedRoot_222.outs {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) : Bool :=
  (sequential_generatedRoot_222.w_25_0 ctx)

structure sequential_generatedRoot_222.Facts_0 {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) (outputs : Bool) : Prop where
  h3_0 : 0 ≤ (sequential_generatedRoot_222.w_3_0 ctx)
  h3_0_1 : (sequential_generatedRoot_222.w_3_0 ctx) < 8
  h4_0 : MxxRuntime.familyGetDynamic (sequential_generatedRoot_222.w_2_0 ctx) (sequential_generatedRoot_222.w_3_0 ctx) (sequential_generatedRoot_222.w_4_0 ctx)
  t3 : 0 ≤ (5)
  h10_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_222_10 params i_0 i (current, ((sequential_generatedRoot_222.w_8_0 ctx), ((sequential_generatedRoot_222.w_3_0 ctx), ((sequential_generatedRoot_222.w_9_0 ctx), ())))) next) (Int.toNat (5)) (sequential_generatedRoot_222.w_7_0 ctx) (sequential_generatedRoot_222.w_10_0 ctx)
  t5 : 0 ≤ (5)
  h14_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_222_14 params i_0 i (current, ((sequential_generatedRoot_222.w_13_0 ctx), ((sequential_generatedRoot_222.w_3_0 ctx), ((sequential_generatedRoot_222.w_9_0 ctx), ())))) next) (Int.toNat (5)) (sequential_generatedRoot_222.w_12_0 ctx) (sequential_generatedRoot_222.w_14_0 ctx)
  h16_0 : (sequential_generatedRoot_222.w_16_0 ctx) ≠ 0
  h18_0 : (sequential_generatedRoot_222.w_18_0 ctx) ≠ 0
  hout : outputs = sequential_generatedRoot_222.outs ctx

theorem sequential_generatedRoot_222.facts_0 {params : Params} {i_0 : Nat} {ctx : sequential_generatedRoot_222.Ctx params i_0} {outputs : Bool}
    (h : sequential_generatedRoot_222.constraints_0 params i_0 (sequential_generatedRoot_222.w_2_0 ctx) (sequential_generatedRoot_222.w_8_0 ctx) (sequential_generatedRoot_222.w_9_0 ctx) (sequential_generatedRoot_222.w_13_0 ctx) (sequential_generatedRoot_222.w_3_0 ctx) (sequential_generatedRoot_222.w_4_0 ctx) (sequential_generatedRoot_222.w_7_0 ctx) (sequential_generatedRoot_222.w_10_0 ctx) (sequential_generatedRoot_222.w_12_0 ctx) (sequential_generatedRoot_222.w_14_0 ctx) (sequential_generatedRoot_222.w_16_0 ctx) (sequential_generatedRoot_222.w_18_0 ctx) (sequential_generatedRoot_222.w_25_0 ctx) outputs) :
    sequential_generatedRoot_222.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure sequential_generatedRoot_222.Facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) (outputs : Bool) : Prop where
  c0 : sequential_generatedRoot_222.Facts_0 ctx outputs

theorem sequential_generatedRoot_222.facts {params : Params} {i_0 : Nat} (ctx : sequential_generatedRoot_222.Ctx params i_0) {outputs : Bool}
    (h : sequential_generatedRoot_222.body params i_0 ctx.inputs outputs ctx.witness) :
    sequential_generatedRoot_222.Facts ctx outputs :=
  have r0 := sequential_generatedRoot_222.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (params : Params) where
  inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 40 → Int) × (Fin 40 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × Int × Int × (Fin 8 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_3_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 5 → Int :=
  ctx.inputs.1

def generatedRoot.w_4_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 5 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_5_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 5 → Int :=
  ctx.inputs.2.2.1

def generatedRoot.w_7_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 240 → Int :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_9_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 240 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_11_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 240 → Int :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_13_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 240 → Int :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_15_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 40 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_17_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 40 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_19_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_21_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_23_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_25_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_27_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_29_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_31_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_33_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_35_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_37_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_39_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_41_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_43_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_45_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_47_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_49_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_51_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_53_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_55_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_57_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_59_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_61_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_63_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_65_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_67_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_69_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_71_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_73_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_75_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_77_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_79_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_81_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_83_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_85_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_87_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_89_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_91_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_93_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_95_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_97_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_99_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_101_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_103_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_105_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_107_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_109_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_111_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_113_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_115_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_117_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_119_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_121_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_123_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_125_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_127_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_129_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_131_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_133_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_135_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_137_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_139_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_141_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_143_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_145_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_147_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_149_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_151_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_153_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_155_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_157_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_159_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_161_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_163_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_165_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_167_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_169_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_171_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_173_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_175_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_177_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_179_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_181_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_183_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_185_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_187_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_189_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_191_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_193_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_195_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_197_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_199_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_201_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_203_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_205_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_207_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_209_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 800 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_212_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_214_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_221_0 {params : Params} (ctx : generatedRoot.Ctx params) : Fin 8 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_0_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_1_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  0

def generatedRoot.w_2_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_0_0 ctx) = (generatedRoot.w_1_0 ctx))

def generatedRoot.w_6_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_6_0

def generatedRoot.w_8_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_8_0

def generatedRoot.w_10_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_10_0

def generatedRoot.w_12_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_12_0

def generatedRoot.w_14_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_14_0

def generatedRoot.w_16_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_16_0

def generatedRoot.w_18_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_18_0

def generatedRoot.w_20_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_20_0

def generatedRoot.w_22_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_22_0

def generatedRoot.w_24_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_24_0

def generatedRoot.w_26_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_26_0

def generatedRoot.w_28_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_28_0

def generatedRoot.w_30_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_30_0

def generatedRoot.w_32_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_32_0

def generatedRoot.w_34_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_34_0

def generatedRoot.w_36_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_36_0

def generatedRoot.w_38_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_38_0

def generatedRoot.w_40_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_40_0

def generatedRoot.w_42_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_42_0

def generatedRoot.w_44_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_44_0

def generatedRoot.w_46_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_46_0

def generatedRoot.w_48_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_48_0

def generatedRoot.w_50_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_50_0

def generatedRoot.w_52_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_52_0

def generatedRoot.w_54_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_54_0

def generatedRoot.w_56_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_56_0

def generatedRoot.w_58_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_58_0

def generatedRoot.w_60_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_60_0

def generatedRoot.w_62_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_62_0

def generatedRoot.w_64_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_64_0

def generatedRoot.w_66_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_66_0

def generatedRoot.w_68_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_68_0

def generatedRoot.w_70_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_70_0

def generatedRoot.w_72_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_72_0

def generatedRoot.w_74_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_74_0

def generatedRoot.w_76_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_76_0

def generatedRoot.w_78_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_78_0

def generatedRoot.w_80_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_80_0

def generatedRoot.w_82_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_82_0

def generatedRoot.w_84_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_84_0

def generatedRoot.w_86_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_86_0

def generatedRoot.w_88_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_88_0

def generatedRoot.w_90_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_90_0

def generatedRoot.w_92_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_92_0

def generatedRoot.w_94_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_94_0

def generatedRoot.w_96_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_96_0

def generatedRoot.w_98_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_98_0

def generatedRoot.w_100_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_100_0

def generatedRoot.w_102_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_102_0

def generatedRoot.w_104_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_104_0

def generatedRoot.w_106_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_106_0

def generatedRoot.w_108_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_108_0

def generatedRoot.w_110_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_110_0

def generatedRoot.w_112_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_112_0

def generatedRoot.w_114_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_114_0

def generatedRoot.w_116_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_116_0

def generatedRoot.w_118_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_118_0

def generatedRoot.w_120_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_120_0

def generatedRoot.w_122_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_122_0

def generatedRoot.w_124_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_124_0

def generatedRoot.w_126_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_126_0

def generatedRoot.w_128_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_128_0

def generatedRoot.w_130_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_130_0

def generatedRoot.w_132_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_132_0

def generatedRoot.w_134_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_134_0

def generatedRoot.w_136_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_136_0

def generatedRoot.w_138_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_138_0

def generatedRoot.w_140_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_140_0

def generatedRoot.w_142_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_142_0

def generatedRoot.w_144_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_144_0

def generatedRoot.w_146_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_146_0

def generatedRoot.w_148_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_148_0

def generatedRoot.w_150_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_150_0

def generatedRoot.w_152_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_152_0

def generatedRoot.w_154_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_154_0

def generatedRoot.w_156_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_156_0

def generatedRoot.w_158_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_158_0

def generatedRoot.w_160_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_160_0

def generatedRoot.w_162_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_162_0

def generatedRoot.w_164_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_164_0

def generatedRoot.w_166_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_166_0

def generatedRoot.w_168_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_168_0

def generatedRoot.w_170_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_170_0

def generatedRoot.w_172_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_172_0

def generatedRoot.w_174_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_174_0

def generatedRoot.w_176_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_176_0

def generatedRoot.w_178_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_178_0

def generatedRoot.w_180_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_180_0

def generatedRoot.w_182_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_182_0

def generatedRoot.w_184_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_184_0

def generatedRoot.w_186_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_186_0

def generatedRoot.w_188_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_188_0

def generatedRoot.w_190_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_190_0

def generatedRoot.w_192_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_192_0

def generatedRoot.w_194_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_194_0

def generatedRoot.w_196_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_196_0

def generatedRoot.w_198_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_198_0

def generatedRoot.w_200_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_200_0

def generatedRoot.w_202_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_202_0

def generatedRoot.w_204_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_204_0

def generatedRoot.w_206_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_206_0

def generatedRoot.w_208_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_208_0

def generatedRoot.w_210_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_210_0

def generatedRoot.w_211_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_210_0 ctx) then 1 else 0

def generatedRoot.w_213_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  1

def generatedRoot.w_215_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_213_0 ctx) - (generatedRoot.w_214_0 ctx))

def generatedRoot.w_216_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_212_0 ctx) = (generatedRoot.w_215_0 ctx))

def generatedRoot.w_217_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  if (generatedRoot.w_216_0 ctx) then 1 else 0

def generatedRoot.w_218_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  ((generatedRoot.w_211_0 ctx) + (generatedRoot.w_217_0 ctx))

def generatedRoot.w_219_0 {params : Params} (ctx : generatedRoot.Ctx params) : Int :=
  2

def generatedRoot.w_220_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  decide ((generatedRoot.w_218_0 ctx) = (generatedRoot.w_219_0 ctx))

def generatedRoot.w_222_0 {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  ctx.witness.w_222_0

def generatedRoot.outs {params : Params} (ctx : generatedRoot.Ctx params) : Bool :=
  (generatedRoot.w_222_0 ctx)

structure generatedRoot.Facts_0 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t0 : 0 ≤ (5)
  h6_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_5_0 ctx), ())))) next) (Int.toNat (5)) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx)
  t2 : 0 ≤ (48)
  h8_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_8 params i (current, ((generatedRoot.w_7_0 ctx), ())) next) (Int.toNat (48)) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx)
  t4 : 0 ≤ (48)
  h10_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_10 params i (current, ((generatedRoot.w_9_0 ctx), ())) next) (Int.toNat (48)) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx)
  t6 : 0 ≤ (48)
  h12_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_12 params i (current, ((generatedRoot.w_11_0 ctx), ())) next) (Int.toNat (48)) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx)
  t8 : 0 ≤ (48)
  h14_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_14 params i (current, ((generatedRoot.w_13_0 ctx), ())) next) (Int.toNat (48)) (generatedRoot.w_12_0 ctx) (generatedRoot.w_14_0 ctx)
  t10 : 0 ≤ (8)
  h16_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_16 params i (current, ((generatedRoot.w_15_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx)
  t12 : 0 ≤ (8)
  h18_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_18 params i (current, ((generatedRoot.w_17_0 ctx), ())) next) (Int.toNat (8)) (generatedRoot.w_16_0 ctx) (generatedRoot.w_18_0 ctx)
  t14 : 0 ≤ (160)
  h20_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_20 params i (current, ((generatedRoot.w_19_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx)
  t16 : 0 ≤ (160)
  h22_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_22 params i (current, ((generatedRoot.w_21_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx)
  t18 : 0 ≤ (160)
  h24_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_24 params i (current, ((generatedRoot.w_23_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_22_0 ctx) (generatedRoot.w_24_0 ctx)
  t20 : 0 ≤ (160)
  h26_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_26 params i (current, ((generatedRoot.w_25_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx)
  t22 : 0 ≤ (160)
  h28_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_28 params i (current, ((generatedRoot.w_27_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx)
  t24 : 0 ≤ (160)
  h30_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_30 params i (current, ((generatedRoot.w_29_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx)
  t26 : 0 ≤ (160)
  h32_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_32 params i (current, ((generatedRoot.w_31_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx)
  t28 : 0 ≤ (160)
  h34_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_34 params i (current, ((generatedRoot.w_33_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_32_0 ctx) (generatedRoot.w_34_0 ctx)
  t30 : 0 ≤ (160)
  h36_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_36 params i (current, ((generatedRoot.w_35_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx)

theorem generatedRoot.facts_0 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_0 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx ∧ generatedRoot.constraints_1 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t32 : 0 ≤ (160)
  h38_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_38 params i (current, ((generatedRoot.w_37_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx)
  t34 : 0 ≤ (160)
  h40_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_40 params i (current, ((generatedRoot.w_39_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_38_0 ctx) (generatedRoot.w_40_0 ctx)
  t36 : 0 ≤ (160)
  h42_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 params i (current, ((generatedRoot.w_41_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx)
  t38 : 0 ≤ (160)
  h44_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_44 params i (current, ((generatedRoot.w_43_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx)
  t40 : 0 ≤ (160)
  h46_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_46 params i (current, ((generatedRoot.w_45_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_44_0 ctx) (generatedRoot.w_46_0 ctx)
  t42 : 0 ≤ (160)
  h48_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_48 params i (current, ((generatedRoot.w_47_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_46_0 ctx) (generatedRoot.w_48_0 ctx)
  t44 : 0 ≤ (160)
  h50_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_50 params i (current, ((generatedRoot.w_49_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx)
  t46 : 0 ≤ (160)
  h52_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_52 params i (current, ((generatedRoot.w_51_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx)
  t48 : 0 ≤ (160)
  h54_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_54 params i (current, ((generatedRoot.w_53_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_52_0 ctx) (generatedRoot.w_54_0 ctx)
  t50 : 0 ≤ (160)
  h56_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_56 params i (current, ((generatedRoot.w_55_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_54_0 ctx) (generatedRoot.w_56_0 ctx)
  t52 : 0 ≤ (160)
  h58_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_58 params i (current, ((generatedRoot.w_57_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx)
  t54 : 0 ≤ (160)
  h60_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_60 params i (current, ((generatedRoot.w_59_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx)
  t56 : 0 ≤ (160)
  h62_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_62 params i (current, ((generatedRoot.w_61_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx)
  t58 : 0 ≤ (160)
  h64_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_64 params i (current, ((generatedRoot.w_63_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx)
  t60 : 0 ≤ (160)
  h66_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_66 params i (current, ((generatedRoot.w_65_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx)
  t62 : 0 ≤ (160)
  h68_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_68 params i (current, ((generatedRoot.w_67_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx)

theorem generatedRoot.facts_1 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_1 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_57_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx ∧ generatedRoot.constraints_2 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t64 : 0 ≤ (160)
  h70_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_70 params i (current, ((generatedRoot.w_69_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx)
  t66 : 0 ≤ (160)
  h72_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_72 params i (current, ((generatedRoot.w_71_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx)
  t68 : 0 ≤ (160)
  h74_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 params i (current, ((generatedRoot.w_73_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx)
  t70 : 0 ≤ (160)
  h76_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 params i (current, ((generatedRoot.w_75_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx)
  t72 : 0 ≤ (160)
  h78_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, ((generatedRoot.w_77_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx)
  t74 : 0 ≤ (160)
  h80_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 params i (current, ((generatedRoot.w_79_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx)
  t76 : 0 ≤ (160)
  h82_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 params i (current, ((generatedRoot.w_81_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx)
  t78 : 0 ≤ (160)
  h84_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 params i (current, ((generatedRoot.w_83_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx)
  t80 : 0 ≤ (160)
  h86_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 params i (current, ((generatedRoot.w_85_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx)
  t82 : 0 ≤ (160)
  h88_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 params i (current, ((generatedRoot.w_87_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx)
  t84 : 0 ≤ (160)
  h90_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_90 params i (current, ((generatedRoot.w_89_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx)
  t86 : 0 ≤ (160)
  h92_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_92 params i (current, ((generatedRoot.w_91_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx)
  t88 : 0 ≤ (160)
  h94_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_94 params i (current, ((generatedRoot.w_93_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx)
  t90 : 0 ≤ (160)
  h96_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_96 params i (current, ((generatedRoot.w_95_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx)
  t92 : 0 ≤ (160)
  h98_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_98 params i (current, ((generatedRoot.w_97_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx)
  t94 : 0 ≤ (160)
  h100_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_100 params i (current, ((generatedRoot.w_99_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx)

theorem generatedRoot.facts_2 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_2 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_74_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_82_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_2 ctx ∧ generatedRoot.constraints_3 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t96 : 0 ≤ (160)
  h102_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_102 params i (current, ((generatedRoot.w_101_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx)
  t98 : 0 ≤ (160)
  h104_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_104 params i (current, ((generatedRoot.w_103_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx)
  t100 : 0 ≤ (160)
  h106_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_106 params i (current, ((generatedRoot.w_105_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx)
  t102 : 0 ≤ (160)
  h108_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_108 params i (current, ((generatedRoot.w_107_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx)
  t104 : 0 ≤ (160)
  h110_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_110 params i (current, ((generatedRoot.w_109_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx)
  t106 : 0 ≤ (160)
  h112_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_112 params i (current, ((generatedRoot.w_111_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx)
  t108 : 0 ≤ (160)
  h114_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_114 params i (current, ((generatedRoot.w_113_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx)
  t110 : 0 ≤ (160)
  h116_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_116 params i (current, ((generatedRoot.w_115_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx)
  t112 : 0 ≤ (160)
  h118_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_118 params i (current, ((generatedRoot.w_117_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx)
  t114 : 0 ≤ (160)
  h120_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_120 params i (current, ((generatedRoot.w_119_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx)
  t116 : 0 ≤ (160)
  h122_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_122 params i (current, ((generatedRoot.w_121_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx)
  t118 : 0 ≤ (160)
  h124_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_124 params i (current, ((generatedRoot.w_123_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx)
  t120 : 0 ≤ (160)
  h126_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_126 params i (current, ((generatedRoot.w_125_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx)
  t122 : 0 ≤ (160)
  h128_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_128 params i (current, ((generatedRoot.w_127_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx)
  t124 : 0 ≤ (160)
  h130_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_130 params i (current, ((generatedRoot.w_129_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx)
  t126 : 0 ≤ (160)
  h132_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_132 params i (current, ((generatedRoot.w_131_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx)

theorem generatedRoot.facts_3 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_3 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_122_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_3 ctx ∧ generatedRoot.constraints_4 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t128 : 0 ≤ (160)
  h134_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_134 params i (current, ((generatedRoot.w_133_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx)
  t130 : 0 ≤ (160)
  h136_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_136 params i (current, ((generatedRoot.w_135_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx)
  t132 : 0 ≤ (160)
  h138_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_138 params i (current, ((generatedRoot.w_137_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx)
  t134 : 0 ≤ (160)
  h140_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_140 params i (current, ((generatedRoot.w_139_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx)
  t136 : 0 ≤ (160)
  h142_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_142 params i (current, ((generatedRoot.w_141_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx)
  t138 : 0 ≤ (160)
  h144_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_144 params i (current, ((generatedRoot.w_143_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx)
  t140 : 0 ≤ (160)
  h146_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_146 params i (current, ((generatedRoot.w_145_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx)
  t142 : 0 ≤ (160)
  h148_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_148 params i (current, ((generatedRoot.w_147_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx)
  t144 : 0 ≤ (160)
  h150_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_150 params i (current, ((generatedRoot.w_149_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx)
  t146 : 0 ≤ (160)
  h152_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_152 params i (current, ((generatedRoot.w_151_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx)
  t148 : 0 ≤ (160)
  h154_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_154 params i (current, ((generatedRoot.w_153_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx)
  t150 : 0 ≤ (160)
  h156_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_156 params i (current, ((generatedRoot.w_155_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx)
  t152 : 0 ≤ (160)
  h158_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_158 params i (current, ((generatedRoot.w_157_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx)
  t154 : 0 ≤ (160)
  h160_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_160 params i (current, ((generatedRoot.w_159_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx)
  t156 : 0 ≤ (160)
  h162_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_162 params i (current, ((generatedRoot.w_161_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx)
  t158 : 0 ≤ (160)
  h164_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_164 params i (current, ((generatedRoot.w_163_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx)

theorem generatedRoot.facts_4 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_4 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_134_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_146_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_4 ctx ∧ generatedRoot.constraints_5 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5 {params : Params} (ctx : generatedRoot.Ctx params) : Prop where
  t160 : 0 ≤ (160)
  h166_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_166 params i (current, ((generatedRoot.w_165_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx)
  t162 : 0 ≤ (160)
  h168_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_168 params i (current, ((generatedRoot.w_167_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx)
  t164 : 0 ≤ (160)
  h170_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_170 params i (current, ((generatedRoot.w_169_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx)
  t166 : 0 ≤ (160)
  h172_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_172 params i (current, ((generatedRoot.w_171_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx)
  t168 : 0 ≤ (160)
  h174_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_174 params i (current, ((generatedRoot.w_173_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx)
  t170 : 0 ≤ (160)
  h176_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_176 params i (current, ((generatedRoot.w_175_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx)
  t172 : 0 ≤ (160)
  h178_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_178 params i (current, ((generatedRoot.w_177_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx)
  t174 : 0 ≤ (160)
  h180_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_180 params i (current, ((generatedRoot.w_179_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx)
  t176 : 0 ≤ (160)
  h182_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_182 params i (current, ((generatedRoot.w_181_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx)
  t178 : 0 ≤ (160)
  h184_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_184 params i (current, ((generatedRoot.w_183_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx)
  t180 : 0 ≤ (160)
  h186_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_186 params i (current, ((generatedRoot.w_185_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx)
  t182 : 0 ≤ (160)
  h188_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_188 params i (current, ((generatedRoot.w_187_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx)
  t184 : 0 ≤ (160)
  h190_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_190 params i (current, ((generatedRoot.w_189_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx)
  t186 : 0 ≤ (160)
  h192_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_192 params i (current, ((generatedRoot.w_191_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx)
  t188 : 0 ≤ (160)
  h194_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_194 params i (current, ((generatedRoot.w_193_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx)
  t190 : 0 ≤ (160)
  h196_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_196 params i (current, ((generatedRoot.w_195_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx)

theorem generatedRoot.facts_5 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_5 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_193_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_5 ctx ∧ generatedRoot.constraints_6 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_6 {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Bool) : Prop where
  t192 : 0 ≤ (160)
  h198_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_198 params i (current, ((generatedRoot.w_197_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx)
  t194 : 0 ≤ (160)
  h200_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_200 params i (current, ((generatedRoot.w_199_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx)
  t196 : 0 ≤ (160)
  h202_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_202 params i (current, ((generatedRoot.w_201_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx)
  t198 : 0 ≤ (160)
  h204_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_204 params i (current, ((generatedRoot.w_203_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx)
  t200 : 0 ≤ (160)
  h206_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_206 params i (current, ((generatedRoot.w_205_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx)
  t202 : 0 ≤ (160)
  h208_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_208 params i (current, ((generatedRoot.w_207_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx)
  t204 : 0 ≤ (160)
  h210_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_210 params i (current, ((generatedRoot.w_209_0 ctx), ())) next) (Int.toNat (160)) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx)
  t206 : 0 ≤ (8)
  h222_0 : MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_222 params i (current, ((generatedRoot.w_221_0 ctx), ((generatedRoot.w_15_0 ctx), ((generatedRoot.w_3_0 ctx), ((generatedRoot.w_17_0 ctx), ()))))) next) (Int.toNat (8)) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_6 {params : Params} {ctx : generatedRoot.Ctx params} {outputs : Bool}
    (h : generatedRoot.constraints_6 params (generatedRoot.w_3_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_221_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) outputs) :
    generatedRoot.Facts_6 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {params : Params} (ctx : generatedRoot.Ctx params) (outputs : Bool) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx
  c6 : generatedRoot.Facts_6 ctx outputs

theorem generatedRoot.facts {params : Params} (ctx : generatedRoot.Ctx params) {outputs : Bool}
    (h : generatedRoot.body params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) r0.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) r1.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) r2.2
  have r4 := generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) r3.2
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) r4.2
  have r6 := generatedRoot.facts_6 (ctx := ctx) (outputs := outputs) r5.2
  ⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5.1, r6⟩

end W_Requirement_0

end
