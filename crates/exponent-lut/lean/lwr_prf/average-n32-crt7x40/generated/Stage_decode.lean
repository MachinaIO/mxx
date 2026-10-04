import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decode

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1
  w_11_0 : Int

abbrev generatedRoot.constraints_0 (w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_10_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) (w_11_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.modulusReduceRuns w_1_0 w_2_0 ∧
  MxxRuntime.modulusReduceRuns w_5_0 w_6_0 ∧
  (1) = 1 ∧
  MxxRuntime.thresholdDecode (2) (1) 0 w_10_0 w_11_0 ∧
  outputs = w_11_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := inputs.2.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.1
  let w_7_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := inputs.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_2_0 w_3_0
  let w_5_0 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 := (1 : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1)
  let w_6_0 := witness.w_6_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_7_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixAdd w_4_0 w_8_0
  let w_10_0 : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 := MxxRuntime.matrixSub w_0_0 w_9_0
  let w_11_0 := witness.w_11_0
  generatedRoot.constraints_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_decode
