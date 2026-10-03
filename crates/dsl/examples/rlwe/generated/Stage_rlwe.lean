import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_rlwe

structure Params where
  «crt_bits» : Int
  «crt_depth» : Int
  «gadget_base_bits» : Int
  «cutoff» : Int
  «sigma» : Rat

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1
  w_2_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1
  w_4_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1
  w_13_0_decoded : Int

abbrev generatedRoot.constraints_0 (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1) (w_10_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1) (w_13_0_decoded : Int) (w_13_0 : Bool) (outputs : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 × Bool × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 108, 119, 101, 45, 101, 120, 97, 109, 112, 108, 101, 47, 97]) ([]) (w_0_0) w_1_0 ∧
  MxxRuntime.gaussianSample (params.«sigma») (params.«cutoff») w_2_0 ∧
  MxxRuntime.gaussianSample (params.«sigma») (params.«cutoff») w_4_0 ∧
  (2) ≠ 0 ∧
  (1) = 1 ∧
  MxxRuntime.thresholdDecode (2) (1) 0 w_12_0 w_13_0_decoded ∧
  outputs = (w_10_0, (w_13_0, ()))

abbrev generatedRoot.body (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × Int × Unit) (outputs : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 × Bool × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : ByteArray := inputs.1
  let w_7_0 : Int := inputs.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixAdd w_3_0 w_4_0
  let w_6_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := (MxxRuntime.matrixPolynomial [(Int.fdiv (1532495540865518635130821056977027158796330141975560193) (2))] : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1)
  let w_8_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.liftInteger w_7_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixAdd w_5_0 w_9_0
  let w_11_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_2_0
  let w_12_0 : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 := MxxRuntime.matrixSub w_10_0 w_11_0
  let w_13_0_decoded := witness.w_13_0_decoded
  let w_13_0 : Bool := decide (w_13_0_decoded ≠ 0)
  generatedRoot.constraints_0 hashModel params w_0_0 w_1_0 w_2_0 w_4_0 w_10_0 w_12_0 w_13_0_decoded w_13_0 outputs

def generatedRoot (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × Int × Unit) (outputs : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 × Bool × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body hashModel params inputs outputs witness


end Stage_rlwe
