import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keys_import_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_6.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_11 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_11.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_14.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_14 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_14.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_19 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_19.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_22.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_22 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_22.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_25.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_25 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_25.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_28.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_28 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_28.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_31.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_31 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_31.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_34.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_34 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_34.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_37.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_37 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_37.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_40.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_40 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_40.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_43 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_43.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_46.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_46 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_46.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_47 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_47.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_48.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_48 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_48.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_49.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_49 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_49.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_50.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_50 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_50.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_51.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_51 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_51.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_52.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_52 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_52.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_53.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_53 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_53.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_54.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_54 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_54.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_55.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_55 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_55.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_56.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_56 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_56.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_57.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_57 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_57.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_58.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_58 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_58.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_59.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_59 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_59.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_60.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_60 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_60.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_61.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_61 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_61.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_62.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_62 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_62.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_63.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_63 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_63.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_64.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_64 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_64.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_65.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_65 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_65.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_66.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_66 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_66.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_67.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_67 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_67.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_68.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_68 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_68.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_69.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_69 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_69.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_70.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_70 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_70.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_71.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_71 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_71.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_72.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_72 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_72.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_73.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_73 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_73.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_74.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_74 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_74.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_75.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_75 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_75.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_76.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_76 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_76.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_77.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_77 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_77.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_78.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_78 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_78.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_79.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_79 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_79.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_80.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_80 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_80.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_81.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_81 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_81.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_82.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_82 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_82.constraints_0 w_0_0 w_1_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Fin 2 → mxxMat0
  w_6_1 : Fin 2 → mxxMat0
  w_6_2 : Fin 2 → mxxMat1
  w_11_0 : Fin 2 → mxxMat0
  w_11_1 : Fin 2 → mxxMat0
  w_11_2 : Fin 2 → mxxMat1
  w_14_0 : Fin 2 → mxxMat0
  w_14_1 : Fin 2 → mxxMat0
  w_14_2 : Fin 2 → mxxMat1
  w_19_0 : Fin 4 → mxxMat0
  w_19_1 : Fin 4 → mxxMat0
  w_19_2 : Fin 4 → mxxMat1
  w_22_0 : Fin 1 → mxxMat0
  w_22_1 : Fin 1 → mxxMat0
  w_25_0 : Fin 1 → mxxMat0
  w_25_1 : Fin 1 → mxxMat0
  w_28_0 : Fin 1 → mxxMat0
  w_28_1 : Fin 1 → mxxMat0
  w_31_0 : Fin 1 → mxxMat0
  w_31_1 : Fin 1 → mxxMat0
  w_34_0 : Fin 1 → mxxMat0
  w_34_1 : Fin 1 → mxxMat0
  w_37_0 : Fin 1 → mxxMat0
  w_37_1 : Fin 1 → mxxMat0
  w_40_0 : Fin 1 → mxxMat0
  w_40_1 : Fin 1 → mxxMat0
  w_43_0 : Fin 8 → mxxMat0
  w_43_1 : Fin 8 → mxxMat0
  w_46_0 : Fin 4 → mxxMat0
  w_46_1 : Fin 4 → mxxMat0
  w_47_0 : Fin 1 → mxxMat0
  w_47_1 : Fin 1 → mxxMat0
  w_48_0 : Fin 1 → mxxMat0
  w_48_1 : Fin 1 → mxxMat0
  w_49_0 : Fin 1 → mxxMat0
  w_49_1 : Fin 1 → mxxMat0
  w_50_0 : Fin 1 → mxxMat0
  w_50_1 : Fin 1 → mxxMat0
  w_51_0 : Fin 1 → mxxMat0
  w_51_1 : Fin 1 → mxxMat0
  w_52_0 : Fin 1 → mxxMat0
  w_52_1 : Fin 1 → mxxMat0
  w_53_0 : Fin 1 → mxxMat0
  w_53_1 : Fin 1 → mxxMat0
  w_54_0 : Fin 8 → mxxMat0
  w_54_1 : Fin 8 → mxxMat0
  w_55_0 : Fin 4 → mxxMat0
  w_55_1 : Fin 4 → mxxMat0
  w_56_0 : Fin 1 → mxxMat0
  w_56_1 : Fin 1 → mxxMat0
  w_57_0 : Fin 1 → mxxMat0
  w_57_1 : Fin 1 → mxxMat0
  w_58_0 : Fin 1 → mxxMat0
  w_58_1 : Fin 1 → mxxMat0
  w_59_0 : Fin 1 → mxxMat0
  w_59_1 : Fin 1 → mxxMat0
  w_60_0 : Fin 1 → mxxMat0
  w_60_1 : Fin 1 → mxxMat0
  w_61_0 : Fin 1 → mxxMat0
  w_61_1 : Fin 1 → mxxMat0
  w_62_0 : Fin 1 → mxxMat0
  w_62_1 : Fin 1 → mxxMat0
  w_63_0 : Fin 8 → mxxMat0
  w_63_1 : Fin 8 → mxxMat0
  w_64_0 : Fin 4 → mxxMat0
  w_64_1 : Fin 4 → mxxMat0
  w_65_0 : Fin 1 → mxxMat0
  w_65_1 : Fin 1 → mxxMat0
  w_66_0 : Fin 1 → mxxMat0
  w_66_1 : Fin 1 → mxxMat0
  w_67_0 : Fin 1 → mxxMat0
  w_67_1 : Fin 1 → mxxMat0
  w_68_0 : Fin 1 → mxxMat0
  w_68_1 : Fin 1 → mxxMat0
  w_69_0 : Fin 1 → mxxMat0
  w_69_1 : Fin 1 → mxxMat0
  w_70_0 : Fin 1 → mxxMat0
  w_70_1 : Fin 1 → mxxMat0
  w_71_0 : Fin 1 → mxxMat0
  w_71_1 : Fin 1 → mxxMat0
  w_72_0 : Fin 8 → mxxMat0
  w_72_1 : Fin 8 → mxxMat0
  w_73_0 : Fin 4 → mxxMat0
  w_73_1 : Fin 4 → mxxMat0
  w_74_0 : Fin 1 → mxxMat0
  w_74_1 : Fin 1 → mxxMat0
  w_75_0 : Fin 1 → mxxMat0
  w_75_1 : Fin 1 → mxxMat0
  w_76_0 : Fin 1 → mxxMat0
  w_76_1 : Fin 1 → mxxMat0
  w_77_0 : Fin 1 → mxxMat0
  w_77_1 : Fin 1 → mxxMat0
  w_78_0 : Fin 1 → mxxMat0
  w_78_1 : Fin 1 → mxxMat0
  w_79_0 : Fin 1 → mxxMat0
  w_79_1 : Fin 1 → mxxMat0
  w_80_0 : Fin 1 → mxxMat0
  w_80_1 : Fin 1 → mxxMat0
  w_81_0 : Fin 8 → mxxMat0
  w_81_1 : Fin 8 → mxxMat0
  w_82_0 : Fin 4 → mxxMat0
  w_82_1 : Fin 4 → mxxMat0

abbrev generatedRoot.constraints_3 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_44_0 : Fin 4 → mxxMat0) (w_45_0 : Fin 4 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat1) (w_11_0 : Fin 2 → mxxMat0) (w_11_1 : Fin 2 → mxxMat0) (w_11_2 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat0) (w_14_1 : Fin 2 → mxxMat0) (w_14_2 : Fin 2 → mxxMat1) (w_19_0 : Fin 4 → mxxMat0) (w_19_1 : Fin 4 → mxxMat0) (w_19_2 : Fin 4 → mxxMat1) (w_22_0 : Fin 1 → mxxMat0) (w_22_1 : Fin 1 → mxxMat0) (w_25_0 : Fin 1 → mxxMat0) (w_25_1 : Fin 1 → mxxMat0) (w_28_0 : Fin 1 → mxxMat0) (w_28_1 : Fin 1 → mxxMat0) (w_31_0 : Fin 1 → mxxMat0) (w_31_1 : Fin 1 → mxxMat0) (w_34_0 : Fin 1 → mxxMat0) (w_34_1 : Fin 1 → mxxMat0) (w_37_0 : Fin 1 → mxxMat0) (w_37_1 : Fin 1 → mxxMat0) (w_40_0 : Fin 1 → mxxMat0) (w_40_1 : Fin 1 → mxxMat0) (w_43_0 : Fin 8 → mxxMat0) (w_43_1 : Fin 8 → mxxMat0) (w_46_0 : Fin 4 → mxxMat0) (w_46_1 : Fin 4 → mxxMat0) (w_47_0 : Fin 1 → mxxMat0) (w_47_1 : Fin 1 → mxxMat0) (w_48_0 : Fin 1 → mxxMat0) (w_48_1 : Fin 1 → mxxMat0) (w_49_0 : Fin 1 → mxxMat0) (w_49_1 : Fin 1 → mxxMat0) (w_50_0 : Fin 1 → mxxMat0) (w_50_1 : Fin 1 → mxxMat0) (w_51_0 : Fin 1 → mxxMat0) (w_51_1 : Fin 1 → mxxMat0) (w_52_0 : Fin 1 → mxxMat0) (w_52_1 : Fin 1 → mxxMat0) (w_53_0 : Fin 1 → mxxMat0) (w_53_1 : Fin 1 → mxxMat0) (w_54_0 : Fin 8 → mxxMat0) (w_54_1 : Fin 8 → mxxMat0) (w_55_0 : Fin 4 → mxxMat0) (w_55_1 : Fin 4 → mxxMat0) (w_56_0 : Fin 1 → mxxMat0) (w_56_1 : Fin 1 → mxxMat0) (w_57_0 : Fin 1 → mxxMat0) (w_57_1 : Fin 1 → mxxMat0) (w_58_0 : Fin 1 → mxxMat0) (w_58_1 : Fin 1 → mxxMat0) (w_59_0 : Fin 1 → mxxMat0) (w_59_1 : Fin 1 → mxxMat0) (w_60_0 : Fin 1 → mxxMat0) (w_60_1 : Fin 1 → mxxMat0) (w_61_0 : Fin 1 → mxxMat0) (w_61_1 : Fin 1 → mxxMat0) (w_62_0 : Fin 1 → mxxMat0) (w_62_1 : Fin 1 → mxxMat0) (w_63_0 : Fin 8 → mxxMat0) (w_63_1 : Fin 8 → mxxMat0) (w_64_0 : Fin 4 → mxxMat0) (w_64_1 : Fin 4 → mxxMat0) (w_65_0 : Fin 1 → mxxMat0) (w_65_1 : Fin 1 → mxxMat0) (w_66_0 : Fin 1 → mxxMat0) (w_66_1 : Fin 1 → mxxMat0) (w_67_0 : Fin 1 → mxxMat0) (w_67_1 : Fin 1 → mxxMat0) (w_68_0 : Fin 1 → mxxMat0) (w_68_1 : Fin 1 → mxxMat0) (w_69_0 : Fin 1 → mxxMat0) (w_69_1 : Fin 1 → mxxMat0) (w_70_0 : Fin 1 → mxxMat0) (w_70_1 : Fin 1 → mxxMat0) (w_71_0 : Fin 1 → mxxMat0) (w_71_1 : Fin 1 → mxxMat0) (w_72_0 : Fin 8 → mxxMat0) (w_72_1 : Fin 8 → mxxMat0) (w_73_0 : Fin 4 → mxxMat0) (w_73_1 : Fin 4 → mxxMat0) (w_74_0 : Fin 1 → mxxMat0) (w_74_1 : Fin 1 → mxxMat0) (w_75_0 : Fin 1 → mxxMat0) (w_75_1 : Fin 1 → mxxMat0) (w_76_0 : Fin 1 → mxxMat0) (w_76_1 : Fin 1 → mxxMat0) (w_77_0 : Fin 1 → mxxMat0) (w_77_1 : Fin 1 → mxxMat0) (w_78_0 : Fin 1 → mxxMat0) (w_78_1 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_79_1 : Fin 1 → mxxMat0) (w_80_0 : Fin 1 → mxxMat0) (w_80_1 : Fin 1 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_81_1 : Fin 8 → mxxMat0) (w_82_0 : Fin 4 → mxxMat0) (w_82_1 : Fin 4 → mxxMat0) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop :=
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_82 tape (path ++ [82, i.val]) params i ((w_44_0 i), ((w_45_0 i), ())) ((w_82_0 i), ((w_82_1 i), ()))) ∧
  outputs = (w_0_0, (w_1_0, (w_6_0, (w_6_1, (w_6_2, (w_7_0, (w_8_0, (w_0_0, (w_1_0, (w_11_0, (w_11_1, (w_11_2, (w_0_0, (w_1_0, (w_14_0, (w_14_1, (w_14_2, (w_19_0, (w_19_1, (w_19_2, (w_22_0, (w_22_1, (w_25_0, (w_25_1, (w_28_0, (w_28_1, (w_31_0, (w_31_1, (w_34_0, (w_34_1, (w_37_0, (w_37_1, (w_40_0, (w_40_1, (w_43_0, (w_43_1, (w_46_0, (w_46_1, (w_47_0, (w_47_1, (w_48_0, (w_48_1, (w_49_0, (w_49_1, (w_50_0, (w_50_1, (w_51_0, (w_51_1, (w_52_0, (w_52_1, (w_53_0, (w_53_1, (w_54_0, (w_54_1, (w_55_0, (w_55_1, (w_56_0, (w_56_1, (w_57_0, (w_57_1, (w_58_0, (w_58_1, (w_59_0, (w_59_1, (w_60_0, (w_60_1, (w_61_0, (w_61_1, (w_62_0, (w_62_1, (w_63_0, (w_63_1, (w_64_0, (w_64_1, (w_65_0, (w_65_1, (w_66_0, (w_66_1, (w_67_0, (w_67_1, (w_68_0, (w_68_1, (w_69_0, (w_69_1, (w_70_0, (w_70_1, (w_71_0, (w_71_1, (w_72_0, (w_72_1, (w_73_0, (w_73_1, (w_74_0, (w_74_1, (w_75_0, (w_75_1, (w_76_0, (w_76_1, (w_77_0, (w_77_1, (w_78_0, (w_78_1, (w_79_0, (w_79_1, (w_80_0, (w_80_1, (w_81_0, (w_81_1, (w_82_0, (w_82_1, ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_2 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_20_0 : Fin 1 → mxxMat0) (w_21_0 : Fin 1 → mxxMat0) (w_23_0 : Fin 1 → mxxMat0) (w_24_0 : Fin 1 → mxxMat0) (w_26_0 : Fin 1 → mxxMat0) (w_27_0 : Fin 1 → mxxMat0) (w_29_0 : Fin 1 → mxxMat0) (w_30_0 : Fin 1 → mxxMat0) (w_32_0 : Fin 1 → mxxMat0) (w_33_0 : Fin 1 → mxxMat0) (w_35_0 : Fin 1 → mxxMat0) (w_36_0 : Fin 1 → mxxMat0) (w_38_0 : Fin 1 → mxxMat0) (w_39_0 : Fin 1 → mxxMat0) (w_41_0 : Fin 8 → mxxMat0) (w_42_0 : Fin 8 → mxxMat0) (w_44_0 : Fin 4 → mxxMat0) (w_45_0 : Fin 4 → mxxMat0) (w_66_0 : Fin 1 → mxxMat0) (w_66_1 : Fin 1 → mxxMat0) (w_67_0 : Fin 1 → mxxMat0) (w_67_1 : Fin 1 → mxxMat0) (w_68_0 : Fin 1 → mxxMat0) (w_68_1 : Fin 1 → mxxMat0) (w_69_0 : Fin 1 → mxxMat0) (w_69_1 : Fin 1 → mxxMat0) (w_70_0 : Fin 1 → mxxMat0) (w_70_1 : Fin 1 → mxxMat0) (w_71_0 : Fin 1 → mxxMat0) (w_71_1 : Fin 1 → mxxMat0) (w_72_0 : Fin 8 → mxxMat0) (w_72_1 : Fin 8 → mxxMat0) (w_73_0 : Fin 4 → mxxMat0) (w_73_1 : Fin 4 → mxxMat0) (w_74_0 : Fin 1 → mxxMat0) (w_74_1 : Fin 1 → mxxMat0) (w_75_0 : Fin 1 → mxxMat0) (w_75_1 : Fin 1 → mxxMat0) (w_76_0 : Fin 1 → mxxMat0) (w_76_1 : Fin 1 → mxxMat0) (w_77_0 : Fin 1 → mxxMat0) (w_77_1 : Fin 1 → mxxMat0) (w_78_0 : Fin 1 → mxxMat0) (w_78_1 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_79_1 : Fin 1 → mxxMat0) (w_80_0 : Fin 1 → mxxMat0) (w_80_1 : Fin 1 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_81_1 : Fin 8 → mxxMat0) : Prop :=
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_66 tape (path ++ [66, i.val]) params i ((w_23_0 i), ((w_24_0 i), ())) ((w_66_0 i), ((w_66_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_67 tape (path ++ [67, i.val]) params i ((w_26_0 i), ((w_27_0 i), ())) ((w_67_0 i), ((w_67_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_68 tape (path ++ [68, i.val]) params i ((w_29_0 i), ((w_30_0 i), ())) ((w_68_0 i), ((w_68_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_69 tape (path ++ [69, i.val]) params i ((w_32_0 i), ((w_33_0 i), ())) ((w_69_0 i), ((w_69_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_70 tape (path ++ [70, i.val]) params i ((w_35_0 i), ((w_36_0 i), ())) ((w_70_0 i), ((w_70_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_71 tape (path ++ [71, i.val]) params i ((w_38_0 i), ((w_39_0 i), ())) ((w_71_0 i), ((w_71_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_72 tape (path ++ [72, i.val]) params i ((w_41_0 i), ((w_42_0 i), ())) ((w_72_0 i), ((w_72_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_73 tape (path ++ [73, i.val]) params i ((w_44_0 i), ((w_45_0 i), ())) ((w_73_0 i), ((w_73_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_74 tape (path ++ [74, i.val]) params i ((w_20_0 i), ((w_21_0 i), ())) ((w_74_0 i), ((w_74_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_75 tape (path ++ [75, i.val]) params i ((w_23_0 i), ((w_24_0 i), ())) ((w_75_0 i), ((w_75_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_76 tape (path ++ [76, i.val]) params i ((w_26_0 i), ((w_27_0 i), ())) ((w_76_0 i), ((w_76_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_77 tape (path ++ [77, i.val]) params i ((w_29_0 i), ((w_30_0 i), ())) ((w_77_0 i), ((w_77_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_78 tape (path ++ [78, i.val]) params i ((w_32_0 i), ((w_33_0 i), ())) ((w_78_0 i), ((w_78_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_79 tape (path ++ [79, i.val]) params i ((w_35_0 i), ((w_36_0 i), ())) ((w_79_0 i), ((w_79_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_80 tape (path ++ [80, i.val]) params i ((w_38_0 i), ((w_39_0 i), ())) ((w_80_0 i), ((w_80_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_81 tape (path ++ [81, i.val]) params i ((w_41_0 i), ((w_42_0 i), ())) ((w_81_0 i), ((w_81_1 i), ())))

abbrev generatedRoot.constraints_1 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_20_0 : Fin 1 → mxxMat0) (w_21_0 : Fin 1 → mxxMat0) (w_23_0 : Fin 1 → mxxMat0) (w_24_0 : Fin 1 → mxxMat0) (w_26_0 : Fin 1 → mxxMat0) (w_27_0 : Fin 1 → mxxMat0) (w_29_0 : Fin 1 → mxxMat0) (w_30_0 : Fin 1 → mxxMat0) (w_32_0 : Fin 1 → mxxMat0) (w_33_0 : Fin 1 → mxxMat0) (w_35_0 : Fin 1 → mxxMat0) (w_36_0 : Fin 1 → mxxMat0) (w_38_0 : Fin 1 → mxxMat0) (w_39_0 : Fin 1 → mxxMat0) (w_41_0 : Fin 8 → mxxMat0) (w_42_0 : Fin 8 → mxxMat0) (w_44_0 : Fin 4 → mxxMat0) (w_45_0 : Fin 4 → mxxMat0) (w_50_0 : Fin 1 → mxxMat0) (w_50_1 : Fin 1 → mxxMat0) (w_51_0 : Fin 1 → mxxMat0) (w_51_1 : Fin 1 → mxxMat0) (w_52_0 : Fin 1 → mxxMat0) (w_52_1 : Fin 1 → mxxMat0) (w_53_0 : Fin 1 → mxxMat0) (w_53_1 : Fin 1 → mxxMat0) (w_54_0 : Fin 8 → mxxMat0) (w_54_1 : Fin 8 → mxxMat0) (w_55_0 : Fin 4 → mxxMat0) (w_55_1 : Fin 4 → mxxMat0) (w_56_0 : Fin 1 → mxxMat0) (w_56_1 : Fin 1 → mxxMat0) (w_57_0 : Fin 1 → mxxMat0) (w_57_1 : Fin 1 → mxxMat0) (w_58_0 : Fin 1 → mxxMat0) (w_58_1 : Fin 1 → mxxMat0) (w_59_0 : Fin 1 → mxxMat0) (w_59_1 : Fin 1 → mxxMat0) (w_60_0 : Fin 1 → mxxMat0) (w_60_1 : Fin 1 → mxxMat0) (w_61_0 : Fin 1 → mxxMat0) (w_61_1 : Fin 1 → mxxMat0) (w_62_0 : Fin 1 → mxxMat0) (w_62_1 : Fin 1 → mxxMat0) (w_63_0 : Fin 8 → mxxMat0) (w_63_1 : Fin 8 → mxxMat0) (w_64_0 : Fin 4 → mxxMat0) (w_64_1 : Fin 4 → mxxMat0) (w_65_0 : Fin 1 → mxxMat0) (w_65_1 : Fin 1 → mxxMat0) : Prop :=
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_50 tape (path ++ [50, i.val]) params i ((w_29_0 i), ((w_30_0 i), ())) ((w_50_0 i), ((w_50_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_51 tape (path ++ [51, i.val]) params i ((w_32_0 i), ((w_33_0 i), ())) ((w_51_0 i), ((w_51_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_52 tape (path ++ [52, i.val]) params i ((w_35_0 i), ((w_36_0 i), ())) ((w_52_0 i), ((w_52_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_53 tape (path ++ [53, i.val]) params i ((w_38_0 i), ((w_39_0 i), ())) ((w_53_0 i), ((w_53_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_54 tape (path ++ [54, i.val]) params i ((w_41_0 i), ((w_42_0 i), ())) ((w_54_0 i), ((w_54_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_55 tape (path ++ [55, i.val]) params i ((w_44_0 i), ((w_45_0 i), ())) ((w_55_0 i), ((w_55_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_56 tape (path ++ [56, i.val]) params i ((w_20_0 i), ((w_21_0 i), ())) ((w_56_0 i), ((w_56_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_57 tape (path ++ [57, i.val]) params i ((w_23_0 i), ((w_24_0 i), ())) ((w_57_0 i), ((w_57_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_58 tape (path ++ [58, i.val]) params i ((w_26_0 i), ((w_27_0 i), ())) ((w_58_0 i), ((w_58_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_59 tape (path ++ [59, i.val]) params i ((w_29_0 i), ((w_30_0 i), ())) ((w_59_0 i), ((w_59_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_60 tape (path ++ [60, i.val]) params i ((w_32_0 i), ((w_33_0 i), ())) ((w_60_0 i), ((w_60_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_61 tape (path ++ [61, i.val]) params i ((w_35_0 i), ((w_36_0 i), ())) ((w_61_0 i), ((w_61_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_62 tape (path ++ [62, i.val]) params i ((w_38_0 i), ((w_39_0 i), ())) ((w_62_0 i), ((w_62_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_63 tape (path ++ [63, i.val]) params i ((w_41_0 i), ((w_42_0 i), ())) ((w_63_0 i), ((w_63_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_64 tape (path ++ [64, i.val]) params i ((w_44_0 i), ((w_45_0 i), ())) ((w_64_0 i), ((w_64_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_65 tape (path ++ [65, i.val]) params i ((w_20_0 i), ((w_21_0 i), ())) ((w_65_0 i), ((w_65_1 i), ())))

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_2_0 : Fin 2 → mxxMat0) (w_3_0 : Fin 2 → mxxMat0) (w_4_0 : Fin 2 → Int) (w_5_0 : Fin 2 → Int) (w_9_0 : Fin 2 → Int) (w_10_0 : Fin 2 → Int) (w_12_0 : Fin 2 → Int) (w_13_0 : Fin 2 → Int) (w_15_0 : Fin 4 → mxxMat0) (w_16_0 : Fin 4 → mxxMat0) (w_17_0 : Fin 4 → Int) (w_18_0 : Fin 4 → Int) (w_20_0 : Fin 1 → mxxMat0) (w_21_0 : Fin 1 → mxxMat0) (w_23_0 : Fin 1 → mxxMat0) (w_24_0 : Fin 1 → mxxMat0) (w_26_0 : Fin 1 → mxxMat0) (w_27_0 : Fin 1 → mxxMat0) (w_29_0 : Fin 1 → mxxMat0) (w_30_0 : Fin 1 → mxxMat0) (w_32_0 : Fin 1 → mxxMat0) (w_33_0 : Fin 1 → mxxMat0) (w_35_0 : Fin 1 → mxxMat0) (w_36_0 : Fin 1 → mxxMat0) (w_38_0 : Fin 1 → mxxMat0) (w_39_0 : Fin 1 → mxxMat0) (w_41_0 : Fin 8 → mxxMat0) (w_42_0 : Fin 8 → mxxMat0) (w_44_0 : Fin 4 → mxxMat0) (w_45_0 : Fin 4 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat1) (w_11_0 : Fin 2 → mxxMat0) (w_11_1 : Fin 2 → mxxMat0) (w_11_2 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat0) (w_14_1 : Fin 2 → mxxMat0) (w_14_2 : Fin 2 → mxxMat1) (w_19_0 : Fin 4 → mxxMat0) (w_19_1 : Fin 4 → mxxMat0) (w_19_2 : Fin 4 → mxxMat1) (w_22_0 : Fin 1 → mxxMat0) (w_22_1 : Fin 1 → mxxMat0) (w_25_0 : Fin 1 → mxxMat0) (w_25_1 : Fin 1 → mxxMat0) (w_28_0 : Fin 1 → mxxMat0) (w_28_1 : Fin 1 → mxxMat0) (w_31_0 : Fin 1 → mxxMat0) (w_31_1 : Fin 1 → mxxMat0) (w_34_0 : Fin 1 → mxxMat0) (w_34_1 : Fin 1 → mxxMat0) (w_37_0 : Fin 1 → mxxMat0) (w_37_1 : Fin 1 → mxxMat0) (w_40_0 : Fin 1 → mxxMat0) (w_40_1 : Fin 1 → mxxMat0) (w_43_0 : Fin 8 → mxxMat0) (w_43_1 : Fin 8 → mxxMat0) (w_46_0 : Fin 4 → mxxMat0) (w_46_1 : Fin 4 → mxxMat0) (w_47_0 : Fin 1 → mxxMat0) (w_47_1 : Fin 1 → mxxMat0) (w_48_0 : Fin 1 → mxxMat0) (w_48_1 : Fin 1 → mxxMat0) (w_49_0 : Fin 1 → mxxMat0) (w_49_1 : Fin 1 → mxxMat0) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i ((w_2_0 i), ((w_3_0 i), ((w_4_0 i), ((w_5_0 i), ())))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ())))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i ((w_2_0 i), ((w_3_0 i), ((w_9_0 i), ((w_10_0 i), ())))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ())))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i ((w_2_0 i), ((w_3_0 i), ((w_12_0 i), ((w_13_0 i), ())))) ((w_14_0 i), ((w_14_1 i), ((w_14_2 i), ())))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_19 tape (path ++ [19, i.val]) params i ((w_15_0 i), ((w_16_0 i), ((w_17_0 i), ((w_18_0 i), ())))) ((w_19_0 i), ((w_19_1 i), ((w_19_2 i), ())))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_22 tape (path ++ [22, i.val]) params i ((w_20_0 i), ((w_21_0 i), ())) ((w_22_0 i), ((w_22_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_25 tape (path ++ [25, i.val]) params i ((w_23_0 i), ((w_24_0 i), ())) ((w_25_0 i), ((w_25_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_28 tape (path ++ [28, i.val]) params i ((w_26_0 i), ((w_27_0 i), ())) ((w_28_0 i), ((w_28_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_31 tape (path ++ [31, i.val]) params i ((w_29_0 i), ((w_30_0 i), ())) ((w_31_0 i), ((w_31_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_34 tape (path ++ [34, i.val]) params i ((w_32_0 i), ((w_33_0 i), ())) ((w_34_0 i), ((w_34_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_37 tape (path ++ [37, i.val]) params i ((w_35_0 i), ((w_36_0 i), ())) ((w_37_0 i), ((w_37_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_40 tape (path ++ [40, i.val]) params i ((w_38_0 i), ((w_39_0 i), ())) ((w_40_0 i), ((w_40_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_43 tape (path ++ [43, i.val]) params i ((w_41_0 i), ((w_42_0 i), ())) ((w_43_0 i), ((w_43_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_46 tape (path ++ [46, i.val]) params i ((w_44_0 i), ((w_45_0 i), ())) ((w_46_0 i), ((w_46_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_47 tape (path ++ [47, i.val]) params i ((w_20_0 i), ((w_21_0 i), ())) ((w_47_0 i), ((w_47_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_48 tape (path ++ [48, i.val]) params i ((w_23_0 i), ((w_24_0 i), ())) ((w_48_0 i), ((w_48_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_49 tape (path ++ [49, i.val]) params i ((w_26_0 i), ((w_27_0 i), ())) ((w_49_0 i), ((w_49_1 i), ())))

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Fin 2 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 2 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 2 → Int := inputs.2.2.2.2.1
  let w_5_0 : Fin 2 → Int := inputs.2.2.2.2.2.1
  let w_7_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_13_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : Fin 4 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 4 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_27_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_33_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_35_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_36_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_42_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_44_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_45_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_14_2 := witness.w_14_2
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_19_2 := witness.w_19_2
  let w_22_0 := witness.w_22_0
  let w_22_1 := witness.w_22_1
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_31_0 := witness.w_31_0
  let w_31_1 := witness.w_31_1
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_37_0 := witness.w_37_0
  let w_37_1 := witness.w_37_1
  let w_40_0 := witness.w_40_0
  let w_40_1 := witness.w_40_1
  let w_43_0 := witness.w_43_0
  let w_43_1 := witness.w_43_1
  let w_46_0 := witness.w_46_0
  let w_46_1 := witness.w_46_1
  let w_47_0 := witness.w_47_0
  let w_47_1 := witness.w_47_1
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_49_0 := witness.w_49_0
  let w_49_1 := witness.w_49_1
  let w_50_0 := witness.w_50_0
  let w_50_1 := witness.w_50_1
  let w_51_0 := witness.w_51_0
  let w_51_1 := witness.w_51_1
  let w_52_0 := witness.w_52_0
  let w_52_1 := witness.w_52_1
  let w_53_0 := witness.w_53_0
  let w_53_1 := witness.w_53_1
  let w_54_0 := witness.w_54_0
  let w_54_1 := witness.w_54_1
  let w_55_0 := witness.w_55_0
  let w_55_1 := witness.w_55_1
  let w_56_0 := witness.w_56_0
  let w_56_1 := witness.w_56_1
  let w_57_0 := witness.w_57_0
  let w_57_1 := witness.w_57_1
  let w_58_0 := witness.w_58_0
  let w_58_1 := witness.w_58_1
  let w_59_0 := witness.w_59_0
  let w_59_1 := witness.w_59_1
  let w_60_0 := witness.w_60_0
  let w_60_1 := witness.w_60_1
  let w_61_0 := witness.w_61_0
  let w_61_1 := witness.w_61_1
  let w_62_0 := witness.w_62_0
  let w_62_1 := witness.w_62_1
  let w_63_0 := witness.w_63_0
  let w_63_1 := witness.w_63_1
  let w_64_0 := witness.w_64_0
  let w_64_1 := witness.w_64_1
  let w_65_0 := witness.w_65_0
  let w_65_1 := witness.w_65_1
  let w_66_0 := witness.w_66_0
  let w_66_1 := witness.w_66_1
  let w_67_0 := witness.w_67_0
  let w_67_1 := witness.w_67_1
  let w_68_0 := witness.w_68_0
  let w_68_1 := witness.w_68_1
  let w_69_0 := witness.w_69_0
  let w_69_1 := witness.w_69_1
  let w_70_0 := witness.w_70_0
  let w_70_1 := witness.w_70_1
  let w_71_0 := witness.w_71_0
  let w_71_1 := witness.w_71_1
  let w_72_0 := witness.w_72_0
  let w_72_1 := witness.w_72_1
  let w_73_0 := witness.w_73_0
  let w_73_1 := witness.w_73_1
  let w_74_0 := witness.w_74_0
  let w_74_1 := witness.w_74_1
  let w_75_0 := witness.w_75_0
  let w_75_1 := witness.w_75_1
  let w_76_0 := witness.w_76_0
  let w_76_1 := witness.w_76_1
  let w_77_0 := witness.w_77_0
  let w_77_1 := witness.w_77_1
  let w_78_0 := witness.w_78_0
  let w_78_1 := witness.w_78_1
  let w_79_0 := witness.w_79_0
  let w_79_1 := witness.w_79_1
  let w_80_0 := witness.w_80_0
  let w_80_1 := witness.w_80_1
  let w_81_0 := witness.w_81_0
  let w_81_1 := witness.w_81_1
  let w_82_0 := witness.w_82_0
  let w_82_1 := witness.w_82_1
  generatedRoot.constraints_0 tape path params w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 ∧
    generatedRoot.constraints_1 tape path params w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 ∧
    generatedRoot.constraints_2 tape path params w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 ∧
    generatedRoot.constraints_3 tape path params w_0_0 w_1_0 w_7_0 w_8_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_keys_import_0_0
