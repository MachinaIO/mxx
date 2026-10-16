import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_keys_import_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_6 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
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

def parallel_generatedRoot_11 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
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

def parallel_generatedRoot_14 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_14.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_17 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_17.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_20 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_20.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_23.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_23 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_23.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_26.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_26 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_26.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_29.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_29 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_29.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_32.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_32 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_32.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_35.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_35 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_35.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_38.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_38 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_38.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_41.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_41 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_41.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_42.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_42 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_42.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_43 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_43.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_44.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_44 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_44.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_45.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_45 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_45.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_46.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_46 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_46.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_47 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_47.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_48.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_48 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_48.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_49.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_49 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_49.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_50.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_50 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_50.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_51.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_51 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_51.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_52.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_52 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_52.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_53.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_53 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_53.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_54.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_54 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_54.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_55.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_55 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_55.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_56.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_56 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_56.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_57.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_57 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_57.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_58.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_58 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_58.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_59.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_59 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_59.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_60.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_60 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_60.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_61.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_61 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_61.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_62.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_62 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_62.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_63.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_63 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_63.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_64.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_64 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_64.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_65.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_65 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_65.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_66.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_66 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_66.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_67.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_67 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_67.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_68.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_68 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_68.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_69.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_69 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_69.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_70.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_70 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_70.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_71.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_71 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_71.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_72.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_72 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_72.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_73.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_73 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_73.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_74.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_74 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_74.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_75.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_75 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_75.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_76.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_76 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_76.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_77.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_77 (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_77.constraints_0 w_0_0 w_1_0 outputs

abbrev generatedRoot.InputType : Type :=
  mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit

abbrev generatedRoot.OutputType : Type :=
  mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit

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

abbrev generatedRoot.constraints_1 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat1) (w_11_0 : Fin 2 → mxxMat0) (w_11_1 : Fin 2 → mxxMat0) (w_11_2 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat0) (w_14_1 : Fin 2 → mxxMat0) (w_14_2 : Fin 2 → mxxMat1) (w_17_0 : Fin 1 → mxxMat0) (w_17_1 : Fin 1 → mxxMat0) (w_20_0 : Fin 1 → mxxMat0) (w_20_1 : Fin 1 → mxxMat0) (w_23_0 : Fin 1 → mxxMat0) (w_23_1 : Fin 1 → mxxMat0) (w_26_0 : Fin 1 → mxxMat0) (w_26_1 : Fin 1 → mxxMat0) (w_29_0 : Fin 1 → mxxMat0) (w_29_1 : Fin 1 → mxxMat0) (w_32_0 : Fin 1 → mxxMat0) (w_32_1 : Fin 1 → mxxMat0) (w_35_0 : Fin 1 → mxxMat0) (w_35_1 : Fin 1 → mxxMat0) (w_38_0 : Fin 8 → mxxMat0) (w_38_1 : Fin 8 → mxxMat0) (w_41_0 : Fin 4 → mxxMat0) (w_41_1 : Fin 4 → mxxMat0) (w_42_0 : Fin 1 → mxxMat0) (w_42_1 : Fin 1 → mxxMat0) (w_43_0 : Fin 1 → mxxMat0) (w_43_1 : Fin 1 → mxxMat0) (w_44_0 : Fin 1 → mxxMat0) (w_44_1 : Fin 1 → mxxMat0) (w_45_0 : Fin 1 → mxxMat0) (w_45_1 : Fin 1 → mxxMat0) (w_46_0 : Fin 1 → mxxMat0) (w_46_1 : Fin 1 → mxxMat0) (w_47_0 : Fin 1 → mxxMat0) (w_47_1 : Fin 1 → mxxMat0) (w_48_0 : Fin 1 → mxxMat0) (w_48_1 : Fin 1 → mxxMat0) (w_49_0 : Fin 8 → mxxMat0) (w_49_1 : Fin 8 → mxxMat0) (w_50_0 : Fin 4 → mxxMat0) (w_50_1 : Fin 4 → mxxMat0) (w_51_0 : Fin 1 → mxxMat0) (w_51_1 : Fin 1 → mxxMat0) (w_52_0 : Fin 1 → mxxMat0) (w_52_1 : Fin 1 → mxxMat0) (w_53_0 : Fin 1 → mxxMat0) (w_53_1 : Fin 1 → mxxMat0) (w_54_0 : Fin 1 → mxxMat0) (w_54_1 : Fin 1 → mxxMat0) (w_55_0 : Fin 1 → mxxMat0) (w_55_1 : Fin 1 → mxxMat0) (w_56_0 : Fin 1 → mxxMat0) (w_56_1 : Fin 1 → mxxMat0) (w_57_0 : Fin 1 → mxxMat0) (w_57_1 : Fin 1 → mxxMat0) (w_58_0 : Fin 8 → mxxMat0) (w_58_1 : Fin 8 → mxxMat0) (w_59_0 : Fin 4 → mxxMat0) (w_59_1 : Fin 4 → mxxMat0) (w_60_0 : Fin 1 → mxxMat0) (w_60_1 : Fin 1 → mxxMat0) (w_61_0 : Fin 1 → mxxMat0) (w_61_1 : Fin 1 → mxxMat0) (w_62_0 : Fin 1 → mxxMat0) (w_62_1 : Fin 1 → mxxMat0) (w_63_0 : Fin 1 → mxxMat0) (w_63_1 : Fin 1 → mxxMat0) (w_64_0 : Fin 1 → mxxMat0) (w_64_1 : Fin 1 → mxxMat0) (w_65_0 : Fin 1 → mxxMat0) (w_65_1 : Fin 1 → mxxMat0) (w_66_0 : Fin 1 → mxxMat0) (w_66_1 : Fin 1 → mxxMat0) (w_67_0 : Fin 8 → mxxMat0) (w_67_1 : Fin 8 → mxxMat0) (w_68_0 : Fin 4 → mxxMat0) (w_68_1 : Fin 4 → mxxMat0) (w_69_0 : Fin 1 → mxxMat0) (w_69_1 : Fin 1 → mxxMat0) (w_70_0 : Fin 1 → mxxMat0) (w_70_1 : Fin 1 → mxxMat0) (w_71_0 : Fin 1 → mxxMat0) (w_71_1 : Fin 1 → mxxMat0) (w_72_0 : Fin 1 → mxxMat0) (w_72_1 : Fin 1 → mxxMat0) (w_73_0 : Fin 1 → mxxMat0) (w_73_1 : Fin 1 → mxxMat0) (w_74_0 : Fin 1 → mxxMat0) (w_74_1 : Fin 1 → mxxMat0) (w_75_0 : Fin 1 → mxxMat0) (w_75_1 : Fin 1 → mxxMat0) (w_76_0 : Fin 8 → mxxMat0) (w_76_1 : Fin 8 → mxxMat0) (w_77_0 : Fin 4 → mxxMat0) (w_77_1 : Fin 4 → mxxMat0) (outputs : (generatedRoot.OutputType)) : Prop :=
  (4) = 4 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (8) = 8 ∧
  (4) = 4 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (8) = 8 ∧
  (4) = 4 ∧
  outputs = (w_0_0, (w_1_0, (w_6_0, (w_6_1, (w_6_2, (w_7_0, (w_8_0, (w_0_0, (w_1_0, (w_11_0, (w_11_1, (w_11_2, (w_0_0, (w_1_0, (w_14_0, (w_14_1, (w_14_2, (w_17_0, (w_17_1, (w_20_0, (w_20_1, (w_23_0, (w_23_1, (w_26_0, (w_26_1, (w_29_0, (w_29_1, (w_32_0, (w_32_1, (w_35_0, (w_35_1, (w_38_0, (w_38_1, (w_41_0, (w_41_1, (w_42_0, (w_42_1, (w_43_0, (w_43_1, (w_44_0, (w_44_1, (w_45_0, (w_45_1, (w_46_0, (w_46_1, (w_47_0, (w_47_1, (w_48_0, (w_48_1, (w_49_0, (w_49_1, (w_50_0, (w_50_1, (w_51_0, (w_51_1, (w_52_0, (w_52_1, (w_53_0, (w_53_1, (w_54_0, (w_54_1, (w_55_0, (w_55_1, (w_56_0, (w_56_1, (w_57_0, (w_57_1, (w_58_0, (w_58_1, (w_59_0, (w_59_1, (w_60_0, (w_60_1, (w_61_0, (w_61_1, (w_62_0, (w_62_1, (w_63_0, (w_63_1, (w_64_0, (w_64_1, (w_65_0, (w_65_1, (w_66_0, (w_66_1, (w_67_0, (w_67_1, (w_68_0, (w_68_1, (w_69_0, (w_69_1, (w_70_0, (w_70_1, (w_71_0, (w_71_1, (w_72_0, (w_72_1, (w_73_0, (w_73_1, (w_74_0, (w_74_1, (w_75_0, (w_75_1, (w_76_0, (w_76_1, (w_77_0, (w_77_1, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_0 (params : Params) (w_2_0 : Fin 2 → mxxMat0) (w_3_0 : Fin 2 → mxxMat0) (w_4_0 : Fin 2 → Int) (w_5_0 : Fin 2 → Int) (w_9_0 : Fin 2 → Int) (w_10_0 : Fin 2 → Int) (w_12_0 : Fin 2 → Int) (w_13_0 : Fin 2 → Int) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat1) (w_11_0 : Fin 2 → mxxMat0) (w_11_1 : Fin 2 → mxxMat0) (w_11_2 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat0) (w_14_1 : Fin 2 → mxxMat0) (w_14_2 : Fin 2 → mxxMat1) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_6 params i ((w_2_0 i), ((w_3_0 i), ((w_4_0 i), ((w_5_0 i), ())))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ())))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_11 params i ((w_2_0 i), ((w_3_0 i), ((w_9_0 i), ((w_10_0 i), ())))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ())))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_14 params i ((w_2_0 i), ((w_3_0 i), ((w_12_0 i), ((w_13_0 i), ())))) ((w_14_0 i), ((w_14_1 i), ((w_14_2 i), ())))) ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (8) = 8 ∧
  (4) = 4 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (8) = 8 ∧
  (4) = 4 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (1) = 1 ∧
  (8) = 8

abbrev generatedRoot.body (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
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
  let w_15_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_27_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_28_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_33_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_34_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_36_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_37_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_40_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_14_2 := witness.w_14_2
  let w_17_0 : Fin 1 → mxxMat0 := w_15_0
  let w_17_1 : Fin 1 → mxxMat0 := w_16_0
  let w_20_0 : Fin 1 → mxxMat0 := w_18_0
  let w_20_1 : Fin 1 → mxxMat0 := w_19_0
  let w_23_0 : Fin 1 → mxxMat0 := w_21_0
  let w_23_1 : Fin 1 → mxxMat0 := w_22_0
  let w_26_0 : Fin 1 → mxxMat0 := w_24_0
  let w_26_1 : Fin 1 → mxxMat0 := w_25_0
  let w_29_0 : Fin 1 → mxxMat0 := w_27_0
  let w_29_1 : Fin 1 → mxxMat0 := w_28_0
  let w_32_0 : Fin 1 → mxxMat0 := w_30_0
  let w_32_1 : Fin 1 → mxxMat0 := w_31_0
  let w_35_0 : Fin 1 → mxxMat0 := w_33_0
  let w_35_1 : Fin 1 → mxxMat0 := w_34_0
  let w_38_0 : Fin 8 → mxxMat0 := w_36_0
  let w_38_1 : Fin 8 → mxxMat0 := w_37_0
  let w_41_0 : Fin 4 → mxxMat0 := w_39_0
  let w_41_1 : Fin 4 → mxxMat0 := w_40_0
  let w_42_0 : Fin 1 → mxxMat0 := w_15_0
  let w_42_1 : Fin 1 → mxxMat0 := w_16_0
  let w_43_0 : Fin 1 → mxxMat0 := w_18_0
  let w_43_1 : Fin 1 → mxxMat0 := w_19_0
  let w_44_0 : Fin 1 → mxxMat0 := w_21_0
  let w_44_1 : Fin 1 → mxxMat0 := w_22_0
  let w_45_0 : Fin 1 → mxxMat0 := w_24_0
  let w_45_1 : Fin 1 → mxxMat0 := w_25_0
  let w_46_0 : Fin 1 → mxxMat0 := w_27_0
  let w_46_1 : Fin 1 → mxxMat0 := w_28_0
  let w_47_0 : Fin 1 → mxxMat0 := w_30_0
  let w_47_1 : Fin 1 → mxxMat0 := w_31_0
  let w_48_0 : Fin 1 → mxxMat0 := w_33_0
  let w_48_1 : Fin 1 → mxxMat0 := w_34_0
  let w_49_0 : Fin 8 → mxxMat0 := w_36_0
  let w_49_1 : Fin 8 → mxxMat0 := w_37_0
  let w_50_0 : Fin 4 → mxxMat0 := w_39_0
  let w_50_1 : Fin 4 → mxxMat0 := w_40_0
  let w_51_0 : Fin 1 → mxxMat0 := w_15_0
  let w_51_1 : Fin 1 → mxxMat0 := w_16_0
  let w_52_0 : Fin 1 → mxxMat0 := w_18_0
  let w_52_1 : Fin 1 → mxxMat0 := w_19_0
  let w_53_0 : Fin 1 → mxxMat0 := w_21_0
  let w_53_1 : Fin 1 → mxxMat0 := w_22_0
  let w_54_0 : Fin 1 → mxxMat0 := w_24_0
  let w_54_1 : Fin 1 → mxxMat0 := w_25_0
  let w_55_0 : Fin 1 → mxxMat0 := w_27_0
  let w_55_1 : Fin 1 → mxxMat0 := w_28_0
  let w_56_0 : Fin 1 → mxxMat0 := w_30_0
  let w_56_1 : Fin 1 → mxxMat0 := w_31_0
  let w_57_0 : Fin 1 → mxxMat0 := w_33_0
  let w_57_1 : Fin 1 → mxxMat0 := w_34_0
  let w_58_0 : Fin 8 → mxxMat0 := w_36_0
  let w_58_1 : Fin 8 → mxxMat0 := w_37_0
  let w_59_0 : Fin 4 → mxxMat0 := w_39_0
  let w_59_1 : Fin 4 → mxxMat0 := w_40_0
  let w_60_0 : Fin 1 → mxxMat0 := w_15_0
  let w_60_1 : Fin 1 → mxxMat0 := w_16_0
  let w_61_0 : Fin 1 → mxxMat0 := w_18_0
  let w_61_1 : Fin 1 → mxxMat0 := w_19_0
  let w_62_0 : Fin 1 → mxxMat0 := w_21_0
  let w_62_1 : Fin 1 → mxxMat0 := w_22_0
  let w_63_0 : Fin 1 → mxxMat0 := w_24_0
  let w_63_1 : Fin 1 → mxxMat0 := w_25_0
  let w_64_0 : Fin 1 → mxxMat0 := w_27_0
  let w_64_1 : Fin 1 → mxxMat0 := w_28_0
  let w_65_0 : Fin 1 → mxxMat0 := w_30_0
  let w_65_1 : Fin 1 → mxxMat0 := w_31_0
  let w_66_0 : Fin 1 → mxxMat0 := w_33_0
  let w_66_1 : Fin 1 → mxxMat0 := w_34_0
  let w_67_0 : Fin 8 → mxxMat0 := w_36_0
  let w_67_1 : Fin 8 → mxxMat0 := w_37_0
  let w_68_0 : Fin 4 → mxxMat0 := w_39_0
  let w_68_1 : Fin 4 → mxxMat0 := w_40_0
  let w_69_0 : Fin 1 → mxxMat0 := w_15_0
  let w_69_1 : Fin 1 → mxxMat0 := w_16_0
  let w_70_0 : Fin 1 → mxxMat0 := w_18_0
  let w_70_1 : Fin 1 → mxxMat0 := w_19_0
  let w_71_0 : Fin 1 → mxxMat0 := w_21_0
  let w_71_1 : Fin 1 → mxxMat0 := w_22_0
  let w_72_0 : Fin 1 → mxxMat0 := w_24_0
  let w_72_1 : Fin 1 → mxxMat0 := w_25_0
  let w_73_0 : Fin 1 → mxxMat0 := w_27_0
  let w_73_1 : Fin 1 → mxxMat0 := w_28_0
  let w_74_0 : Fin 1 → mxxMat0 := w_30_0
  let w_74_1 : Fin 1 → mxxMat0 := w_31_0
  let w_75_0 : Fin 1 → mxxMat0 := w_33_0
  let w_75_1 : Fin 1 → mxxMat0 := w_34_0
  let w_76_0 : Fin 8 → mxxMat0 := w_36_0
  let w_76_1 : Fin 8 → mxxMat0 := w_37_0
  let w_77_0 : Fin 4 → mxxMat0 := w_39_0
  let w_77_1 : Fin 4 → mxxMat0 := w_40_0
  generatedRoot.constraints_0 params w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 ∧
    generatedRoot.constraints_1 w_0_0 w_1_0 w_7_0 w_8_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_17_0 w_17_1 w_20_0 w_20_1 w_23_0 w_23_1 w_26_0 w_26_1 w_29_0 w_29_1 w_32_0 w_32_1 w_35_0 w_35_1 w_38_0 w_38_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 outputs

def generatedRoot (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end W_Stage_keys_import_0_0
