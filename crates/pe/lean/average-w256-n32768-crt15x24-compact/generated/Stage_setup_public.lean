import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_setup_public

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489
local notation "mxxNat1" => 72341562789056444115684979687416806180395197053951206217914228266014512572833794748552875371413453242085860536342419038795864559652387809846630825946845083145910068784615984026523621685186503403424255574409217
local notation "mxxNat2" => 1009041927918404748912066953523025233623263629931203392403660676397045184266783237099217072052453834753

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat0
  w_4_0 : mxxMat0
  w_5_0 : mxxMat1
  w_6_0 : mxxMat0

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_1_0 : mxxMat1) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_9_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  MxxRuntime.centeredRebaseRuns w_1_0 w_2_0 ∧
  MxxRuntime.uniformResidueSampleAt tape (path ++ [4]) w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_4_0 w_5_0 ∧
  MxxRuntime.centeredRebaseRuns w_5_0 w_6_0 ∧
  outputs = w_9_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1) (outputs : mxxMat0) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : mxxMat1 := inputs
  let w_0_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_6_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixSub w_4_0 w_7_0
  let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_3_0 w_8_0
  generatedRoot.constraints_0 tape path w_1_0 w_2_0 w_4_0 w_5_0 w_6_0 w_9_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1) (outputs : mxxMat0) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_setup_public
