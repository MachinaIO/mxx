import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_setup_public

structure Params where
  unit : Unit

local notation "mxxNat0" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat1" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat2" => 67598865718973452911259282735194974463966008168342023559927392187609292957345997883634735633837025843140326117569327049993577673305918340947654355600136203031252562778990577685529163020701990351591297285624954881
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 501
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 501
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1

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
