import Stage_decode

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Stage_decode

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 501
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 65536 501 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 65536 88 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 501
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 44
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 88

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × ByteArray × mxxMat1 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × (Fin 128 → mxxMat3) × (Fin 37 → mxxMat4) × (Fin 37 → mxxMat4) × (Fin 4736 → mxxMat5) × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat5 × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.2.1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.inputs.2.2.1

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.1

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.1

def generatedRoot.w_26_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.1

def generatedRoot.w_30_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

def generatedRoot.w_34_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_38_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_42_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_46_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_50_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_54_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_58_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_62_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_66_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_70_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_74_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_78_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_82_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_86_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_90_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_94_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_98_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_102_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_106_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_110_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_114_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_118_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_122_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_126_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_130_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_134_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_138_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_142_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_146_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_150_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_154_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_158_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_162_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_166_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_170_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_174_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_178_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_182_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_186_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_190_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_194_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_198_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_202_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_206_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_210_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_214_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_218_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_222_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_226_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_230_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_234_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_238_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_242_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_246_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_250_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_254_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_258_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_262_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_266_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_270_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_274_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_278_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_282_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_286_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_290_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_294_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_298_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_302_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_306_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_310_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_314_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_318_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_322_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_326_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_330_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_334_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_338_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_342_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_346_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_350_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_354_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_358_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_362_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_366_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_370_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_374_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_378_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_382_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_386_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_390_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_394_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_398_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_402_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_406_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_410_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_414_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_418_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_422_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_426_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_430_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_434_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_438_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_442_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_446_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_450_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_454_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_458_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_462_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_466_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_470_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_474_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_478_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_482_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_486_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_490_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_494_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_498_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_502_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_506_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_510_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_514_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_518_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat2 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_537_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 128 → mxxMat3 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_541_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 37 → mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_542_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 37 → mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_546_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4736 → mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_549_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_550_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_552_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_553_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_556_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat5 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_2_0

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_0_0 ctx) (generatedRoot.w_2_0 ctx)

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat6 :=
  ctx.witness.w_6_0

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_7_0

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat6 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat6 :=
  MxxRuntime.matrixSub (generatedRoot.w_6_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx)

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_4_0 ctx) (generatedRoot.w_12_0 ctx)

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_13_0 ctx) (generatedRoot.w_16_0 ctx)

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_18_0 ctx)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_17_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_23_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_22_0 ctx)

def generatedRoot.w_24_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_24_0

def generatedRoot.w_25_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_21_0 ctx) (generatedRoot.w_24_0 ctx)

def generatedRoot.w_27_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_26_0 ctx)

def generatedRoot.w_28_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_28_0

def generatedRoot.w_29_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_25_0 ctx) (generatedRoot.w_28_0 ctx)

def generatedRoot.w_31_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_30_0 ctx)

def generatedRoot.w_32_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_32_0

def generatedRoot.w_33_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_29_0 ctx) (generatedRoot.w_32_0 ctx)

def generatedRoot.w_35_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_34_0 ctx)

def generatedRoot.w_36_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_36_0

def generatedRoot.w_37_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_33_0 ctx) (generatedRoot.w_36_0 ctx)

def generatedRoot.w_39_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_38_0 ctx)

def generatedRoot.w_40_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_40_0

def generatedRoot.w_41_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_37_0 ctx) (generatedRoot.w_40_0 ctx)

def generatedRoot.w_43_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_42_0 ctx)

def generatedRoot.w_44_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_44_0

def generatedRoot.w_45_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_41_0 ctx) (generatedRoot.w_44_0 ctx)

def generatedRoot.w_47_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_46_0 ctx)

def generatedRoot.w_48_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_48_0

def generatedRoot.w_49_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_45_0 ctx) (generatedRoot.w_48_0 ctx)

def generatedRoot.w_51_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_50_0 ctx)

def generatedRoot.w_52_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_52_0

def generatedRoot.w_53_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_49_0 ctx) (generatedRoot.w_52_0 ctx)

def generatedRoot.w_55_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_54_0 ctx)

def generatedRoot.w_56_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_56_0

def generatedRoot.w_57_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_53_0 ctx) (generatedRoot.w_56_0 ctx)

def generatedRoot.w_59_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_58_0 ctx)

def generatedRoot.w_60_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_60_0

def generatedRoot.w_61_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_57_0 ctx) (generatedRoot.w_60_0 ctx)

def generatedRoot.w_63_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_62_0 ctx)

def generatedRoot.w_64_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_64_0

def generatedRoot.w_65_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_61_0 ctx) (generatedRoot.w_64_0 ctx)

def generatedRoot.w_67_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_66_0 ctx)

def generatedRoot.w_68_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_68_0

def generatedRoot.w_69_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_65_0 ctx) (generatedRoot.w_68_0 ctx)

def generatedRoot.w_71_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_70_0 ctx)

def generatedRoot.w_72_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_72_0

def generatedRoot.w_73_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_69_0 ctx) (generatedRoot.w_72_0 ctx)

def generatedRoot.w_75_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_74_0 ctx)

def generatedRoot.w_76_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_76_0

def generatedRoot.w_77_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_73_0 ctx) (generatedRoot.w_76_0 ctx)

def generatedRoot.w_79_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_78_0 ctx)

def generatedRoot.w_80_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_80_0

def generatedRoot.w_81_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_77_0 ctx) (generatedRoot.w_80_0 ctx)

def generatedRoot.w_83_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_82_0 ctx)

def generatedRoot.w_84_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_84_0

def generatedRoot.w_85_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_81_0 ctx) (generatedRoot.w_84_0 ctx)

def generatedRoot.w_87_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_86_0 ctx)

def generatedRoot.w_88_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_88_0

def generatedRoot.w_89_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_85_0 ctx) (generatedRoot.w_88_0 ctx)

def generatedRoot.w_91_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_90_0 ctx)

def generatedRoot.w_92_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_92_0

def generatedRoot.w_93_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_89_0 ctx) (generatedRoot.w_92_0 ctx)

def generatedRoot.w_95_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_94_0 ctx)

def generatedRoot.w_96_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_96_0

def generatedRoot.w_97_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_93_0 ctx) (generatedRoot.w_96_0 ctx)

def generatedRoot.w_99_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_98_0 ctx)

def generatedRoot.w_100_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_100_0

def generatedRoot.w_101_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_97_0 ctx) (generatedRoot.w_100_0 ctx)

def generatedRoot.w_103_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_102_0 ctx)

def generatedRoot.w_104_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_104_0

def generatedRoot.w_105_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_101_0 ctx) (generatedRoot.w_104_0 ctx)

def generatedRoot.w_107_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_106_0 ctx)

def generatedRoot.w_108_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_108_0

def generatedRoot.w_109_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_105_0 ctx) (generatedRoot.w_108_0 ctx)

def generatedRoot.w_111_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_110_0 ctx)

def generatedRoot.w_112_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_112_0

def generatedRoot.w_113_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_109_0 ctx) (generatedRoot.w_112_0 ctx)

def generatedRoot.w_115_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_114_0 ctx)

def generatedRoot.w_116_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_116_0

def generatedRoot.w_117_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_113_0 ctx) (generatedRoot.w_116_0 ctx)

def generatedRoot.w_119_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_118_0 ctx)

def generatedRoot.w_120_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_120_0

def generatedRoot.w_121_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_117_0 ctx) (generatedRoot.w_120_0 ctx)

def generatedRoot.w_123_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_122_0 ctx)

def generatedRoot.w_124_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_124_0

def generatedRoot.w_125_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_121_0 ctx) (generatedRoot.w_124_0 ctx)

def generatedRoot.w_127_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_126_0 ctx)

def generatedRoot.w_128_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_128_0

def generatedRoot.w_129_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_125_0 ctx) (generatedRoot.w_128_0 ctx)

def generatedRoot.w_131_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_130_0 ctx)

def generatedRoot.w_132_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_132_0

def generatedRoot.w_133_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_129_0 ctx) (generatedRoot.w_132_0 ctx)

def generatedRoot.w_135_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_134_0 ctx)

def generatedRoot.w_136_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_136_0

def generatedRoot.w_137_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_133_0 ctx) (generatedRoot.w_136_0 ctx)

def generatedRoot.w_139_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_138_0 ctx)

def generatedRoot.w_140_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_140_0

def generatedRoot.w_141_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_137_0 ctx) (generatedRoot.w_140_0 ctx)

def generatedRoot.w_143_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_142_0 ctx)

def generatedRoot.w_144_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_144_0

def generatedRoot.w_145_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_141_0 ctx) (generatedRoot.w_144_0 ctx)

def generatedRoot.w_147_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_146_0 ctx)

def generatedRoot.w_148_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_148_0

def generatedRoot.w_149_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_145_0 ctx) (generatedRoot.w_148_0 ctx)

def generatedRoot.w_151_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_150_0 ctx)

def generatedRoot.w_152_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_152_0

def generatedRoot.w_153_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_149_0 ctx) (generatedRoot.w_152_0 ctx)

def generatedRoot.w_155_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_154_0 ctx)

def generatedRoot.w_156_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_156_0

def generatedRoot.w_157_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_153_0 ctx) (generatedRoot.w_156_0 ctx)

def generatedRoot.w_159_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_158_0 ctx)

def generatedRoot.w_160_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_160_0

def generatedRoot.w_161_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_157_0 ctx) (generatedRoot.w_160_0 ctx)

def generatedRoot.w_163_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_162_0 ctx)

def generatedRoot.w_164_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_164_0

def generatedRoot.w_165_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_0 ctx) (generatedRoot.w_164_0 ctx)

def generatedRoot.w_167_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_166_0 ctx)

def generatedRoot.w_168_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_168_0

def generatedRoot.w_169_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_165_0 ctx) (generatedRoot.w_168_0 ctx)

def generatedRoot.w_171_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_170_0 ctx)

def generatedRoot.w_172_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_172_0

def generatedRoot.w_173_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_169_0 ctx) (generatedRoot.w_172_0 ctx)

def generatedRoot.w_175_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_174_0 ctx)

def generatedRoot.w_176_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_176_0

def generatedRoot.w_177_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_173_0 ctx) (generatedRoot.w_176_0 ctx)

def generatedRoot.w_179_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_178_0 ctx)

def generatedRoot.w_180_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_180_0

def generatedRoot.w_181_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_177_0 ctx) (generatedRoot.w_180_0 ctx)

def generatedRoot.w_183_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_182_0 ctx)

def generatedRoot.w_184_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_184_0

def generatedRoot.w_185_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_181_0 ctx) (generatedRoot.w_184_0 ctx)

def generatedRoot.w_187_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_186_0 ctx)

def generatedRoot.w_188_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_188_0

def generatedRoot.w_189_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_185_0 ctx) (generatedRoot.w_188_0 ctx)

def generatedRoot.w_191_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_190_0 ctx)

def generatedRoot.w_192_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_192_0

def generatedRoot.w_193_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_189_0 ctx) (generatedRoot.w_192_0 ctx)

def generatedRoot.w_195_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_194_0 ctx)

def generatedRoot.w_196_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_196_0

def generatedRoot.w_197_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_193_0 ctx) (generatedRoot.w_196_0 ctx)

def generatedRoot.w_199_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_198_0 ctx)

def generatedRoot.w_200_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_200_0

def generatedRoot.w_201_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_197_0 ctx) (generatedRoot.w_200_0 ctx)

def generatedRoot.w_203_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_202_0 ctx)

def generatedRoot.w_204_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_204_0

def generatedRoot.w_205_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_201_0 ctx) (generatedRoot.w_204_0 ctx)

def generatedRoot.w_207_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_206_0 ctx)

def generatedRoot.w_208_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_208_0

def generatedRoot.w_209_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_205_0 ctx) (generatedRoot.w_208_0 ctx)

def generatedRoot.w_211_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_210_0 ctx)

def generatedRoot.w_212_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_212_0

def generatedRoot.w_213_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_209_0 ctx) (generatedRoot.w_212_0 ctx)

def generatedRoot.w_215_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_214_0 ctx)

def generatedRoot.w_216_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_216_0

def generatedRoot.w_217_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_213_0 ctx) (generatedRoot.w_216_0 ctx)

def generatedRoot.w_219_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_218_0 ctx)

def generatedRoot.w_220_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_220_0

def generatedRoot.w_221_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_217_0 ctx) (generatedRoot.w_220_0 ctx)

def generatedRoot.w_223_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_222_0 ctx)

def generatedRoot.w_224_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_224_0

def generatedRoot.w_225_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_221_0 ctx) (generatedRoot.w_224_0 ctx)

def generatedRoot.w_227_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_226_0 ctx)

def generatedRoot.w_228_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_228_0

def generatedRoot.w_229_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_225_0 ctx) (generatedRoot.w_228_0 ctx)

def generatedRoot.w_231_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_230_0 ctx)

def generatedRoot.w_232_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_232_0

def generatedRoot.w_233_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_229_0 ctx) (generatedRoot.w_232_0 ctx)

def generatedRoot.w_235_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_234_0 ctx)

def generatedRoot.w_236_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_236_0

def generatedRoot.w_237_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_233_0 ctx) (generatedRoot.w_236_0 ctx)

def generatedRoot.w_239_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_238_0 ctx)

def generatedRoot.w_240_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_240_0

def generatedRoot.w_241_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_237_0 ctx) (generatedRoot.w_240_0 ctx)

def generatedRoot.w_243_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_242_0 ctx)

def generatedRoot.w_244_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_244_0

def generatedRoot.w_245_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_241_0 ctx) (generatedRoot.w_244_0 ctx)

def generatedRoot.w_247_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_246_0 ctx)

def generatedRoot.w_248_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_248_0

def generatedRoot.w_249_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_245_0 ctx) (generatedRoot.w_248_0 ctx)

def generatedRoot.w_251_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_250_0 ctx)

def generatedRoot.w_252_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_252_0

def generatedRoot.w_253_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_249_0 ctx) (generatedRoot.w_252_0 ctx)

def generatedRoot.w_255_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_254_0 ctx)

def generatedRoot.w_256_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_256_0

def generatedRoot.w_257_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_253_0 ctx) (generatedRoot.w_256_0 ctx)

def generatedRoot.w_259_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_258_0 ctx)

def generatedRoot.w_260_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_260_0

def generatedRoot.w_261_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_257_0 ctx) (generatedRoot.w_260_0 ctx)

def generatedRoot.w_263_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_262_0 ctx)

def generatedRoot.w_264_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_264_0

def generatedRoot.w_265_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_261_0 ctx) (generatedRoot.w_264_0 ctx)

def generatedRoot.w_267_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_266_0 ctx)

def generatedRoot.w_268_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_268_0

def generatedRoot.w_269_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_265_0 ctx) (generatedRoot.w_268_0 ctx)

def generatedRoot.w_271_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_270_0 ctx)

def generatedRoot.w_272_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_272_0

def generatedRoot.w_273_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_269_0 ctx) (generatedRoot.w_272_0 ctx)

def generatedRoot.w_275_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_274_0 ctx)

def generatedRoot.w_276_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_276_0

def generatedRoot.w_277_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_273_0 ctx) (generatedRoot.w_276_0 ctx)

def generatedRoot.w_279_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_278_0 ctx)

def generatedRoot.w_280_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_280_0

def generatedRoot.w_281_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_277_0 ctx) (generatedRoot.w_280_0 ctx)

def generatedRoot.w_283_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_282_0 ctx)

def generatedRoot.w_284_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_284_0

def generatedRoot.w_285_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_281_0 ctx) (generatedRoot.w_284_0 ctx)

def generatedRoot.w_287_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_286_0 ctx)

def generatedRoot.w_288_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_288_0

def generatedRoot.w_289_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_285_0 ctx) (generatedRoot.w_288_0 ctx)

def generatedRoot.w_291_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_290_0 ctx)

def generatedRoot.w_292_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_292_0

def generatedRoot.w_293_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_289_0 ctx) (generatedRoot.w_292_0 ctx)

def generatedRoot.w_295_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_294_0 ctx)

def generatedRoot.w_296_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_296_0

def generatedRoot.w_297_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_293_0 ctx) (generatedRoot.w_296_0 ctx)

def generatedRoot.w_299_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_298_0 ctx)

def generatedRoot.w_300_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_300_0

def generatedRoot.w_301_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_297_0 ctx) (generatedRoot.w_300_0 ctx)

def generatedRoot.w_303_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_302_0 ctx)

def generatedRoot.w_304_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_304_0

def generatedRoot.w_305_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_301_0 ctx) (generatedRoot.w_304_0 ctx)

def generatedRoot.w_307_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_306_0 ctx)

def generatedRoot.w_308_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_308_0

def generatedRoot.w_309_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_305_0 ctx) (generatedRoot.w_308_0 ctx)

def generatedRoot.w_311_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_310_0 ctx)

def generatedRoot.w_312_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_312_0

def generatedRoot.w_313_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_309_0 ctx) (generatedRoot.w_312_0 ctx)

def generatedRoot.w_315_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_314_0 ctx)

def generatedRoot.w_316_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_316_0

def generatedRoot.w_317_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_313_0 ctx) (generatedRoot.w_316_0 ctx)

def generatedRoot.w_319_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_318_0 ctx)

def generatedRoot.w_320_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_320_0

def generatedRoot.w_321_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_317_0 ctx) (generatedRoot.w_320_0 ctx)

def generatedRoot.w_323_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_322_0 ctx)

def generatedRoot.w_324_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_324_0

def generatedRoot.w_325_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_321_0 ctx) (generatedRoot.w_324_0 ctx)

def generatedRoot.w_327_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_326_0 ctx)

def generatedRoot.w_328_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_328_0

def generatedRoot.w_329_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_325_0 ctx) (generatedRoot.w_328_0 ctx)

def generatedRoot.w_331_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_330_0 ctx)

def generatedRoot.w_332_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_332_0

def generatedRoot.w_333_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_329_0 ctx) (generatedRoot.w_332_0 ctx)

def generatedRoot.w_335_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_334_0 ctx)

def generatedRoot.w_336_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_336_0

def generatedRoot.w_337_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_333_0 ctx) (generatedRoot.w_336_0 ctx)

def generatedRoot.w_339_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_338_0 ctx)

def generatedRoot.w_340_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_340_0

def generatedRoot.w_341_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_337_0 ctx) (generatedRoot.w_340_0 ctx)

def generatedRoot.w_343_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_342_0 ctx)

def generatedRoot.w_344_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_344_0

def generatedRoot.w_345_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_341_0 ctx) (generatedRoot.w_344_0 ctx)

def generatedRoot.w_347_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_346_0 ctx)

def generatedRoot.w_348_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_348_0

def generatedRoot.w_349_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_345_0 ctx) (generatedRoot.w_348_0 ctx)

def generatedRoot.w_351_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_350_0 ctx)

def generatedRoot.w_352_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_352_0

def generatedRoot.w_353_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_349_0 ctx) (generatedRoot.w_352_0 ctx)

def generatedRoot.w_355_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_354_0 ctx)

def generatedRoot.w_356_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_356_0

def generatedRoot.w_357_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_353_0 ctx) (generatedRoot.w_356_0 ctx)

def generatedRoot.w_359_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_358_0 ctx)

def generatedRoot.w_360_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_360_0

def generatedRoot.w_361_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_357_0 ctx) (generatedRoot.w_360_0 ctx)

def generatedRoot.w_363_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_362_0 ctx)

def generatedRoot.w_364_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_364_0

def generatedRoot.w_365_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_361_0 ctx) (generatedRoot.w_364_0 ctx)

def generatedRoot.w_367_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_366_0 ctx)

def generatedRoot.w_368_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_368_0

def generatedRoot.w_369_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_365_0 ctx) (generatedRoot.w_368_0 ctx)

def generatedRoot.w_371_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_370_0 ctx)

def generatedRoot.w_372_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_372_0

def generatedRoot.w_373_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_369_0 ctx) (generatedRoot.w_372_0 ctx)

def generatedRoot.w_375_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_374_0 ctx)

def generatedRoot.w_376_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_376_0

def generatedRoot.w_377_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_373_0 ctx) (generatedRoot.w_376_0 ctx)

def generatedRoot.w_379_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_378_0 ctx)

def generatedRoot.w_380_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_380_0

def generatedRoot.w_381_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_377_0 ctx) (generatedRoot.w_380_0 ctx)

def generatedRoot.w_383_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_382_0 ctx)

def generatedRoot.w_384_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_384_0

def generatedRoot.w_385_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_381_0 ctx) (generatedRoot.w_384_0 ctx)

def generatedRoot.w_387_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_386_0 ctx)

def generatedRoot.w_388_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_388_0

def generatedRoot.w_389_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_385_0 ctx) (generatedRoot.w_388_0 ctx)

def generatedRoot.w_391_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_390_0 ctx)

def generatedRoot.w_392_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_392_0

def generatedRoot.w_393_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_389_0 ctx) (generatedRoot.w_392_0 ctx)

def generatedRoot.w_395_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_394_0 ctx)

def generatedRoot.w_396_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_396_0

def generatedRoot.w_397_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_393_0 ctx) (generatedRoot.w_396_0 ctx)

def generatedRoot.w_399_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_398_0 ctx)

def generatedRoot.w_400_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_400_0

def generatedRoot.w_401_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_397_0 ctx) (generatedRoot.w_400_0 ctx)

def generatedRoot.w_403_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_402_0 ctx)

def generatedRoot.w_404_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_404_0

def generatedRoot.w_405_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_401_0 ctx) (generatedRoot.w_404_0 ctx)

def generatedRoot.w_407_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_406_0 ctx)

def generatedRoot.w_408_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_408_0

def generatedRoot.w_409_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_405_0 ctx) (generatedRoot.w_408_0 ctx)

def generatedRoot.w_411_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_410_0 ctx)

def generatedRoot.w_412_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_412_0

def generatedRoot.w_413_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_409_0 ctx) (generatedRoot.w_412_0 ctx)

def generatedRoot.w_415_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_414_0 ctx)

def generatedRoot.w_416_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_416_0

def generatedRoot.w_417_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_413_0 ctx) (generatedRoot.w_416_0 ctx)

def generatedRoot.w_419_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_418_0 ctx)

def generatedRoot.w_420_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_420_0

def generatedRoot.w_421_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_417_0 ctx) (generatedRoot.w_420_0 ctx)

def generatedRoot.w_423_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_422_0 ctx)

def generatedRoot.w_424_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_424_0

def generatedRoot.w_425_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_421_0 ctx) (generatedRoot.w_424_0 ctx)

def generatedRoot.w_427_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_426_0 ctx)

def generatedRoot.w_428_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_428_0

def generatedRoot.w_429_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_425_0 ctx) (generatedRoot.w_428_0 ctx)

def generatedRoot.w_431_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_430_0 ctx)

def generatedRoot.w_432_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_432_0

def generatedRoot.w_433_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_429_0 ctx) (generatedRoot.w_432_0 ctx)

def generatedRoot.w_435_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_434_0 ctx)

def generatedRoot.w_436_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_436_0

def generatedRoot.w_437_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_433_0 ctx) (generatedRoot.w_436_0 ctx)

def generatedRoot.w_439_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_438_0 ctx)

def generatedRoot.w_440_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_440_0

def generatedRoot.w_441_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_437_0 ctx) (generatedRoot.w_440_0 ctx)

def generatedRoot.w_443_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_442_0 ctx)

def generatedRoot.w_444_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_444_0

def generatedRoot.w_445_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_441_0 ctx) (generatedRoot.w_444_0 ctx)

def generatedRoot.w_447_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_446_0 ctx)

def generatedRoot.w_448_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_448_0

def generatedRoot.w_449_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_445_0 ctx) (generatedRoot.w_448_0 ctx)

def generatedRoot.w_451_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_450_0 ctx)

def generatedRoot.w_452_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_452_0

def generatedRoot.w_453_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_449_0 ctx) (generatedRoot.w_452_0 ctx)

def generatedRoot.w_455_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_454_0 ctx)

def generatedRoot.w_456_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_456_0

def generatedRoot.w_457_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_453_0 ctx) (generatedRoot.w_456_0 ctx)

def generatedRoot.w_459_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_458_0 ctx)

def generatedRoot.w_460_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_460_0

def generatedRoot.w_461_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_457_0 ctx) (generatedRoot.w_460_0 ctx)

def generatedRoot.w_463_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_462_0 ctx)

def generatedRoot.w_464_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_464_0

def generatedRoot.w_465_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_461_0 ctx) (generatedRoot.w_464_0 ctx)

def generatedRoot.w_467_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_466_0 ctx)

def generatedRoot.w_468_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_468_0

def generatedRoot.w_469_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_465_0 ctx) (generatedRoot.w_468_0 ctx)

def generatedRoot.w_471_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_470_0 ctx)

def generatedRoot.w_472_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_472_0

def generatedRoot.w_473_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_469_0 ctx) (generatedRoot.w_472_0 ctx)

def generatedRoot.w_475_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_474_0 ctx)

def generatedRoot.w_476_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_476_0

def generatedRoot.w_477_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_473_0 ctx) (generatedRoot.w_476_0 ctx)

def generatedRoot.w_479_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_478_0 ctx)

def generatedRoot.w_480_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_480_0

def generatedRoot.w_481_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_477_0 ctx) (generatedRoot.w_480_0 ctx)

def generatedRoot.w_483_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_482_0 ctx)

def generatedRoot.w_484_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_484_0

def generatedRoot.w_485_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_481_0 ctx) (generatedRoot.w_484_0 ctx)

def generatedRoot.w_487_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_486_0 ctx)

def generatedRoot.w_488_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_488_0

def generatedRoot.w_489_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_485_0 ctx) (generatedRoot.w_488_0 ctx)

def generatedRoot.w_491_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_490_0 ctx)

def generatedRoot.w_492_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_492_0

def generatedRoot.w_493_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_489_0 ctx) (generatedRoot.w_492_0 ctx)

def generatedRoot.w_495_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_494_0 ctx)

def generatedRoot.w_496_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_496_0

def generatedRoot.w_497_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_493_0 ctx) (generatedRoot.w_496_0 ctx)

def generatedRoot.w_499_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_498_0 ctx)

def generatedRoot.w_500_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_500_0

def generatedRoot.w_501_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_497_0 ctx) (generatedRoot.w_500_0 ctx)

def generatedRoot.w_503_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_502_0 ctx)

def generatedRoot.w_504_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_504_0

def generatedRoot.w_505_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_501_0 ctx) (generatedRoot.w_504_0 ctx)

def generatedRoot.w_507_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_506_0 ctx)

def generatedRoot.w_508_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_508_0

def generatedRoot.w_509_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_505_0 ctx) (generatedRoot.w_508_0 ctx)

def generatedRoot.w_511_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_510_0 ctx)

def generatedRoot.w_512_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_512_0

def generatedRoot.w_513_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_509_0 ctx) (generatedRoot.w_512_0 ctx)

def generatedRoot.w_515_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_514_0 ctx)

def generatedRoot.w_516_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_516_0

def generatedRoot.w_517_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_513_0 ctx) (generatedRoot.w_516_0 ctx)

def generatedRoot.w_519_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_9_0 ctx) (generatedRoot.w_518_0 ctx)

def generatedRoot.w_520_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_520_0

def generatedRoot.w_521_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_517_0 ctx) (generatedRoot.w_520_0 ctx)

def generatedRoot.w_522_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_523_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat7 :=
  ctx.witness.w_523_0

def generatedRoot.w_524_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.witness.w_524_0

def generatedRoot.w_525_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat8 :=
  ctx.witness.w_525_0

def generatedRoot.w_526_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat8 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx)

def generatedRoot.w_527_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.witness.w_527_0

def generatedRoot.w_528_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  (generatedRoot.w_527_0 ctx)

def generatedRoot.w_529_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_529_0

def generatedRoot.w_530_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat7 :=
  ctx.witness.w_530_0

def generatedRoot.w_531_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.witness.w_531_0

def generatedRoot.w_532_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat8 :=
  ctx.witness.w_532_0

def generatedRoot.w_533_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat8 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_530_0 ctx) (generatedRoot.w_532_0 ctx)

def generatedRoot.w_534_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.witness.w_534_0

def generatedRoot.w_535_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  (generatedRoot.w_534_0 ctx)

def generatedRoot.w_536_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  MxxRuntime.matrixSub (generatedRoot.w_528_0 ctx) (generatedRoot.w_535_0 ctx)

def generatedRoot.w_538_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_538_0

def generatedRoot.w_539_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_521_0 ctx) (generatedRoot.w_538_0 ctx)

def generatedRoot.w_540_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (0 : mxxMat0)

def generatedRoot.w_543_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  (0 : mxxMat4)

def generatedRoot.w_544_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat4 :=
  ctx.witness.w_544_0

def generatedRoot.w_545_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat9 :=
  ctx.witness.w_545_0

def generatedRoot.w_547_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_547_0

def generatedRoot.w_548_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_539_0 ctx) (generatedRoot.w_547_0 ctx)

def generatedRoot.w_551_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat9 :=
  ctx.witness.w_551_0

def generatedRoot.w_554_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat9 :=
  ctx.witness.w_554_0

def generatedRoot.w_555_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat9 :=
  MxxRuntime.matrixSub (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx)

def generatedRoot.w_557_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx)

def generatedRoot.w_558_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_548_0 ctx) (generatedRoot.w_557_0 ctx)

def generatedRoot.w_559_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixSub (generatedRoot.w_3_0 ctx) (generatedRoot.w_558_0 ctx)

def generatedRoot.w_560_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 65536 → Int :=
  ctx.witness.w_560_0

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 65536 → Int :=
  (generatedRoot.w_560_0 ctx)

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h2_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_2_0 ctx)
  h6_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)
  h7_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_7_0 ctx)
  h8_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx)
  h12_0 : MxxRuntime.ringAutomorphismRuns (1) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx)
  h16_0 : MxxRuntime.ringAutomorphismRuns (1025) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx)
  h20_0 : MxxRuntime.ringAutomorphismRuns (2049) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx)
  h24_0 : MxxRuntime.ringAutomorphismRuns (3073) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx)
  h28_0 : MxxRuntime.ringAutomorphismRuns (4097) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx)
  h32_0 : MxxRuntime.ringAutomorphismRuns (5121) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx)
  h36_0 : MxxRuntime.ringAutomorphismRuns (6145) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx)
  h40_0 : MxxRuntime.ringAutomorphismRuns (7169) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx)
  h44_0 : MxxRuntime.ringAutomorphismRuns (8193) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx)
  h48_0 : MxxRuntime.ringAutomorphismRuns (9217) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx)
  h52_0 : MxxRuntime.ringAutomorphismRuns (10241) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx)
  h56_0 : MxxRuntime.ringAutomorphismRuns (11265) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx)
  h60_0 : MxxRuntime.ringAutomorphismRuns (12289) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx)
  h64_0 : MxxRuntime.ringAutomorphismRuns (13313) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)
  h68_0 : MxxRuntime.ringAutomorphismRuns (14337) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx)
  h72_0 : MxxRuntime.ringAutomorphismRuns (15361) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx)
  h76_0 : MxxRuntime.ringAutomorphismRuns (16385) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx)
  h80_0 : MxxRuntime.ringAutomorphismRuns (17409) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx)
  h84_0 : MxxRuntime.ringAutomorphismRuns (18433) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx)
  h88_0 : MxxRuntime.ringAutomorphismRuns (19457) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx)
  h92_0 : MxxRuntime.ringAutomorphismRuns (20481) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx)
  h96_0 : MxxRuntime.ringAutomorphismRuns (21505) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx)
  h100_0 : MxxRuntime.ringAutomorphismRuns (22529) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx)
  h104_0 : MxxRuntime.ringAutomorphismRuns (23553) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx)
  h108_0 : MxxRuntime.ringAutomorphismRuns (24577) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx)
  h112_0 : MxxRuntime.ringAutomorphismRuns (25601) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx)
  h116_0 : MxxRuntime.ringAutomorphismRuns (26625) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx)
  h120_0 : MxxRuntime.ringAutomorphismRuns (27649) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx)

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_80_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_88_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_111_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_119_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx ∧ generatedRoot.constraints_1 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h124_0 : MxxRuntime.ringAutomorphismRuns (28673) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx)
  h128_0 : MxxRuntime.ringAutomorphismRuns (29697) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx)
  h132_0 : MxxRuntime.ringAutomorphismRuns (30721) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx)
  h136_0 : MxxRuntime.ringAutomorphismRuns (31745) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx)
  h140_0 : MxxRuntime.ringAutomorphismRuns (32769) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx)
  h144_0 : MxxRuntime.ringAutomorphismRuns (33793) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx)
  h148_0 : MxxRuntime.ringAutomorphismRuns (34817) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx)
  h152_0 : MxxRuntime.ringAutomorphismRuns (35841) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx)
  h156_0 : MxxRuntime.ringAutomorphismRuns (36865) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx)
  h160_0 : MxxRuntime.ringAutomorphismRuns (37889) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx)
  h164_0 : MxxRuntime.ringAutomorphismRuns (38913) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx)
  h168_0 : MxxRuntime.ringAutomorphismRuns (39937) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx)
  h172_0 : MxxRuntime.ringAutomorphismRuns (40961) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx)
  h176_0 : MxxRuntime.ringAutomorphismRuns (41985) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx)
  h180_0 : MxxRuntime.ringAutomorphismRuns (43009) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx)
  h184_0 : MxxRuntime.ringAutomorphismRuns (44033) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx)
  h188_0 : MxxRuntime.ringAutomorphismRuns (45057) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx)
  h192_0 : MxxRuntime.ringAutomorphismRuns (46081) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx)
  h196_0 : MxxRuntime.ringAutomorphismRuns (47105) (generatedRoot.w_195_0 ctx) (generatedRoot.w_196_0 ctx)
  h200_0 : MxxRuntime.ringAutomorphismRuns (48129) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx)
  h204_0 : MxxRuntime.ringAutomorphismRuns (49153) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx)
  h208_0 : MxxRuntime.ringAutomorphismRuns (50177) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx)
  h212_0 : MxxRuntime.ringAutomorphismRuns (51201) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx)
  h216_0 : MxxRuntime.ringAutomorphismRuns (52225) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx)
  h220_0 : MxxRuntime.ringAutomorphismRuns (53249) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx)
  h224_0 : MxxRuntime.ringAutomorphismRuns (54273) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx)
  h228_0 : MxxRuntime.ringAutomorphismRuns (55297) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx)
  h232_0 : MxxRuntime.ringAutomorphismRuns (56321) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx)
  h236_0 : MxxRuntime.ringAutomorphismRuns (57345) (generatedRoot.w_235_0 ctx) (generatedRoot.w_236_0 ctx)
  h240_0 : MxxRuntime.ringAutomorphismRuns (58369) (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx)
  h244_0 : MxxRuntime.ringAutomorphismRuns (59393) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx)
  h248_0 : MxxRuntime.ringAutomorphismRuns (60417) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx)

theorem generatedRoot.facts_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_1 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_135_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_196_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_203_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_223_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_1 ctx ∧ generatedRoot.constraints_2 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h252_0 : MxxRuntime.ringAutomorphismRuns (61441) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx)
  h256_0 : MxxRuntime.ringAutomorphismRuns (62465) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx)
  h260_0 : MxxRuntime.ringAutomorphismRuns (63489) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx)
  h264_0 : MxxRuntime.ringAutomorphismRuns (64513) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx)
  h268_0 : MxxRuntime.ringAutomorphismRuns (65537) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx)
  h272_0 : MxxRuntime.ringAutomorphismRuns (66561) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx)
  h276_0 : MxxRuntime.ringAutomorphismRuns (67585) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx)
  h280_0 : MxxRuntime.ringAutomorphismRuns (68609) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx)
  h284_0 : MxxRuntime.ringAutomorphismRuns (69633) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx)
  h288_0 : MxxRuntime.ringAutomorphismRuns (70657) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx)
  h292_0 : MxxRuntime.ringAutomorphismRuns (71681) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx)
  h296_0 : MxxRuntime.ringAutomorphismRuns (72705) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx)
  h300_0 : MxxRuntime.ringAutomorphismRuns (73729) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx)
  h304_0 : MxxRuntime.ringAutomorphismRuns (74753) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx)
  h308_0 : MxxRuntime.ringAutomorphismRuns (75777) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx)
  h312_0 : MxxRuntime.ringAutomorphismRuns (76801) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx)
  h316_0 : MxxRuntime.ringAutomorphismRuns (77825) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx)
  h320_0 : MxxRuntime.ringAutomorphismRuns (78849) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx)
  h324_0 : MxxRuntime.ringAutomorphismRuns (79873) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx)
  h328_0 : MxxRuntime.ringAutomorphismRuns (80897) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx)
  h332_0 : MxxRuntime.ringAutomorphismRuns (81921) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx)
  h336_0 : MxxRuntime.ringAutomorphismRuns (82945) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx)
  h340_0 : MxxRuntime.ringAutomorphismRuns (83969) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx)
  h344_0 : MxxRuntime.ringAutomorphismRuns (84993) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx)
  h348_0 : MxxRuntime.ringAutomorphismRuns (86017) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx)
  h352_0 : MxxRuntime.ringAutomorphismRuns (87041) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx)
  h356_0 : MxxRuntime.ringAutomorphismRuns (88065) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx)
  h360_0 : MxxRuntime.ringAutomorphismRuns (89089) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx)
  h364_0 : MxxRuntime.ringAutomorphismRuns (90113) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx)
  h368_0 : MxxRuntime.ringAutomorphismRuns (91137) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx)
  h372_0 : MxxRuntime.ringAutomorphismRuns (92161) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx)
  h376_0 : MxxRuntime.ringAutomorphismRuns (93185) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx)

theorem generatedRoot.facts_2 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_2 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_272_0 ctx) (generatedRoot.w_275_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_279_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_332_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_339_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_2 ctx ∧ generatedRoot.constraints_3 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h380_0 : MxxRuntime.ringAutomorphismRuns (94209) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx)
  h384_0 : MxxRuntime.ringAutomorphismRuns (95233) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx)
  h388_0 : MxxRuntime.ringAutomorphismRuns (96257) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx)
  h392_0 : MxxRuntime.ringAutomorphismRuns (97281) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx)
  h396_0 : MxxRuntime.ringAutomorphismRuns (98305) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx)
  h400_0 : MxxRuntime.ringAutomorphismRuns (99329) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx)
  h404_0 : MxxRuntime.ringAutomorphismRuns (100353) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx)
  h408_0 : MxxRuntime.ringAutomorphismRuns (101377) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx)
  h412_0 : MxxRuntime.ringAutomorphismRuns (102401) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx)
  h416_0 : MxxRuntime.ringAutomorphismRuns (103425) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx)
  h420_0 : MxxRuntime.ringAutomorphismRuns (104449) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx)
  h424_0 : MxxRuntime.ringAutomorphismRuns (105473) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx)
  h428_0 : MxxRuntime.ringAutomorphismRuns (106497) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx)
  h432_0 : MxxRuntime.ringAutomorphismRuns (107521) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx)
  h436_0 : MxxRuntime.ringAutomorphismRuns (108545) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx)
  h440_0 : MxxRuntime.ringAutomorphismRuns (109569) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx)
  h444_0 : MxxRuntime.ringAutomorphismRuns (110593) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx)
  h448_0 : MxxRuntime.ringAutomorphismRuns (111617) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx)
  h452_0 : MxxRuntime.ringAutomorphismRuns (112641) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx)
  h456_0 : MxxRuntime.ringAutomorphismRuns (113665) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx)
  h460_0 : MxxRuntime.ringAutomorphismRuns (114689) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx)
  h464_0 : MxxRuntime.ringAutomorphismRuns (115713) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx)
  h468_0 : MxxRuntime.ringAutomorphismRuns (116737) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx)
  h472_0 : MxxRuntime.ringAutomorphismRuns (117761) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx)
  h476_0 : MxxRuntime.ringAutomorphismRuns (118785) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx)
  h480_0 : MxxRuntime.ringAutomorphismRuns (119809) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx)
  h484_0 : MxxRuntime.ringAutomorphismRuns (120833) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx)
  h488_0 : MxxRuntime.ringAutomorphismRuns (121857) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx)
  h492_0 : MxxRuntime.ringAutomorphismRuns (122881) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx)
  h496_0 : MxxRuntime.ringAutomorphismRuns (123905) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx)
  h500_0 : MxxRuntime.ringAutomorphismRuns (124929) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx)
  h504_0 : MxxRuntime.ringAutomorphismRuns (125953) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx)

theorem generatedRoot.facts_3 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_3 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_388_0 ctx) (generatedRoot.w_391_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_395_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_455_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_504_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_3 ctx ∧ generatedRoot.constraints_4 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Prop where
  h508_0 : MxxRuntime.ringAutomorphismRuns (126977) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx)
  h512_0 : MxxRuntime.ringAutomorphismRuns (128001) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx)
  h516_0 : MxxRuntime.ringAutomorphismRuns (129025) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx)
  h520_0 : MxxRuntime.ringAutomorphismRuns (130049) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx)
  t132 : 0 ≤ 0
  t133 : 0 < 1
  t134 : 1 ≤ 1
  t135 : 0 ≤ 0
  t136 : 0 < 1
  t137 : 1 ≤ 501
  h523_0 : MxxRuntime.sliceMatrix (generatedRoot.w_5_0 ctx) 0 1 0 1 (generatedRoot.w_523_0 ctx)
  h524_0 : MxxRuntime.gadgetMatrixRuns backend 65536 44 (generatedRoot.w_524_0 ctx)
  h525_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx)
  h527_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx)
  h529_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) ((generatedRoot.w_1_0 ctx)) (generatedRoot.w_529_0 ctx)
  t143 : 0 ≤ 0
  t144 : 0 < 1
  t145 : 1 ≤ 1
  t146 : 0 ≤ 0
  t147 : 0 < 1
  t148 : 1 ≤ 501
  h530_0 : MxxRuntime.sliceMatrix (generatedRoot.w_529_0 ctx) 0 1 0 1 (generatedRoot.w_530_0 ctx)
  h531_0 : MxxRuntime.gadgetMatrixRuns backend 65536 44 (generatedRoot.w_531_0 ctx)
  h532_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx)
  h534_0 : MxxRuntime.modulusSwitchRuns (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx)
  t153 : 0 ≤ (128)
  h538_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_538 backend hashModel tape (path ++ [538, i]) params i (current, ((generatedRoot.w_536_0 ctx), ((generatedRoot.w_537_0 ctx), ()))) next) (Int.toNat (128)) (generatedRoot.w_522_0 ctx) (generatedRoot.w_538_0 ctx)
  h544_0 : MxxRuntime.gadgetMatrixRuns backend 65536 44 (generatedRoot.w_544_0 ctx)
  h545_0 : MxxRuntime.concatColumns (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx)
  t157 : 0 ≤ (4736)
  h547_0 : MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_547 backend hashModel tape (path ++ [547, i]) params i (current, ((generatedRoot.w_541_0 ctx), ((generatedRoot.w_542_0 ctx), ((generatedRoot.w_1_0 ctx), ((generatedRoot.w_545_0 ctx), ((generatedRoot.w_546_0 ctx), ())))))) next) (Int.toNat (4736)) (generatedRoot.w_540_0 ctx) (generatedRoot.w_547_0 ctx)
  h551_0 : MxxRuntime.concatColumns (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx)

theorem generatedRoot.facts_4 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_4 backend hashModel tape path params (generatedRoot.w_1_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_507_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_511_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_4 ctx ∧ generatedRoot.constraints_5 (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs :=
  ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩, h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : Fin 65536 → Int) : Prop where
  h554_0 : MxxRuntime.concatColumns (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx)
  t161 : (65536) = 65536
  h560_0 : (∀ k : Fin 65536, MxxRuntime.thresholdDecode (2) (65536) (k.val : Int) (generatedRoot.w_559_0 ctx) ((generatedRoot.w_560_0 ctx) k))
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_5 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : Fin 65536 → Int}
    (h : generatedRoot.constraints_5 (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) outputs) :
    generatedRoot.Facts_5 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : Fin 65536 → Int) : Prop where
  c0 : generatedRoot.Facts_0 ctx
  c1 : generatedRoot.Facts_1 ctx
  c2 : generatedRoot.Facts_2 ctx
  c3 : generatedRoot.Facts_3 ctx
  c4 : generatedRoot.Facts_4 ctx
  c5 : generatedRoot.Facts_5 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : Fin 65536 → Int}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  have r1 := generatedRoot.facts_1 (ctx := ctx) (outputs := outputs) r0.2
  have r2 := generatedRoot.facts_2 (ctx := ctx) (outputs := outputs) r1.2
  have r3 := generatedRoot.facts_3 (ctx := ctx) (outputs := outputs) r2.2
  have r4 := generatedRoot.facts_4 (ctx := ctx) (outputs := outputs) r3.2
  have r5 := generatedRoot.facts_5 (ctx := ctx) (outputs := outputs) r4.2
  ⟨r0.1, r1.1, r2.1, r3.1, r4.1, r5⟩

end Stage_decode

end
