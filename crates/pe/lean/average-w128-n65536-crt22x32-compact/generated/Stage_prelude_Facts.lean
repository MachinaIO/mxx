import Stage_prelude

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
set_option linter.unusedVariables false

noncomputable section

namespace Stage_prelude

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat2" => 39039761963948063856004653999987206021567571570771970785059555410812382343247291324210616315082566718137759516127340486041270513026828923337402589479793114710260407273133913650612796659174738653801475847949582336
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 501
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : mxxMat0 × (Fin 4 → Int) × ByteArray × mxxMat1 × (Fin 65536 → Int) × Unit
  witness : generatedRoot.Witness

def generatedRoot.w_0_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.inputs.1

def generatedRoot.w_1_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → Int :=
  ctx.inputs.2.1

def generatedRoot.w_2_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : ByteArray :=
  ctx.inputs.2.2.1

def generatedRoot.w_6_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.inputs.2.2.2.1

def generatedRoot.w_19_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 65536 → Int :=
  ctx.inputs.2.2.2.2.1

def generatedRoot.w_3_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (1 : mxxMat0)

def generatedRoot.w_4_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat2 :=
  ctx.witness.w_4_0

def generatedRoot.w_4_1 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : Fin 4 → mxxMat2 :=
  ctx.witness.w_4_1

def generatedRoot.w_5_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat3 :=
  ctx.witness.w_5_0

def generatedRoot.w_7_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx)

def generatedRoot.w_8_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_8_0

def generatedRoot.w_9_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx)

def generatedRoot.w_10_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  ctx.witness.w_10_0

def generatedRoot.w_11_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat1 :=
  MxxRuntime.matrixAdd (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx)

def generatedRoot.w_12_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_12_0

def generatedRoot.w_13_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_0_0 ctx) (generatedRoot.w_12_0 ctx)

def generatedRoot.w_14_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_14_0

def generatedRoot.w_15_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx)

def generatedRoot.w_16_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_16_0

def generatedRoot.w_17_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx)

def generatedRoot.w_18_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat0)

def generatedRoot.w_20_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  ctx.witness.w_20_0

def generatedRoot.w_21_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_18_0 ctx) (generatedRoot.w_20_0 ctx)

def generatedRoot.w_22_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_17_0 ctx) (generatedRoot.w_21_0 ctx)

def generatedRoot.outs {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) : (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat1 × mxxMat0 × Unit :=
  ((generatedRoot.w_4_0 ctx), ((generatedRoot.w_4_1 ctx), ((generatedRoot.w_11_0 ctx), ((generatedRoot.w_22_0 ctx), ()))))

structure generatedRoot.Facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat1 × mxxMat0 × Unit) : Prop where
  t0 : (4) = 4
  h4_1 : (∀ i : Fin 4, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i ((generatedRoot.w_0_0 ctx), (((generatedRoot.w_1_0 ctx) i), ((generatedRoot.w_2_0 ctx), ((generatedRoot.w_3_0 ctx), ())))) (((generatedRoot.w_4_0 ctx) i), (((generatedRoot.w_4_1 ctx) i), ())))
  h5_0 : MxxRuntime.centeredRebaseRuns (generatedRoot.w_0_0 ctx) (generatedRoot.w_5_0 ctx)
  h8_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) ((generatedRoot.w_2_0 ctx)) (generatedRoot.w_8_0 ctx)
  h10_0 : MxxRuntime.gaussianSampleAt tape (path ++ [10]) ((319/100)) (31) (generatedRoot.w_10_0 ctx)
  h12_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) ((generatedRoot.w_2_0 ctx)) (generatedRoot.w_12_0 ctx)
  h14_0 : MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) ((generatedRoot.w_2_0 ctx)) (generatedRoot.w_14_0 ctx)
  h16_0 : MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((319/100)) (31) (generatedRoot.w_16_0 ctx)
  h20_0 : MxxRuntime.polynomialFromValues false (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx)
  hout : outputs = generatedRoot.outs ctx

theorem generatedRoot.facts_0 {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} {ctx : generatedRoot.Ctx backend hashModel tape path params} {outputs : (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat1 × mxxMat0 × Unit}
    (h : generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_4_1 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_22_0 ctx) outputs) :
    generatedRoot.Facts_0 ctx outputs :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1, h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2⟩

structure generatedRoot.Facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) (outputs : (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat1 × mxxMat0 × Unit) : Prop where
  c0 : generatedRoot.Facts_0 ctx outputs

theorem generatedRoot.facts {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params} (ctx : generatedRoot.Ctx backend hashModel tape path params) {outputs : (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat1 × mxxMat0 × Unit}
    (h : generatedRoot.body backend hashModel tape path params ctx.inputs outputs ctx.witness) :
    generatedRoot.Facts ctx outputs :=
  have r0 := generatedRoot.facts_0 (ctx := ctx) (outputs := outputs) h
  ⟨r0⟩

end Stage_prelude

end
