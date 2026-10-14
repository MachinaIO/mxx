import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt

structure Params where
  «instance_width» : Int
  «witness_width» : Int
  «depth» : Int
  «max_layer_width» : Int

local notation "mxxNat0" => 4199032064717278048011690883356525194992379936270639851797609809837895001364583592120037022226933909756656247900567822877571943016382306787143098810852811877625094742794960970844464922736833994470900666890053873639951149595456467501578465489991201126936586213728973902928874195506364204092973456027957091035104033910144070999857957500784999761836271291625467579499159857168554211193900637753055337630723292189696552347584654668781975710936869040185900394936649540219155894302636794805037545092627623095295031156777595773839388205056154665410311909809928858184354714366825048284928528842065424423855742725909289792505579894072947687733339223164412449622420831234281347965788132250049307707916047981319958031047384043172197347037167219293201743113127640459541149769614328709349508709870843040403487326960044099133215467314285406796203969485466854796895434690359338936679750042849348645128711268154511807115512978606017699127176171118501823149906613558897808138553858269286024476424336624820204590405295202501406869464141371126400791156111431121824813240387458335842239626864422671983894448859681136181263116997808611107286960133658221079372912733342982873610141257058432555021315430606781087208937908116551487489763593259903335437701918007432243654584045216429208588469763848330504816089550994380373030983742413755534916626936673748720385176721131930800530006167535825582016926525916604855546785318221679725469580564577732284734918601942896007731189942964222876921201035429303469194013473347100890468165307689477317496416756507840589003407953909556403972932074503554151240400133823255344596986541629757423369488383747884930896054192926747733786939321934033569958509718450784778530079804441055024077339908378600944084314478262103060830951207359174327208858572433587319191518818104399843912724262474520250906723908969366931485154629018624014039436058244997120526095341155216026946955843017105177242603954702487135149423867859100549025473381894053375150742122730467433963366202366470252343397361168442424880824500088146595956676009983814832754606224041047120953569123662484271425661221552981321751181508634796283318670361231881012842451682502050715014948102974511858240109863210827624539578470669990240162164428808387161705646124819423890769725258101450033623514670556914665252281334377570961232803525415599260484697269987012025928186936345949660940960927813899157089668926423682982782036630368513616402562333519001461474812006154044942834728610426823941779893452275713
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 524288 2 164275026681676416
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 524288 2 8184
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 524288 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 524288 1 8184
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 524288 1 2

abbrev parallel_generatedRoot_5.constraints_0 (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ (8184 * (Int.ofNat i_0)) ∧
  (8184 * (Int.ofNat i_0)) < (8184 + (8184 * (Int.ofNat i_0))) ∧
  (8184 + (8184 * (Int.ofNat i_0))) ≤ 164275026681676416 ∧
  MxxRuntime.sliceMatrix w_0_0 0 2 (8184 * (Int.ofNat i_0)) (8184 + (8184 * (Int.ofNat i_0))) w_1_0 ∧
  outputs = w_1_0

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs
  ∃ (w_1_0 : mxxMat1),
    parallel_generatedRoot_5.constraints_0 i_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_7.constraints_0 (w_1_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  outputs = w_1_0

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs
  let w_1_0 : mxxMat2 := MxxRuntime.liftInteger w_0_0
  parallel_generatedRoot_7.constraints_0 w_1_0 outputs

abbrev parallel_generatedRoot_10.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_3_0 : mxxMat2) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (outputs : mxxMat3 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [7]) ((319/100)) (20) w_7_0 ∧
  outputs = (w_8_0, (w_3_0, ()))

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat4 × mxxMat1 × mxxMat2 × mxxMat3 × Unit) (outputs : mxxMat3 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat4 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_3_0 : mxxMat2 := inputs.2.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : mxxMat3 := MxxRuntime.matrixMul w_0_0 w_1_0
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 : mxxMat3 := MxxRuntime.matrixSub w_2_0 w_5_0
  ∃ (w_7_0 : mxxMat3),
    let w_8_0 : mxxMat3 := MxxRuntime.matrixAdd w_6_0 w_7_0
    parallel_generatedRoot_10.constraints_0 tape path w_3_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat4
  w_4_0 : mxxMat0
  w_5_0 : Fin 20072706095024 → mxxMat1
  w_7_0 : Fin 20072706095024 → mxxMat2
  w_8_0 : mxxMat1
  w_10_0 : Fin 20072706095024 → mxxMat3
  w_10_1 : Fin 20072706095024 → mxxMat2
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat3
  w_21_0 : mxxMat3
  w_23_0 : mxxMat3

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat2) (w_3_0 : ByteArray) (w_6_0 : Fin 20072706095024 → Int) (w_1_0 : mxxMat2) (w_2_0 : mxxMat4) (w_4_0 : mxxMat0) (w_5_0 : Fin 20072706095024 → mxxMat1) (w_7_0 : Fin 20072706095024 → mxxMat2) (w_8_0 : mxxMat1) (w_9_0 : mxxMat3) (w_10_0 : Fin 20072706095024 → mxxMat3) (w_10_1 : Fin 20072706095024 → mxxMat2) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_16_0 : mxxMat2) (w_17_0 : mxxMat1) (w_18_0 : mxxMat3) (w_19_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (w_23_0 : mxxMat3) (outputs : (Fin 20072706095024 → mxxMat3) × mxxMat3 × Unit) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([98, 111, 111, 108, 101, 97, 110, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 115]) ([]) (w_3_0) w_4_0 ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i w_4_0 (w_5_0 i)) ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_7 backend hashModel tape (path ++ [7, i.val]) params i (w_6_0 i) (w_7_0 i)) ∧
  MxxRuntime.gadgetMatrixRuns backend 4 4092 w_8_0 ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_10 backend hashModel tape (path ++ [10, i.val]) params i (w_2_0, ((w_5_0 i), ((w_7_0 i), (w_9_0, ())))) ((w_10_0 i), ((w_10_1 i), ()))) ∧
  MxxRuntime.hashSample (hashModel) ([98, 111, 111, 108, 101, 97, 110, 45, 98, 103, 103, 47, 111, 110, 101]) ([]) (w_3_0) w_11_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8184 ∧
  8184 ≤ 8184 ∧
  MxxRuntime.sliceMatrix w_11_0 0 2 0 8184 w_12_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4 4092 w_17_0 ∧
  MxxRuntime.tensorRuns w_16_0 w_18_0 w_19_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [21]) ((319/100)) (20) w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8184 ∧
  8184 ≤ 8184 ∧
  MxxRuntime.sliceMatrix w_22_0 0 1 0 8184 w_23_0 ∧
  outputs = (w_10_0, (w_23_0, ()))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat2 × ByteArray × (Fin 20072706095024 → Int) × Unit) (outputs : (Fin 20072706095024 → mxxMat3) × mxxMat3 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_3_0 : ByteArray := inputs.2.1
  let w_6_0 : Fin 20072706095024 → Int := inputs.2.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat3 := MxxRuntime.matrixMul w_2_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat1 := w_12_0
  let w_14_0 : mxxMat3 := MxxRuntime.matrixMul w_2_0 w_13_0
  let w_15_0 : mxxMat2 := (1 : mxxMat2)
  let w_16_0 : mxxMat2 := w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat3 := MxxRuntime.matrixMul w_2_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat3 := MxxRuntime.matrixSub w_14_0 w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat3 := MxxRuntime.matrixAdd w_20_0 w_21_0
  let w_23_0 := witness.w_23_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_3_0 w_6_0 w_1_0 w_2_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_10_1 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat2 × ByteArray × (Fin 20072706095024 → Int) × Unit) (outputs : (Fin 20072706095024 → mxxMat3) × mxxMat3 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_encrypt
