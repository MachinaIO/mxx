import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_evaluate

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
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 524288 8184 8184
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 524288 2 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat0 524288 8184 1

abbrev parallel_generatedRoot_3.constraints_0 (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ (8184 * (Int.ofNat i_0)) ∧
  (8184 * (Int.ofNat i_0)) < (8184 + (8184 * (Int.ofNat i_0))) ∧
  (8184 + (8184 * (Int.ofNat i_0))) ≤ 164275026681676416 ∧
  MxxRuntime.sliceMatrix w_0_0 0 2 (8184 * (Int.ofNat i_0)) (8184 + (8184 * (Int.ofNat i_0))) w_1_0 ∧
  outputs = w_1_0

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs
  ∃ (w_1_0 : mxxMat1),
    parallel_generatedRoot_3.constraints_0 i_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_5.constraints_0 (w_1_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  outputs = w_1_0

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs
  let w_1_0 : mxxMat2 := MxxRuntime.liftInteger w_0_0
  parallel_generatedRoot_5.constraints_0 w_1_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat3) (w_1_0 : mxxMat1) (w_2_0 : mxxMat2) (outputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_2_0, ())))

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) (outputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat2 := inputs.2.2.1
  parallel_generatedRoot_6.constraints_0 w_0_0 w_1_0 w_2_0 outputs

abbrev parallel_sequential_generatedRoot_15_12.constraints_1 (w_19_0 : Fin 20072706095024 → mxxMat1) (w_26_0 : Fin 20072706095024 → mxxMat2) (w_36_0 : mxxMat1) (w_46_0 : mxxMat2) (w_5_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_21_0 : Int) (w_27_0 : mxxMat2) (w_35_0 : mxxMat3) (w_37_0 : mxxMat1) (w_38_0 : mxxMat1) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat1) (w_47_0 : mxxMat2) (w_48_0 : mxxMat2) (w_49_0 : mxxMat2) (w_50_0 : mxxMat2) (w_53_0 : mxxMat2) (w_54_0 : mxxMat2) (w_55_0 : mxxMat2) (outputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_19_0 w_16_0 w_38_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 6 ∧
  6 = 6 ∧
  MxxRuntime.select w_13_0 [w_37_0, w_36_0, w_38_0, w_39_0, w_40_0, w_43_0] w_44_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_5_0 [w_37_0, w_44_0] w_45_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_26_0 w_21_0 w_49_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 6 ∧
  6 = 6 ∧
  MxxRuntime.select w_13_0 [w_47_0, w_46_0, w_27_0, w_48_0, w_50_0, w_53_0] w_54_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_5_0 [w_47_0, w_54_0] w_55_0 ∧
  outputs = (w_35_0, (w_45_0, (w_55_0, ())))

abbrev parallel_sequential_generatedRoot_15_12.constraints_0 (backend : MxxRuntime.BackendContext) (w_6_0 : mxxMat3) (w_8_0 : Fin 4817449462805760 → Int) (w_14_0 : Fin 20072706095024 → mxxMat3) (w_15_0 : Fin 4817449462805760 → Int) (w_19_0 : Fin 20072706095024 → mxxMat1) (w_20_0 : Fin 4817449462805760 → Int) (w_26_0 : Fin 20072706095024 → mxxMat2) (w_36_0 : mxxMat1) (w_46_0 : mxxMat2) (w_5_0 : Int) (w_7_0 : mxxMat3) (w_12_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_17_0 : mxxMat3) (w_18_0 : mxxMat3) (w_21_0 : Int) (w_22_0 : mxxMat1) (w_23_0 : mxxMat4) (w_25_0 : mxxMat3) (w_27_0 : mxxMat2) (w_29_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_37_0 : mxxMat1) (w_38_0 : mxxMat1) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat1) (w_47_0 : mxxMat2) (w_48_0 : mxxMat2) (w_49_0 : mxxMat2) (w_50_0 : mxxMat2) (w_53_0 : mxxMat2) (w_54_0 : mxxMat2) (w_55_0 : mxxMat2) (outputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) : Prop :=
  0 ≤ w_12_0 ∧
  w_12_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_12_0 w_13_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_12_0 w_16_0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_16_0 w_17_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_12_0 w_21_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4 4092 w_22_0 w_23_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_21_0 w_25_0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_26_0 w_16_0 w_27_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 6 ∧
  6 = 6 ∧
  MxxRuntime.select w_13_0 [w_7_0, w_6_0, w_17_0, w_18_0, w_29_0, w_33_0] w_34_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_5_0 [w_7_0, w_34_0] w_35_0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 20072706095024 ∧
  parallel_sequential_generatedRoot_15_12.constraints_1 w_19_0 w_26_0 w_36_0 w_46_0 w_5_0 w_13_0 w_16_0 w_21_0 w_27_0 w_35_0 w_37_0 w_38_0 w_39_0 w_40_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_49_0 w_50_0 w_53_0 w_54_0 w_55_0 outputs

def parallel_sequential_generatedRoot_15_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × mxxMat3 × (Fin 4817449462805760 → Int) × Int × (Fin 20072706095024 → mxxMat3) × (Fin 4817449462805760 → Int) × (Fin 20072706095024 → mxxMat1) × (Fin 4817449462805760 → Int) × (Fin 20072706095024 → mxxMat2) × mxxMat1 × mxxMat2 × Unit) (outputs : mxxMat3 × mxxMat1 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Int := inputs.1
  let w_6_0 : mxxMat3 := inputs.2.1
  let w_8_0 : Fin 4817449462805760 → Int := inputs.2.2.1
  let w_9_0 : Int := inputs.2.2.2.1
  let w_14_0 : Fin 20072706095024 → mxxMat3 := inputs.2.2.2.2.1
  let w_15_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 20072706095024 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.2.2.1
  let w_26_0 : Fin 20072706095024 → mxxMat2 := inputs.2.2.2.2.2.2.2.2.1
  let w_36_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat2 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_1)
  let w_2_0 : Int := 1
  let w_3_0 : Int := (w_1_0 - w_2_0)
  let w_4_0 : Bool := decide (w_0_0 ≤ w_3_0)
  let w_5_0 : Int := if w_4_0 then 1 else 0
  let w_7_0 : mxxMat3 := MxxRuntime.matrixSub w_6_0 w_6_0
  let w_10_0 : Int := params.«max_layer_width»
  let w_11_0 : Int := (w_9_0 * w_10_0)
  let w_12_0 : Int := (w_11_0 + w_0_0)
  ∃ (w_13_0 : Int),
    ∃ (w_16_0 : Int),
      ∃ (w_17_0 : mxxMat3),
        let w_18_0 : mxxMat3 := MxxRuntime.matrixSub w_6_0 w_17_0
        ∃ (w_21_0 : Int),
          ∃ (w_22_0 : mxxMat1),
            ∃ (w_23_0 : mxxMat4),
              let w_24_0 : mxxMat3 := MxxRuntime.matrixMul w_17_0 w_23_0
              ∃ (w_25_0 : mxxMat3),
                ∃ (w_27_0 : mxxMat2),
                  let w_28_0 : mxxMat3 := MxxRuntime.matrixMulScalarRight w_25_0 w_27_0
                  let w_29_0 : mxxMat3 := MxxRuntime.matrixAdd w_24_0 w_28_0
                  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_17_0 w_25_0
                  let w_31_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(2)] : mxxMat2)
                  let w_32_0 : mxxMat3 := MxxRuntime.matrixMulScalarRight w_29_0 w_31_0
                  let w_33_0 : mxxMat3 := MxxRuntime.matrixSub w_30_0 w_32_0
                  ∃ (w_34_0 : mxxMat3),
                    ∃ (w_35_0 : mxxMat3),
                      let w_37_0 : mxxMat1 := MxxRuntime.matrixSub w_36_0 w_36_0
                      ∃ (w_38_0 : mxxMat1),
                        let w_39_0 : mxxMat1 := MxxRuntime.matrixSub w_36_0 w_38_0
                        let w_40_0 : mxxMat1 := MxxRuntime.matrixMul w_38_0 w_23_0
                        let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_38_0 w_22_0
                        let w_42_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_40_0 w_31_0
                        let w_43_0 : mxxMat1 := MxxRuntime.matrixSub w_41_0 w_42_0
                        ∃ (w_44_0 : mxxMat1),
                          ∃ (w_45_0 : mxxMat1),
                            let w_47_0 : mxxMat2 := MxxRuntime.matrixSub w_46_0 w_46_0
                            let w_48_0 : mxxMat2 := MxxRuntime.matrixSub w_46_0 w_27_0
                            ∃ (w_49_0 : mxxMat2),
                              let w_50_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_27_0 w_49_0
                              let w_51_0 : mxxMat2 := MxxRuntime.matrixAdd w_27_0 w_49_0
                              let w_52_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_50_0 w_31_0
                              let w_53_0 : mxxMat2 := MxxRuntime.matrixSub w_51_0 w_52_0
                              ∃ (w_54_0 : mxxMat2),
                                ∃ (w_55_0 : mxxMat2),
                                  parallel_sequential_generatedRoot_15_12.constraints_0 backend w_6_0 w_8_0 w_14_0 w_15_0 w_19_0 w_20_0 w_26_0 w_36_0 w_46_0 w_5_0 w_7_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_21_0 w_22_0 w_23_0 w_25_0 w_27_0 w_29_0 w_33_0 w_34_0 w_35_0 w_37_0 w_38_0 w_39_0 w_40_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_49_0 w_50_0 w_53_0 w_54_0 w_55_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_15.Witness where
  w_2_0 : Int
  w_12_0 : Fin 20072706095024 → mxxMat3
  w_12_1 : Fin 20072706095024 → mxxMat1
  w_12_2 : Fin 20072706095024 → mxxMat2

abbrev sequential_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_5_0 : Fin 20072706095024 → mxxMat3) (w_7_0 : Fin 20072706095024 → mxxMat1) (w_9_0 : Fin 20072706095024 → mxxMat2) (w_0_0 : Fin 240 → Int) (w_3_0 : mxxMat3) (w_4_0 : Fin 4817449462805760 → Int) (w_6_0 : Fin 4817449462805760 → Int) (w_8_0 : Fin 4817449462805760 → Int) (w_10_0 : mxxMat1) (w_11_0 : mxxMat2) (w_1_0 : Int) (w_2_0 : Int) (w_12_0 : Fin 20072706095024 → mxxMat3) (w_12_1 : Fin 20072706095024 → mxxMat1) (w_12_2 : Fin 20072706095024 → mxxMat2) (outputs : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_sequential_generatedRoot_15_12 backend hashModel tape (path ++ [12, i.val]) params i_0 i (w_2_0, (w_3_0, (w_4_0, (w_1_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, (w_10_0, (w_11_0, ()))))))))))) ((w_12_0 i), ((w_12_1 i), ((w_12_2 i), ())))) ∧
  outputs = (w_12_0, (w_12_1, (w_12_2, ())))

abbrev sequential_generatedRoot_15.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × (Fin 240 → Int) × mxxMat3 × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × mxxMat1 × mxxMat2 × Unit) (outputs : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × Unit) (witness : sequential_generatedRoot_15.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_5_0 : Fin 20072706095024 → mxxMat3 := inputs.1
  let w_7_0 : Fin 20072706095024 → mxxMat1 := inputs.2.1
  let w_9_0 : Fin 20072706095024 → mxxMat2 := inputs.2.2.1
  let w_0_0 : Fin 240 → Int := inputs.2.2.2.1
  let w_3_0 : mxxMat3 := inputs.2.2.2.2.1
  let w_4_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.1
  let w_6_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_11_0 : mxxMat2 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 := witness.w_2_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_12_2 := witness.w_12_2
  sequential_generatedRoot_15.constraints_0 backend hashModel tape path params i_0 w_5_0 w_7_0 w_9_0 w_0_0 w_3_0 w_4_0 w_6_0 w_8_0 w_10_0 w_11_0 w_1_0 w_2_0 w_12_0 w_12_1 w_12_2 outputs

def sequential_generatedRoot_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × (Fin 240 → Int) × mxxMat3 × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × mxxMat1 × mxxMat2 × Unit) (outputs : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_15.Witness,
    sequential_generatedRoot_15.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat0
  w_3_0 : Fin 20072706095024 → mxxMat1
  w_5_0 : Fin 20072706095024 → mxxMat2
  w_6_0 : Fin 20072706095024 → mxxMat3
  w_6_1 : Fin 20072706095024 → mxxMat1
  w_6_2 : Fin 20072706095024 → mxxMat2
  w_12_0 : mxxMat1
  w_13_0 : mxxMat1
  w_15_0 : Fin 20072706095024 → mxxMat3
  w_15_1 : Fin 20072706095024 → mxxMat1
  w_15_2 : Fin 20072706095024 → mxxMat2
  w_17_0 : Int
  w_18_0 : mxxMat1
  w_21_0 : mxxMat5
  w_22_0 : mxxMat6
  w_24_0 : mxxMat3

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 20072706095024 → mxxMat3) (w_1_0 : ByteArray) (w_4_0 : Fin 20072706095024 → Int) (w_7_0 : Fin 240 → Int) (w_8_0 : mxxMat3) (w_9_0 : Fin 4817449462805760 → Int) (w_10_0 : Fin 4817449462805760 → Int) (w_11_0 : Fin 4817449462805760 → Int) (w_16_0 : Fin 1 → Int) (w_2_0 : mxxMat0) (w_3_0 : Fin 20072706095024 → mxxMat1) (w_5_0 : Fin 20072706095024 → mxxMat2) (w_6_0 : Fin 20072706095024 → mxxMat3) (w_6_1 : Fin 20072706095024 → mxxMat1) (w_6_2 : Fin 20072706095024 → mxxMat2) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat2) (w_15_0 : Fin 20072706095024 → mxxMat3) (w_15_1 : Fin 20072706095024 → mxxMat1) (w_15_2 : Fin 20072706095024 → mxxMat2) (w_17_0 : Int) (w_18_0 : mxxMat1) (w_19_0 : mxxMat2) (w_20_0 : mxxMat2) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat5) (w_24_0 : mxxMat3) (w_25_0 : mxxMat2) (outputs : mxxMat5 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([98, 111, 111, 108, 101, 97, 110, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 115]) ([]) (w_1_0) w_2_0 ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_3 backend hashModel tape (path ++ [3, i.val]) params i w_2_0 (w_3_0 i)) ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i (w_4_0 i) (w_5_0 i)) ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i ((w_0_0 i), ((w_3_0 i), ((w_5_0 i), ()))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ())))) ∧
  MxxRuntime.hashSample (hashModel) ([98, 111, 111, 108, 101, 97, 110, 45, 98, 103, 103, 47, 111, 110, 101]) ([]) (w_1_0) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 8184 ∧
  8184 ≤ 8184 ∧
  MxxRuntime.sliceMatrix w_12_0 0 2 0 8184 w_13_0 ∧
  0 ≤ (params.«depth») ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : (Fin 20072706095024 → mxxMat3) × (Fin 20072706095024 → mxxMat1) × (Fin 20072706095024 → mxxMat2) × Unit) => sequential_generatedRoot_15 backend hashModel tape (path ++ [15, i]) params i (current.1, (current.2.1, (current.2.2.1, (w_7_0, (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_13_0, (w_14_0, ())))))))))) next) (Int.toNat (params.«depth»)) (w_6_0, (w_6_1, (w_6_2, ()))) (w_15_0, (w_15_1, (w_15_2, ()))) ∧
  MxxRuntime.familyGetStatic w_16_0 (0) w_17_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_15_1 w_17_0 w_18_0 ∧
  (2) ≠ 0 ∧
  MxxRuntime.concatRows w_19_0 w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4 4092 w_21_0 w_22_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_17_0 w_24_0 ∧
  outputs = (w_23_0, (w_25_0, ()))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 20072706095024 → mxxMat3) × ByteArray × (Fin 20072706095024 → Int) × (Fin 240 → Int) × mxxMat3 × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 1 → Int) × Unit) (outputs : mxxMat5 × mxxMat2 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 20072706095024 → mxxMat3 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : Fin 20072706095024 → Int := inputs.2.2.1
  let w_7_0 : Fin 240 → Int := inputs.2.2.2.1
  let w_8_0 : mxxMat3 := inputs.2.2.2.2.1
  let w_9_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat2 := (1 : mxxMat2)
  let w_15_0 := witness.w_15_0
  let w_15_1 := witness.w_15_1
  let w_15_2 := witness.w_15_2
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat2 := (0 : mxxMat2)
  let w_20_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [((Int.fdiv (mxxNat0) (2)))] : mxxMat2)
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat5 := MxxRuntime.matrixMul w_18_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat2 := MxxRuntime.matrixMul w_24_0 w_22_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_4_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_16_0 w_2_0 w_3_0 w_5_0 w_6_0 w_6_1 w_6_2 w_12_0 w_13_0 w_14_0 w_15_0 w_15_1 w_15_2 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 20072706095024 → mxxMat3) × ByteArray × (Fin 20072706095024 → Int) × (Fin 240 → Int) × mxxMat3 × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 1 → Int) × Unit) (outputs : mxxMat5 × mxxMat2 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_evaluate
