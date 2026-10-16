import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decode

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489
local notation "mxxNat1" => 72341562789056444115684979687416806180395197053951206217914228266014512572833794748552875371413453242085860536342419038795864559652387809846630825946845083145910068784615984026523621685186503403424255574409217

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 295 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 32768 30 1
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 60
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 45
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 45
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 30
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 1

abbrev sequential_generatedRoot_11.constraints_0 (i_0 : Nat) (w_2_0 : Fin 256 → mxxMat0) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 256 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (4353), (4609), (4865), (5121), (5377), (5633), (5889), (6145), (6401), (6657), (6913), (7169), (7425), (7681), (7937), (8193), (8449), (8705), (8961), (9217), (9473), (9729), (9985), (10241), (10497), (10753), (11009), (11265), (11521), (11777), (12033), (12289), (12545), (12801), (13057), (13313), (13569), (13825), (14081), (14337), (14593), (14849), (15105), (15361), (15617), (15873), (16129), (16385), (16641), (16897), (17153), (17409), (17665), (17921), (18177), (18433), (18689), (18945), (19201), (19457), (19713), (19969), (20225), (20481), (20737), (20993), (21249), (21505), (21761), (22017), (22273), (22529), (22785), (23041), (23297), (23553), (23809), (24065), (24321), (24577), (24833), (25089), (25345), (25601), (25857), (26113), (26369), (26625), (26881), (27137), (27393), (27649), (27905), (28161), (28417), (28673), (28929), (29185), (29441), (29697), (29953), (30209), (30465), (30721), (30977), (31233), (31489), (31745), (32001), (32257), (32513), (32769), (33025), (33281), (33537), (33793), (34049), (34305), (34561), (34817), (35073), (35329), (35585), (35841), (36097), (36353), (36609), (36865), (37121), (37377), (37633), (37889), (38145), (38401), (38657), (38913), (39169), (39425), (39681), (39937), (40193), (40449), (40705), (40961), (41217), (41473), (41729), (41985), (42241), (42497), (42753), (43009), (43265), (43521), (43777), (44033), (44289), (44545), (44801), (45057), (45313), (45569), (45825), (46081), (46337), (46593), (46849), (47105), (47361), (47617), (47873), (48129), (48385), (48641), (48897), (49153), (49409), (49665), (49921), (50177), (50433), (50689), (50945), (51201), (51457), (51713), (51969), (52225), (52481), (52737), (52993), (53249), (53505), (53761), (54017), (54273), (54529), (54785), (55041), (55297), (55553), (55809), (56065), (56321), (56577), (56833), (57089), (57345), (57601), (57857), (58113), (58369), (58625), (58881), (59137), (59393), (59649), (59905), (60161), (60417), (60673), (60929), (61185), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])) w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_generatedRoot_11 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat2 × (Fin 256 → mxxMat0) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat2 := inputs.2.1
  let w_2_0 : Fin 256 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_generatedRoot_11.constraints_0 i_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_generatedRoot_28.constraints_0 (i_0 : Nat) (w_1_0 : mxxMat3) (w_3_0 : Fin 256 → mxxMat4) (w_2_0 : mxxMat3) (w_4_0 : Int) (w_5_0 : mxxMat4) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 256 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (4353), (4609), (4865), (5121), (5377), (5633), (5889), (6145), (6401), (6657), (6913), (7169), (7425), (7681), (7937), (8193), (8449), (8705), (8961), (9217), (9473), (9729), (9985), (10241), (10497), (10753), (11009), (11265), (11521), (11777), (12033), (12289), (12545), (12801), (13057), (13313), (13569), (13825), (14081), (14337), (14593), (14849), (15105), (15361), (15617), (15873), (16129), (16385), (16641), (16897), (17153), (17409), (17665), (17921), (18177), (18433), (18689), (18945), (19201), (19457), (19713), (19969), (20225), (20481), (20737), (20993), (21249), (21505), (21761), (22017), (22273), (22529), (22785), (23041), (23297), (23553), (23809), (24065), (24321), (24577), (24833), (25089), (25345), (25601), (25857), (26113), (26369), (26625), (26881), (27137), (27393), (27649), (27905), (28161), (28417), (28673), (28929), (29185), (29441), (29697), (29953), (30209), (30465), (30721), (30977), (31233), (31489), (31745), (32001), (32257), (32513), (32769), (33025), (33281), (33537), (33793), (34049), (34305), (34561), (34817), (35073), (35329), (35585), (35841), (36097), (36353), (36609), (36865), (37121), (37377), (37633), (37889), (38145), (38401), (38657), (38913), (39169), (39425), (39681), (39937), (40193), (40449), (40705), (40961), (41217), (41473), (41729), (41985), (42241), (42497), (42753), (43009), (43265), (43521), (43777), (44033), (44289), (44545), (44801), (45057), (45313), (45569), (45825), (46081), (46337), (46593), (46849), (47105), (47361), (47617), (47873), (48129), (48385), (48641), (48897), (49153), (49409), (49665), (49921), (50177), (50433), (50689), (50945), (51201), (51457), (51713), (51969), (52225), (52481), (52737), (52993), (53249), (53505), (53761), (54017), (54273), (54529), (54785), (55041), (55297), (55553), (55809), (56065), (56321), (56577), (56833), (57089), (57345), (57601), (57857), (58113), (58369), (58625), (58881), (59137), (59393), (59649), (59905), (60161), (60417), (60673), (60929), (61185), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])) w_1_0 w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  outputs = w_7_0

def sequential_generatedRoot_28 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat3 × (Fin 256 → mxxMat4) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat3 := inputs.2.1
  let w_3_0 : Fin 256 → mxxMat4 := inputs.2.2.1
  ∃ (w_2_0 : mxxMat3),
    let w_4_0 : Int := (Int.ofNat i_0)
    ∃ (w_5_0 : mxxMat4),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_generatedRoot_28.constraints_0 i_0 w_1_0 w_3_0 w_2_0 w_4_0 w_5_0 w_7_0 outputs

abbrev sequential_sequential_generatedRoot_37_8.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : Fin 45 → mxxMat3) (w_4_0 : Fin 45 → mxxMat3) (w_7_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat5) (w_11_0 : mxxMat5) (w_12_0 : mxxMat5) (w_16_0 : mxxMat6) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 45 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 45 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_1)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_7_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_1)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_7_0) w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 256 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (4353), (4609), (4865), (5121), (5377), (5633), (5889), (6145), (6401), (6657), (6913), (7169), (7425), (7681), (7937), (8193), (8449), (8705), (8961), (9217), (9473), (9729), (9985), (10241), (10497), (10753), (11009), (11265), (11521), (11777), (12033), (12289), (12545), (12801), (13057), (13313), (13569), (13825), (14081), (14337), (14593), (14849), (15105), (15361), (15617), (15873), (16129), (16385), (16641), (16897), (17153), (17409), (17665), (17921), (18177), (18433), (18689), (18945), (19201), (19457), (19713), (19969), (20225), (20481), (20737), (20993), (21249), (21505), (21761), (22017), (22273), (22529), (22785), (23041), (23297), (23553), (23809), (24065), (24321), (24577), (24833), (25089), (25345), (25601), (25857), (26113), (26369), (26625), (26881), (27137), (27393), (27649), (27905), (28161), (28417), (28673), (28929), (29185), (29441), (29697), (29953), (30209), (30465), (30721), (30977), (31233), (31489), (31745), (32001), (32257), (32513), (32769), (33025), (33281), (33537), (33793), (34049), (34305), (34561), (34817), (35073), (35329), (35585), (35841), (36097), (36353), (36609), (36865), (37121), (37377), (37633), (37889), (38145), (38401), (38657), (38913), (39169), (39425), (39681), (39937), (40193), (40449), (40705), (40961), (41217), (41473), (41729), (41985), (42241), (42497), (42753), (43009), (43265), (43521), (43777), (44033), (44289), (44545), (44801), (45057), (45313), (45569), (45825), (46081), (46337), (46593), (46849), (47105), (47361), (47617), (47873), (48129), (48385), (48641), (48897), (49153), (49409), (49665), (49921), (50177), (50433), (50689), (50945), (51201), (51457), (51713), (51969), (52225), (52481), (52737), (52993), (53249), (53505), (53761), (54017), (54273), (54529), (54785), (55041), (55297), (55553), (55809), (56065), (56321), (56577), (56833), (57089), (57345), (57601), (57857), (58113), (58369), (58625), (58881), (59137), (59393), (59649), (59905), (60161), (60417), (60673), (60929), (61185), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])) w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 45 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_17_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_37_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 45 → mxxMat3) × (Fin 45 → mxxMat3) × ByteArray × mxxMat5 × mxxMat7 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 45 → mxxMat3 := inputs.2.1
  let w_4_0 : Fin 45 → mxxMat3 := inputs.2.2.1
  let w_7_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : mxxMat5 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat7 := inputs.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat3),
    ∃ (w_5_0 : mxxMat3),
      ∃ (w_6_0 : mxxMat5),
        ∃ (w_8_0 : mxxMat3),
          ∃ (w_9_0 : mxxMat3),
            ∃ (w_10_0 : mxxMat5),
              let w_11_0 : mxxMat5 := MxxRuntime.matrixSub w_6_0 w_10_0
              ∃ (w_12_0 : mxxMat5),
                let w_14_0 : mxxMat5 := MxxRuntime.matrixAdd w_12_0 w_13_0
                let w_16_0 : mxxMat6 := MxxRuntime.matrixMul w_14_0 w_15_0
                ∃ (w_17_0 : mxxMat1),
                  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
                  sequential_sequential_generatedRoot_37_8.constraints_0 hashModel i_0 i_1 w_1_0 w_4_0 w_7_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_37.Witness where
  w_7_0 : mxxMat7
  w_8_0 : mxxMat1

abbrev sequential_generatedRoot_37.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : Fin 45 → mxxMat3) (w_2_0 : Fin 45 → mxxMat3) (w_3_0 : ByteArray) (w_4_0 : mxxMat5) (w_5_0 : Fin 256 → mxxMat7) (w_6_0 : Int) (w_7_0 : mxxMat7) (w_8_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  0 ≤ (45) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_37_8 backend hashModel tape (path ++ [8, i]) params i_0 i (current, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_7_0, ())))))) next) (Int.toNat (45)) w_0_0 w_8_0 ∧
  outputs = w_8_0

abbrev sequential_generatedRoot_37.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 45 → mxxMat3) × (Fin 45 → mxxMat3) × ByteArray × mxxMat5 × (Fin 256 → mxxMat7) × Unit) (outputs : mxxMat1) (witness : sequential_generatedRoot_37.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 45 → mxxMat3 := inputs.2.1
  let w_2_0 : Fin 45 → mxxMat3 := inputs.2.2.1
  let w_3_0 : ByteArray := inputs.2.2.2.1
  let w_4_0 : mxxMat5 := inputs.2.2.2.2.1
  let w_5_0 : Fin 256 → mxxMat7 := inputs.2.2.2.2.2.1
  let w_6_0 : Int := (Int.ofNat i_0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  sequential_generatedRoot_37.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 outputs

def sequential_generatedRoot_37 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 45 → mxxMat3) × (Fin 45 → mxxMat3) × ByteArray × mxxMat5 × (Fin 256 → mxxMat7) × Unit) (outputs : mxxMat1) : Prop :=
  ∃ witness : sequential_generatedRoot_37.Witness,
    sequential_generatedRoot_37.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat2
  w_7_0 : mxxMat8
  w_8_0 : mxxMat2
  w_11_0 : mxxMat1
  w_13_0 : mxxMat9
  w_14_0 : mxxMat3
  w_15_0 : mxxMat10
  w_17_0 : mxxMat3
  w_19_0 : mxxMat8
  w_20_0 : mxxMat9
  w_21_0 : mxxMat3
  w_22_0 : mxxMat10
  w_24_0 : mxxMat3
  w_28_0 : mxxMat1
  w_34_0 : mxxMat3
  w_35_0 : mxxMat5
  w_37_0 : mxxMat1
  w_41_0 : mxxMat5
  w_44_0 : mxxMat5
  w_50_0 : Fin 32768 → Int

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : ByteArray) (w_31_0 : Fin 45 → mxxMat3) (w_32_0 : Fin 45 → mxxMat3) (w_36_0 : Fin 256 → mxxMat7) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (w_43_0 : mxxMat3) (w_30_0 : mxxMat1) (w_35_0 : mxxMat5) (w_37_0 : mxxMat1) (w_41_0 : mxxMat5) (w_44_0 : mxxMat5) (w_49_0 : mxxMat1) (w_50_0 : Fin 32768 → Int) (outputs : Fin 32768 → Int) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_37 backend hashModel tape (path ++ [37, i]) params i (current, (w_31_0, (w_32_0, (w_1_0, (w_35_0, (w_36_0, ())))))) next) (Int.toNat (256)) w_30_0 w_37_0 ∧
  MxxRuntime.concatColumns w_39_0 w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_42_0 w_43_0 w_44_0 ∧
  (32768) = 32768 ∧
  (∀ k : Fin 32768, MxxRuntime.thresholdDecode (2) (32768) (k.val : Int) w_49_0 (w_50_0 k)) ∧
  outputs = w_50_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : ByteArray) (w_5_0 : mxxMat8) (w_10_0 : Fin 256 → mxxMat0) (w_27_0 : Fin 256 → mxxMat4) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat2) (w_7_0 : mxxMat8) (w_8_0 : mxxMat2) (w_9_0 : mxxMat2) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat9) (w_14_0 : mxxMat3) (w_15_0 : mxxMat10) (w_16_0 : mxxMat10) (w_17_0 : mxxMat3) (w_19_0 : mxxMat8) (w_20_0 : mxxMat9) (w_21_0 : mxxMat3) (w_22_0 : mxxMat10) (w_23_0 : mxxMat10) (w_24_0 : mxxMat3) (w_26_0 : mxxMat3) (w_28_0 : mxxMat1) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat5) : Prop :=
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_1_0) w_7_0 ∧
  MxxRuntime.modulusSwitchRuns w_7_0 w_8_0 ∧
  0 ≤ (256) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_11 backend hashModel tape (path ++ [11, i]) params i (current, (w_9_0, (w_10_0, ()))) next) (Int.toNat (256)) w_4_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 295 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 0 1 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.modulusSwitchRuns w_16_0 w_17_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_1_0) w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 295 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 1 w_20_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_21_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  0 ≤ (256) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_28 backend hashModel tape (path ++ [28, i]) params i (current, (w_26_0, (w_27_0, ()))) next) (Int.toNat (256)) w_12_0 w_28_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_34_0 ∧
  MxxRuntime.concatColumns w_33_0 w_34_0 w_35_0 ∧
  0 ≤ (256)

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × ByteArray × mxxMat8 × (Fin 256 → mxxMat0) × (Fin 256 → mxxMat4) × (Fin 45 → mxxMat3) × (Fin 45 → mxxMat3) × (Fin 256 → mxxMat7) × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat11 × Unit) (outputs : Fin 32768 → Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.1
  let w_10_0 : Fin 256 → mxxMat0 := inputs.2.2.2.1
  let w_27_0 : Fin 256 → mxxMat4 := inputs.2.2.2.2.1
  let w_31_0 : Fin 45 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_32_0 : Fin 45 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_36_0 : Fin 256 → mxxMat7 := inputs.2.2.2.2.2.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.2.2.2.2.2.1
  let w_40_0 : mxxMat3 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_42_0 : mxxMat3 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : mxxMat3 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat11 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := MxxRuntime.matrixSub w_0_0 w_2_0
  let w_4_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat2 := MxxRuntime.matrixSub w_6_0 w_8_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := (0 : mxxMat1)
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_13_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat3 := w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat10 := MxxRuntime.matrixMulScalarLeft w_20_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat3 := w_24_0
  let w_26_0 : mxxMat3 := MxxRuntime.matrixSub w_18_0 w_25_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat1 := MxxRuntime.matrixAdd w_11_0 w_28_0
  let w_30_0 : mxxMat1 := (0 : mxxMat1)
  let w_33_0 : mxxMat3 := (0 : mxxMat3)
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat1 := MxxRuntime.matrixAdd w_29_0 w_37_0
  let w_41_0 := witness.w_41_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : mxxMat5 := MxxRuntime.matrixSub w_41_0 w_44_0
  let w_47_0 : mxxMat1 := MxxRuntime.matrixMul w_45_0 w_46_0
  let w_48_0 : mxxMat1 := MxxRuntime.matrixAdd w_38_0 w_47_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixSub w_3_0 w_48_0
  let w_50_0 := witness.w_50_0
  generatedRoot.constraints_0 backend hashModel tape path params w_1_0 w_5_0 w_10_0 w_27_0 w_2_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_28_0 w_33_0 w_34_0 w_35_0 ∧
    generatedRoot.constraints_1 backend hashModel tape path params w_1_0 w_31_0 w_32_0 w_36_0 w_39_0 w_40_0 w_42_0 w_43_0 w_30_0 w_35_0 w_37_0 w_41_0 w_44_0 w_49_0 w_50_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × ByteArray × mxxMat8 × (Fin 256 → mxxMat0) × (Fin 256 → mxxMat4) × (Fin 45 → mxxMat3) × (Fin 45 → mxxMat3) × (Fin 256 → mxxMat7) × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat11 × Unit) (outputs : Fin 32768 → Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_decode
