import Backend
import Stage_payload
import Stage_mask
import Stage_preprocess
import Stage_encrypt
import Stage_online
import Stage_decode
import Requirement_0
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : ByteArray
  input_1 : Int
  input_2 : Int
  input_3 : Fin 17 → Int
  input_4 : Fin 17 → Int
  input_5 : Fin 17 → Int
  input_6 : Fin 14 → Int
  input_7 : Fin 14 → Int
  input_8 : Fin 14 → Int
  input_9 : Fin 14 → Int
  input_10 : Fin 14 → Int
  input_11 : Fin 14 → Int
  input_12 : Fin 14 → Int
  input_13 : Fin 14 → Int
  input_14 : Fin 14 → Int
  input_15 : Fin 14 → Int

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  ((1 : Int) ≤ external.input_1 ∧ external.input_1 ≤ (1 : Int)) ∧
  ((0 : Int) ≤ external.input_2 ∧ external.input_2 ≤ (0 : Int)) ∧
  (∀ contract_i_0 : Fin 17, ((0 : Int) ≤ (external.input_3 contract_i_0) ∧ (external.input_3 contract_i_0) ≤ (31 : Int))) ∧
  (∀ contract_i_0 : Fin 17, ((-1 : Int) ≤ (external.input_4 contract_i_0) ∧ (external.input_4 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 17, ((0 : Int) ≤ (external.input_5 contract_i_0) ∧ (external.input_5 contract_i_0) ≤ (31 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_6 contract_i_0) ∧ (external.input_6 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_7 contract_i_0) ∧ (external.input_7 contract_i_0) ≤ (0 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_8 contract_i_0) ∧ (external.input_8 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_9 contract_i_0) ∧ (external.input_9 contract_i_0) ≤ (0 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_10 contract_i_0) ∧ (external.input_10 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_11 contract_i_0) ∧ (external.input_11 contract_i_0) ≤ (0 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_12 contract_i_0) ∧ (external.input_12 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_13 contract_i_0) ∧ (external.input_13 contract_i_0) ≤ (0 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_14 contract_i_0) ∧ (external.input_14 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 14, ((0 : Int) ≤ (external.input_15 contract_i_0) ∧ (external.input_15 contract_i_0) ≤ (0 : Int)))

structure Execution where
  «stage_0» : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1
  «stage_1» : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1
  «stage_2» : (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1099511626049 32 1 1) × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 14 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit
  «stage_3» : Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12 × (Fin 17 → Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) × (Fin 17 → Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 14) × (Fin 17 → Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 1) × (Fin 16 → Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) × (Fin 16 → Mxx.Primitives.ExactMatrix 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 32 1 12) × (Fin 16 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 16 → Mxx.Primitives.ExactMatrix 1766847023753256444127058376275403912430991675449840265546213289008802881 32 1 10) × (Fin 16 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 16 → Mxx.Primitives.ExactMatrix 1606938016486074846153923224048467814823508072370602761938369 32 1 8) × (Fin 16 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 16 → Mxx.Primitives.ExactMatrix 1461501619046042689013583379439969400432371472961 32 1 6) × (Fin 16 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 16 → Mxx.Primitives.ExactMatrix 1329227985420794846003957960673218177 32 1 4) × (Fin 16 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × (Fin 16 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 2) × Unit
  «stage_4» : Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Mxx.Primitives.ExactMatrix 1099511626049 32 1 1 × Unit
  «stage_5» : Int
  «requirement_0» : Bool
  «ideal» : Int

def stage_0_params : Stage_payload.Params := { «unit» := () }

def stage_1_params : Stage_mask.Params := { «unit» := () }

def stage_2_params : Stage_preprocess.Params := { «unit» := () }

def stage_3_params : Stage_encrypt.Params := { «unit» := () }

def stage_4_params : Stage_online.Params := { «unit» := () }

def stage_5_params : Stage_decode.Params := { «unit» := () }

def requirement_0_params : Requirement_0.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_payload.generatedRoot stage_0_params (()) execution.«stage_0» ∧
  Stage_mask.generatedRoot stage_1_params (()) execution.«stage_1» ∧
  Stage_preprocess.generatedRoot Backend.backend hashModel stage_2_params ((external.input_0, execution.«stage_0», external.input_6, external.input_7, external.input_8, external.input_9, external.input_10, external.input_11, external.input_12, external.input_13, external.input_14, external.input_15, ())) execution.«stage_2» ∧
  Stage_encrypt.generatedRoot Backend.backend hashModel stage_3_params ((execution.«stage_1», external.input_0, execution.«stage_0», external.input_1, external.input_2, external.input_4, external.input_5, ())) execution.«stage_3» ∧
  Stage_online.generatedRoot Backend.backend hashModel stage_4_params ((execution.«stage_2».1, execution.«stage_2».2.1, execution.«stage_2».2.2.1, execution.«stage_2».2.2.2.1, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_0, execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_3».1, execution.«stage_3».2.2.2.2.1, execution.«stage_3».2.1, execution.«stage_2».2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ())) execution.«stage_4» ∧
  Stage_decode.generatedRoot stage_5_params ((execution.«stage_4».2.2.2.2.2.1, execution.«stage_1», execution.«stage_4».2.1, execution.«stage_4».2.2.2.1, ())) execution.«stage_5» ∧
  Requirement_0.generatedRoot requirement_0_params ((external.input_4, external.input_5, external.input_3, external.input_6, external.input_8, external.input_10, external.input_12, external.input_14, ())) execution.«requirement_0» ∧
  Ideal.generatedRoot ideal_params ((external.input_6, external.input_3, external.input_8, external.input_10, external.input_12, external.input_14, ())) execution.«ideal» ∧
  execution.«requirement_0» = true

/-- Every execution's endpoint equals the ideal one. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    execution.«stage_5» = execution.«ideal»

end GeneratedClaim
