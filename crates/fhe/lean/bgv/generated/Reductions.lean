import Scheme
import Sim_0_encrypt
import Sim_0_setup
import Sim_1_encrypt
import Sim_1_setup
import Sim_2_encrypt
import Sim_2_setup

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

/-! The simulators of the reductions of `GeneratedScheme.BoundedSecurityClaim` as
computations, and IND-CPA security with them, for its proof. Generated deterministically;
never edit by hand. -/

namespace GeneratedScheme

open MxxSecurity MxxFhe

def Sim_0_encrypt_params : Sim_0_encrypt.Params := { «unit» := () }
def Sim_0_setup_params : Sim_0_setup.Params := { «unit» := () }
def Sim_1_encrypt_params : Sim_1_encrypt.Params := { «unit» := () }
def Sim_1_setup_params : Sim_1_setup.Params := { «unit» := () }
def Sim_2_encrypt_params : Sim_2_encrypt.Params := { «unit» := () }
def Sim_2_setup_params : Sim_2_setup.Params := { «unit» := () }

noncomputable def simulator_0 : MxxFhe.Simulator scheme (MxxSecurity.AssumptionId.Challenge MxxSecurity.AssumptionId.rlweKeySwitchCircular params_0) where
  State := Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Unit
  setup challenge := do
    let state ← Sim_0_setup.generatedRoot.comp Sim_0_setup_params ((challenge.2.1), ((challenge.2.2), ((challenge.1.1), ((challenge.1.2), ()))))
    pure (state.2.1, state.1, state)
  encrypt state m := do
    let outputs ← Sim_0_encrypt.generatedRoot.comp Sim_0_encrypt_params ((state.2.1), ((m), ()))
    pure (.ct0 outputs)

noncomputable def simulator_1 : MxxFhe.Simulator scheme (MxxSecurity.AssumptionId.Challenge MxxSecurity.AssumptionId.decisionRlwe params_1) where
  State := Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Unit
  setup challenge := do
    let state ← Sim_1_setup.generatedRoot.comp Sim_1_setup_params ((challenge.1), ((challenge.2), ()))
    pure (state.2.1, state.1, state)
  encrypt state m := do
    let outputs ← Sim_1_encrypt.generatedRoot.comp Sim_1_encrypt_params ((state.2.1), ((m), ()))
    pure (.ct0 outputs)

noncomputable def simulator_2 : MxxFhe.Simulator scheme (MxxSecurity.AssumptionId.Challenge MxxSecurity.AssumptionId.decisionRlwe params_2) where
  State := Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 2 × Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Unit
  setup challenge := do
    let state ← Sim_2_setup.generatedRoot.comp Sim_2_setup_params ((challenge.2), ((challenge.1), ()))
    pure (state.2.2.1, state.2.1, state)
  encrypt state m := do
    let outputs ← Sim_2_encrypt.generatedRoot.comp Sim_2_encrypt_params ((state.1), ((m), ()))
    pure (.ct0 outputs)

/-- The reductions. -/
noncomputable def uses : List (MxxFhe.AssumptionUse scheme) :=
  [⟨MxxSecurity.AssumptionId.rlweKeySwitchCircular, params_0, simulator_0⟩,
    ⟨MxxSecurity.AssumptionId.decisionRlwe, params_1, simulator_1⟩,
    ⟨MxxSecurity.AssumptionId.decisionRlwe, params_2, simulator_2⟩]

/-- IND-CPA security with these reductions, without a statistical term. -/
def SimulatedSecurity : Prop :=
  MxxFhe.Secure scheme uses 0

end GeneratedScheme
