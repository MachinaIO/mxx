import Dio2Online0AssemblyP
import Dio2Online1State

/-!
# Concrete decoding noise inequalities

Imported root errors and helpers have bound one. Exact switching and selector preprocessing
have bound zero. A selector cell has bound `671095193600`; two binary buckets have bounds
`1342190387200` and `17592359185285120000`. The state LUT input uses the cell bound, whereas
the sigma, fresh, and smudge readouts use the second bucket bound, not the imported unit bound.

The largest step-zero decoding left side is
`680571431906852705552444235198697479435629248211202433992` (188.794720 bits), below every
`Q_0 / q_p` (approximately 200 bits). The automorphism target dominates this bound.

At step one, fresh-only root errors give a 109.219295-bit left side. The separate algebraic
expansion `8 * Bbeta * Bzero + Bone + Bfresh`, when all four budgets are at most one, is at
most ten. Even using ten for both the imported roots and hint gives a 112.541223-bit left side.
`online1_output_noise_from_transition` keeps this expansion explicit rather than resetting
the next input budget to one. `expanded_error_bound` bounds the actual error expression
returned by `step_expand_rel`, and `expanded_input_bound` evaluates its unit-piece case.
`Dio2StepTransition.step_refresh_rel` retains only fresh errors; `step_expand_rel` additionally
requires noisy forms for the new key columns. The latter forms and their unit budgets are not
established by these arithmetic theorems, nor are they identified with old target errors.

Only decoder `hnoise` hypotheses are discharged here. Encoding relations, norm hypotheses,
the actual fresh support bound, and the smudge residue equations and bound four remain separate
obligations. In particular, numerical acceptance does not prove a smudge residue is small.
-/

set_option maxHeartbeats 8000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

namespace Noise

/-- The selector-cell budget for unit imported labels and a seed of the given budget. -/
def selectorCellBound (seed : Nat) : Nat :=
  (25 + 25) * 8 * ((25 + 25) * 8 * 1 * 2048 + 8 * 1 * seed) * 2048 + 8 * 1 * 0

/-- The two-cell sum forming one binary selector bucket. -/
def selectorBucketBound (seed : Nat) : Nat := 2 * selectorCellBound seed

/-- The concrete second-bucket bound used by all three step-zero readout families. -/
def readoutInputBound : Nat := selectorBucketBound (selectorBucketBound 1)

/-- The maximum target-row budget, attained by every nonidentity automorphism target. -/
def targetBound : Nat := 830775673711943892691742738445346028568707072000000

/-- The maximum full step-zero left side, including the strict cutoff's factor two. -/
def step0Left : Nat :=
  680571431906852705552444235198697479435629248211202433992

/-- Imported unit rows, exact preprocessing, and the derived PRF input budgets. -/
structure Inputs0 {backend : BackendContext} {model : HashModel}
    {params : W_Stage_online_0.Params}
    {context : W_Stage_online_0.generatedRoot.Ctx backend model params}
    {mask : ErrorPoly 8} (inputs : Online0Inputs context mask) : Prop where
  /-- The sigma PRF output has the second-bucket budget. -/
  sigma : inputs.σB = readoutInputBound
  /-- Readout helpers are imported unit rows. -/
  helper : inputs.Bh = 1
  /-- Sampler helpers are imported unit rows. -/
  samplerHelper : inputs.BhS = 1
  /-- Sigma readout switching is exact. -/
  sigmaReadout : inputs.σBBr = 0
  /-- Sigma sampler switching is exact. -/
  sigmaSampler : inputs.σBBS = 0
  /-- The public unit is imported with unit error. -/
  one : inputs.oneB = 1
  /-- The private unit is imported with unit error. -/
  privateUnit : inputs.privB = 1
  /-- The state LUT input is a selector cell, not a fresh imported row. -/
  state : inputs.stB = selectorCellBound 1
  /-- State helpers have unit error. -/
  stateHelper : inputs.stBh = 1
  /-- State switching is exact. -/
  stateSwitch : inputs.stBB = 0
  /-- Imported labels have unit error. -/
  label : inputs.Ba = 1
  /-- The imported state label has unit error. -/
  stateLabel : inputs.Bo = 1
  /-- The seven automorphism helpers have unit error. -/
  autHelpers : inputs.a3Bh = 1 ∧ inputs.a5Bh = 1 ∧ inputs.a7Bh = 1 ∧
    inputs.a9Bh = 1 ∧ inputs.a11Bh = 1 ∧ inputs.a13Bh = 1 ∧ inputs.a15Bh = 1
  /-- All seven automorphism switches are exact. -/
  autSwitches : inputs.a3BB = 0 ∧ inputs.a5BB = 0 ∧ inputs.a7BB = 0 ∧
    inputs.a9BB = 0 ∧ inputs.a11BB = 0 ∧ inputs.a13BB = 0 ∧ inputs.a15BB = 0
  /-- The fresh PRF output has the second-bucket budget. -/
  fresh : inputs.fB = readoutInputBound
  /-- Fresh switching is exact. -/
  freshSwitch : inputs.fBB = 0
  /-- The smudge PRF output has the second-bucket budget. -/
  smudge : inputs.sB = readoutInputBound
  /-- Smudge switching is exact. -/
  smudgeSwitch : inputs.sBB = 0
  /-- The residue budget is four; this is not a proof of a residue equation. -/
  residue : inputs.BM = 4

section StepZero

variable {backend : BackendContext} {model : HashModel} {params : W_Stage_online_0.Params}
  {context : W_Stage_online_0.generatedRoot.Ctx backend model params} {mask : ErrorPoly 8}
  {inputs : Online0Inputs context mask}

/-- Evaluate the cell and both bucket budgets independently of any execution context. -/
theorem selector_bounds : selectorCellBound 1 = 671095193600 ∧
    selectorBucketBound 1 = 1342190387200 ∧ readoutInputBound = 17592359185285120000 := by
  norm_num [selectorCellBound, selectorBucketBound, readoutInputBound]

/-- Evaluate every non-target carrier budget occurring in the step-zero decoder. -/
theorem carrier_bounds (bounds : Inputs0 inputs) :
    inputs.σBound = 6044688588822783587178439036108800 ∧
    inputs.gBound = 2475904445981812157308288629190164480000 ∧
    inputs.c1Bound = 409600 ∧ inputs.stBound = 2199044730391756800 ∧
    inputs.pubBound = 1014130461074150259633475022516291371008000000 ∧
    inputs.freshBound = 922346281253476499508428800 ∧
    inputs.smudgeBound = 922346281253476499508428800 := by
  norm_num [Online0Inputs.σBound, Online0Inputs.gBound, Online0Inputs.c1Bound,
    Online0Inputs.stBound, Online0Inputs.pubBound, Online0Inputs.freshBound,
    Online0Inputs.smudgeBound, bounds.sigma, bounds.helper, bounds.samplerHelper,
    bounds.sigmaReadout, bounds.sigmaSampler, bounds.one, bounds.state, bounds.stateHelper,
    bounds.stateSwitch, bounds.fresh, bounds.freshSwitch, bounds.smudge, bounds.smudgeSwitch,
    selectorCellBound, selectorBucketBound, readoutInputBound]

/-- Evaluate all label targets and bound every member of the existing 36-target vector. -/
theorem target_bounds (bounds : Inputs0 inputs) :
    inputs.lab425Bound = 8113043688593202077067800180405211559362560000 ∧
    inputs.lab543Bound = 8113043688593202077068025364509767478804480000 ∧
    inputs.lab626Bound = 8113043688593202077068025362310725432770560000 ∧
    inputs.lab680Bound = 8113043688593202077067800180405211559362560000 ∧
    inputs.lab734Bound = 8113043688593202077067800180405211559362560000 ∧
    inputs.sigBound = 1014130461074150259633475022516291371008000000 ∧
    (∀ part, inputs.cxBound part ≤ targetBound) := by
  obtain ⟨hsigma, hg, hc, hs, hp, _, _⟩ := carrier_bounds bounds
  obtain ⟨ha3, ha5, ha7, ha9, ha11, ha13, ha15⟩ := bounds.autHelpers
  obtain ⟨hz3, hz5, hz7, hz9, hz11, hz13, hz15⟩ := bounds.autSwitches
  norm_num [Online0Inputs.cxBound, Online0Inputs.lab425Bound, Online0Inputs.lab543Bound,
    Online0Inputs.lab626Bound, Online0Inputs.lab680Bound, Online0Inputs.lab734Bound,
    Online0Inputs.sigBound, Online0Inputs.aut3Bound, Online0Inputs.aut5Bound,
    Online0Inputs.aut7Bound, Online0Inputs.aut9Bound, Online0Inputs.aut11Bound,
    Online0Inputs.aut13Bound, Online0Inputs.aut15Bound, Fin.forall_fin_succ,
    hsigma, hg, hc, hs, hp, bounds.privateUnit, bounds.label, bounds.stateLabel,
    ha3, ha5, ha7, ha9, ha11, ha13, ha15, hz3, hz5, hz7, hz9, hz11, hz13, hz15, targetBound]

/-- Bound the complete step-zero left side uniformly over the refresh-part family. -/
theorem step0_left_le (bounds : Inputs0 inputs) (part : Fin 36) :
    2 * (54 * 8 * 1 * 22011706 +
      (25 * 8 * inputs.gBound * 2048 + 25 * 8 * inputs.c1Bound * 2048 +
        25 * 8 * inputs.cxBound part * 2048) +
      25 * 8 * ((25 * 25) * 8 * inputs.freshBound * 2048) * 2048 +
      (25 * 25) * 8 * inputs.smudgeBound * 2048 + inputs.BM) ≤ step0Left := by
  obtain ⟨_, hg, hc, _, _, hf, hs⟩ := carrier_bounds bounds
  have htarget := (target_bounds bounds).2.2.2.2.2.2 part
  rw [hg, hc, hf, hs, bounds.residue]
  norm_num [targetBound, step0Left] at htarget ⊢
  omega

/-- The maximum step-zero left side is below every CRT prime's strict decoding cutoff. -/
theorem step0_cutoff (prime : Fin 5) : step0Left < online0PrimeQuotient prime := by
  have all : ∀ prime, step0Left < online0PrimeQuotient prime := by
    norm_num [step0Left, online0PrimeQuotient, Fin.forall_fin_succ]
  exact all prime

/-- Discharge `hnoise` of `part_term0` and `claim_online0_term0` for every refresh part. -/
theorem online0_noise (bounds : Inputs0 inputs) (part : Fin 36) : inputs.Noise0 part := by
  exact lt_of_le_of_lt (step0_left_le bounds part) (step0_cutoff 0)

/-- Discharge every prime's `hnoise`, uniformly over primes and refresh parts. -/
theorem online0_all_primes_noise (inputsP : Online0InputsP context mask)
    (bounds : Inputs0 inputsP.toOnline0Inputs) (prime : Fin 5) (part : Fin 36) :
    inputsP.Noise prime part := by
  exact lt_of_le_of_lt (step0_left_le bounds part) (step0_cutoff prime)

end StepZero

/-- The step-one selector cell with a common budget for both labels and state. -/
def finalCellBound (bound : Nat) : Nat :=
  (25 + 25) * 8 * ((25 + 25) * 8 * bound * 2048 + 8 * 1 * bound) * 2048 + 8 * 1 * 0

/-- The first step-one PRF bucket, with a common imported-root budget. -/
def finalBucket0Bound (bound : Nat) : Nat := 2 * finalCellBound bound

/-- The second step-one PRF bucket, retaining the first bucket's seed budget. -/
def finalBucket1Bound (bound : Nat) : Nat :=
  2 * ((25 + 25) * 8 * ((25 + 25) * 8 * bound * 2048 +
    8 * 1 * finalBucket0Bound bound) * 2048 + 8 * 1 * 0)

/-- The four-branch state LUT budget consumed by `online1_output`. -/
def finalStateBound (bound : Nat) : Nat :=
  3 * ((25 + 25) * 8 * bound * 2048 + (25 + 25) * 8 * finalCellBound bound * 2048) +
    ((25 + 25) * 8 * bound * 2048 + (25 + 25) * 8 * finalCellBound bound * 2048) +
    4 * (8 * 1 * 0)

/-- The smudge cell budget before its final multiplication by half digits. -/
def finalSmudgeBound (bound : Nat) : Nat :=
  8 * (7 * ((25 + 25) * 8 * bound * 2048 +
      (25 + 25) * 8 * finalBucket1Bound bound * 2048) +
    ((25 + 25) * 8 * bound * 2048 +
      (25 + 25) * 8 * finalBucket1Bound bound * 2048) + 8 * (8 * 1 * 0))

/-- The next-root budget returned by `step_expand_rel`, not an initial-import budget. -/
def expandedInputBound (mask first second fresh : Nat) : Nat :=
  8 * mask * first + second + fresh

/-- Unit bounds on the three error pieces and next mask give expanded input budget ten. -/
theorem expanded_input_bound {mask first second fresh : Nat}
    (hmask : mask ≤ 1) (hfirst : first ≤ 1) (hsecond : second ≤ 1)
    (hfresh : fresh ≤ 1) : expandedInputBound mask first second fresh ≤ 10 := by
  have hproduct : mask * first ≤ 1 := by
    simpa using Nat.mul_le_mul hmask hfirst
  unfold expandedInputBound
  rw [Nat.mul_assoc]
  omega

/-- Bound precisely the fresh-plus-two-piece error witness produced by `step_expand_rel`. -/
theorem expanded_error_bound {columns mask first second fresh : Nat}
    {next : ErrorPoly 8} {error0 error1 freshError : ErrorMatrix 8 1 columns}
    (hnext : polyNorm next ≤ mask) (herror0 : CoeffBound error0 first)
    (herror1 : CoeffBound error1 second) (hfresh : CoeffBound freshError fresh) :
    CoeffBound (next • error0 + error1 + freshError)
      (expandedInputBound mask first second fresh) := by
  exact coeffBound_add
    (coeffBound_add (coeffBound_smul (by decide) hnext herror0) herror1) hfresh

/-- The unit-piece specialization of `step_expand_rel` has budget ten. -/
theorem expanded_unit_bound : 8 * 1 * 1 + 1 + 1 = (10 : Nat) := by
  norm_num

/-- Evaluate the noninitial selector budgets used with expanded step-one input bound ten. -/
theorem final_selector_bounds : finalCellBound 10 = 6710951936000 ∧
    finalBucket0Bound 10 = 13421903872000 ∧
    finalBucket1Bound 10 = 175923591852851200000 := by
  norm_num [finalCellBound, finalBucket0Bound, finalBucket1Bound]

/-- Evaluate the step-one state and smudge budgets at both supported transition budgets. -/
theorem final_bounds :
    finalStateBound 1 = 2199044730391756800 ∧
    finalSmudgeBound 1 = 922346281253476499508428800 ∧
    finalStateBound 10 = 21990447303917568000 ∧
    finalSmudgeBound 10 = 9223462812534764995084288000 := by
  norm_num [finalStateBound, finalSmudgeBound, finalCellBound,
    finalBucket0Bound, finalBucket1Bound]

/-- Fresh-only unit root and hint budgets discharge the exact final decoder inequality. -/
theorem online1_output_noise_unit :
    2 * (54 * 8 * 1 * 22011706 + 25 * 8 * finalStateBound 1 * 2048 +
      25 * 8 * finalSmudgeBound 1 * 2048 + 4) <
      1606938044254428791916078459060479968194539524630727034176641 := by
  norm_num [finalStateBound, finalSmudgeBound, finalCellBound,
    finalBucket0Bound, finalBucket1Bound]

/-- Expanded unit-piece budgets discharge the final decoder inequality, even for hint ten. -/
theorem online1_output_noise_expanded :
    2 * (54 * 8 * 10 * 22011706 + 25 * 8 * finalStateBound 10 * 2048 +
      25 * 8 * finalSmudgeBound 10 * 2048 + 4) <
      1606938044254428791916078459060479968194539524630727034176641 := by
  norm_num [finalStateBound, finalSmudgeBound, finalCellBound,
    finalBucket0Bound, finalBucket1Bound]

/-- The full expanded step-one left side is exactly the reported 112.541223-bit integer. -/
theorem final_left_value :
    2 * (54 * 8 * 10 * 22011706 + 25 * 8 * finalStateBound 10 * 2048 +
      25 * 8 * finalSmudgeBound 10 * 2048 + 4) =
      7555860754043053915342510616339848 := by
  norm_num [finalStateBound, finalSmudgeBound, finalCellBound,
    finalBucket0Bound, finalBucket1Bound]

/-- Discharge `online1_output.hnoise` for any error witnesses within the expanded budgets. -/
theorem online1_output_noise {hint state smudge residual : Nat}
    (hhint : hint ≤ 10) (hstate : state ≤ finalStateBound 10)
    (hsmudge : smudge ≤ 25 * 8 * finalSmudgeBound 10 * 2048) (hresidual : residual ≤ 4) :
    2 * (54 * 8 * hint * 22011706 + 25 * 8 * state * 2048 + smudge + residual) <
      1606938044254428791916078459060479968194539524630727034176641 := by
  have hcutoff := online1_output_noise_expanded
  norm_num [finalStateBound, finalSmudgeBound, finalCellBound,
    finalBucket0Bound, finalBucket1Bound] at hstate hsmudge hcutoff
  omega

/-- Discharge `online1_output.hnoise` with the transition's expanded, rather than unit, input. -/
theorem online1_output_noise_from_transition
    {mask first second fresh hint state smudge residual : Nat}
    (hmask : mask ≤ 1) (hfirst : first ≤ 1) (hsecond : second ≤ 1)
    (hfresh : fresh ≤ 1) (hhint : hint ≤ 10)
    (hstate : state ≤ finalStateBound (expandedInputBound mask first second fresh))
    (hsmudge : smudge ≤
      25 * 8 * finalSmudgeBound (expandedInputBound mask first second fresh) * 2048)
    (hresidual : residual ≤ 4) :
    2 * (54 * 8 * hint * 22011706 + 25 * 8 * state * 2048 + smudge + residual) <
      1606938044254428791916078459060479968194539524630727034176641 := by
  have hinput := expanded_input_bound hmask hfirst hsecond hfresh
  apply online1_output_noise hhint ?_ ?_ hresidual
  · dsimp [finalStateBound, finalCellBound] at hstate ⊢
    omega
  · dsimp [finalSmudgeBound, finalBucket1Bound, finalBucket0Bound, finalCellBound]
      at hsmudge ⊢
    omega

end Noise

end Dio2Proof
