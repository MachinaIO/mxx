import LwrChainAll
import LwrModelLut

/-!
# The finalized readout of a run is its model

The readout LUT and its finalization turn the state of the model at the LUT input into the model's
finalized error.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape}

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- The finalized readout of a run: an encoding of the half-integer readout with scale `k`, whose
error is the model's finalized error, and the numerators of every ModDown of the model are
divisible by their dropped factors. -/
theorem chainFinal {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    ∃ w, Stage_online_eval.generatedRoot.body Backend.backend model tape [4]
        GeneratedClaim.stage_4_params
        (execution.«stage_2».1, execution.«stage_2».2.1, execution.«stage_2».2.2.1,
          execution.«stage_2».2.2.2.1,
          execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          external.input_0,
          execution.«stage_3».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_3».1,
          execution.«stage_3».2.2.2.2.1, execution.«stage_3».2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          execution.«stage_2».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
          ()) execution.«stage_4» w ∧
      div0 model external.input_0 external tape ∧ div1 model external.input_0 external tape ∧
      div2 model external.input_0 external tape ∧ div3 model external.input_0 external tape ∧
      div4 model external.input_0 external tape ∧ div5 model external.input_0 external tape ∧
      div6 model external.input_0 external tape ∧ div7 model external.input_0 external tape ∧
      div8 model external.input_0 external tape ∧ div9 model external.input_0 external tape ∧
      div10 model external.input_0 external tape ∧ div11 model external.input_0 external tape ∧
      div12 model external.input_0 external tape ∧ div13 model external.input_0 external tape ∧
      div14 model external.input_0 external tape ∧ div15 model external.input_0 external tape ∧
      div16 model external.input_0 external tape ∧ div17 model external.input_0 external tape ∧
      div18 model external.input_0 external tape ∧ div19 model external.input_0 external tape ∧
      div20 model external.input_0 external tape ∧ div21 model external.input_0 external tape ∧
      div22 model external.input_0 external tape ∧ div23 model external.input_0 external tape ∧
      div24 model external.input_0 external tape ∧ div25 model external.input_0 external tape ∧
      div26 model external.input_0 external tape ∧ div27 model external.input_0 external tape ∧
      div28 model external.input_0 external tape ∧ div29 model external.input_0 external tape ∧
      div30 model external.input_0 external tape ∧ div31 model external.input_0 external tape ∧
      div32 model external.input_0 external tape ∧ div33 model external.input_0 external tape ∧
      divLut model external.input_0 external tape ∧
      (∃ d : ErrorMatrix N 4 1, finalDigits = d ∧ CoeffBound d 512 ∧
        (∀ r c (i : Fin N), i.val ≠ 0 → (d r c).coeff i = 0) ∧
        ∃ G : ExactMatrix Q1 N 1 4, gadgetMatrixRuns Backend.backend 1024 4 G ∧
          G * reduceMatrix Q1 N 4 1 d = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1)) ∧
      RelE (maskM tape) ![![matrixMul w.w_803_0 w.w_806_0, matrixMul w.w_822_0 w.w_806_0],
          ![matrixMul w.w_826_0 w.w_806_0, matrixMul w.w_830_0 w.w_806_0]]
        (fun j ↦ reducePoly Q1 N ((outScale : ErrorPoly N) * lutValue (zOf external 34) *
          unitPair (payloadM tape) j) • (1 : ExactMatrix Q1 N 1 1))
        ![matrixMul w.w_836_0 w.w_806_0, matrixMul w.w_842_0 w.w_806_0]
        (finalErr model external.input_0 external tape) := by
  obtain ⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hdiv5, hdiv6, hdiv7, hdiv8, hdiv9, hdiv10, hdiv11, hdiv12, hdiv13, hdiv14, hdiv15, hdiv16, hdiv17, hdiv18, hdiv19, hdiv20, hdiv21, hdiv22, hdiv23, hdiv24, hdiv25, hdiv26, hdiv27, hdiv28, hdiv29, hdiv30, hdiv31, hdiv32, hdiv33, hk, hrel⟩ := chain hruns
  obtain ⟨hvalid, hpay, hmask, hpre, henc, -, -, hreq, -, hreqtrue⟩ := hruns
  obtain ⟨-, -, -, ha, -, -, hs0, -, hs1, -, hs2, -, hs3, -, hs4, -, hs5, -, hs6, -, hs7, -, hs8, -, hs9, -, hs10, -, hs11, -, hs12, -, hs13, -, hs14, -, hs15, -, hs16, -, hs17, -, hs18, -, hs19, -, hs20, -, hs21, -, hs22, -, hs23, -, hs24, -, hs25, -, hs26, -, hs27, -, hs28, -, hs29, -, hs30, -, hs31, -, hs32, -, hs33, -⟩ := hvalid
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hsum33⟩ := requirement_holds hreq hreqtrue
  obtain ⟨sel33, hsel33⟩ := one_hot hs33 hsum33
  have hspM := payload_model hpay
  have hsmM := mask_model hmask
  obtain ⟨g4, hg4⟩ := gadget_exists (m := 4) layout1 rfl rfl rfl
  have hg4u : centeredRebaseRuns g4 (centeredRebase g4 : ExactMatrix Q2 N 1 4) := ⟨hQ2, rfl⟩
  -- The block-33 target collapses to the LUT input target.
  have hnn : ∀ b' i, 0 ≤ selOf external b' i := by
    intro b' i
    fin_cases b'
    exacts [(hs0 i).1, (hs1 i).1, (hs2 i).1, (hs3 i).1, (hs4 i).1, (hs5 i).1, (hs6 i).1,
      (hs7 i).1, (hs8 i).1, (hs9 i).1, (hs10 i).1, (hs11 i).1, (hs12 i).1, (hs13 i).1,
      (hs14 i).1, (hs15 i).1, (hs16 i).1, (hs17 i).1, (hs18 i).1, (hs19 i).1, (hs20 i).1,
      (hs21 i).1, (hs22 i).1, (hs23 i).1, (hs24 i).1, (hs25 i).1, (hs26 i).1, (hs27 i).1,
      (hs28 i).1, (hs29 i).1, (hs30 i).1, (hs31 i).1, (hs32 i).1, (hs33 i).1]
  have hstep33 : muOf external 34 = alphaOf external ⟨coords 33 sel33, coords_lt 33 sel33⟩ * muOf external 33 :=
    muOf_step external 33 (sel := sel33) hsel33 hnn (fun k ↦ (ha k).1)
  have hin := hrel.target_eq (collapse _ _ _ (fun i ↦ alphaOf external ⟨coords 33 i, coords_lt 33 i⟩)
    (nuOf external 33) _ sel33 hsel33 (fun i ↦ rfl))
  beta_reduce at hin
  rw [← hstep33, gReb_eq hg4 hg4u] at hin
  rw [muOf] at hin
  have hhelp := encrypt_helperE henc hsmM hg4 hg4u
  have hsw := fun i ↦ (preprocess_switchE hpre hspM hg4 i).2
  obtain ⟨⟨dfin, hfd1, hfd2, hfd3, hfd4⟩, Rsum, hsum, v, hvp, hout⟩ := lutRoundE hb hg4 hg4u hin hhelp hsw
  have hfd := (⟨dfin, hfd1, hfd2, hfd3, g4, hg4, hfd4⟩ : ∃ d : ErrorMatrix N 4 1, finalDigits = d ∧
    CoeffBound d 512 ∧ (∀ r c (i : Fin N), i.val ≠ 0 → (d r c).coeff i = 0) ∧
    ∃ G : ExactMatrix Q1 N 1 4, gadgetMatrixRuns Backend.backend 1024 4 G ∧
      G * reduceMatrix Q1 N 4 1 d = (matrixPolynomial [outScale] : ExactMatrix Q1 N 1 1))
  -- The switching ciphertexts and the input key are the model's.
  have hfam : (fun k j ↦ ![![execution.«stage_2».1, execution.«stage_2».2.1],
      ![execution.«stage_2».2.2.1, execution.«stage_2».2.2.2.1]] k j) =
      lutFam model external.input_0 tape := by
    funext k j i
    have htop := congrFun (preprocess_switchTop hpre i) j
    fin_cases k
    · fin_cases j <;> exact htop
    · have h := bot_of_rhs ((preprocess_switchE hpre hspM hg4 i).2 j)
      fin_cases j
      · simp only [lutFam, lutSwitchBot, gadgetOf_eq hg4]
        rw [← htop]
        exact h
      · simp only [lutFam, lutSwitchBot, gadgetOf_eq hg4]
        rw [← htop]
        exact h
  rw [hk, hfam] at hsum hvp
  have hR : Rsum = lutRow model external.input_0 external tape := hsum.eq_encRow
  have hv : ∀ j, v j = lutV model external.input_0 external tape j := by
    intro j
    have h := fun row column ↦ hvp j row column
    rw [hR] at h
    exact div_of_identity (by norm_num) h
  have hdivLut : divLut model external.input_0 external tape := by
    intro j row column
    have h := hvp j row column
    rw [hR, hv j] at h
    exact h
  refine ⟨w, hb, hdiv0, hdiv1, hdiv2, hdiv3, hdiv4, hdiv5, hdiv6, hdiv7, hdiv8, hdiv9, hdiv10, hdiv11, hdiv12, hdiv13, hdiv14, hdiv15, hdiv16, hdiv17, hdiv18, hdiv19, hdiv20, hdiv21, hdiv22, hdiv23, hdiv24, hdiv25, hdiv26, hdiv27, hdiv28, hdiv29, hdiv30, hdiv31, hdiv32, hdiv33, hdivLut, hfd, ?_⟩
  have herr : (fun j ↦ (v j - ∑ i : Fin 128, substitution (sigma i).toNat (sigma_odd i)
      (Xe (n := N) (512 * zOf external 34)) • switchError tape [2] i j) * finalDigits) =
      finalErr model external.input_0 external tape := by
    funext j
    rw [hv j]
    rfl
  rw [← herr]
  exact hout

end LwrProof
