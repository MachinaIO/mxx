import PeDecryptBits

/-!
# The decryption stage

`decode_phase` with its inputs identified: the bits are those of the phase of `decrypt_bits`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
theorem decode_final {model : HashModel} {tape : SampleTape} {path : List Nat}
    {dctx : Stage_decode.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_9_params}
    {o : Fin N → Int} (D : Stage_decode.generatedRoot.Facts dctx o)
    {cm : ExactMatrix Q N 1 1} {key : ByteArray} {bQ : ExactMatrix PQ N 1 295}
    {p' : Fin 256 → ErrorMatrix N 295 1} {vC' : Fin 256 → ErrorMatrix N 30 1}
    {v' : Fin 11520 → ErrorMatrix N 60 1} {kF' : ErrorMatrix N 60 1}
    {hf0 hf1 : Fin 45 → ExactMatrix Q N 1 30} {c0 c1 a10 a11 : ExactMatrix Q N 1 30}
    (h0 : Stage_decode.generatedRoot.w_0_0 dctx = cm) (h1 : Stage_decode.generatedRoot.w_1_0 dctx = key)
    (h5 : Stage_decode.generatedRoot.w_5_0 dctx = bQ)
    (hanc : anchorIn dctx = fun i ↦ reduceMatrix Q N 295 1 (p' i))
    (h537 : Stage_decode.generatedRoot.w_27_0 dctx = fun i ↦ reduceMatrix Q N 30 1 (vC' i))
    (h541 : Stage_decode.generatedRoot.w_31_0 dctx = hf0)
    (h542 : Stage_decode.generatedRoot.w_32_0 dctx = hf1)
    (h546 : packedHelpers (Stage_decode.generatedRoot.w_36_0 dctx) = fun q ↦ reduceMatrix Q N 60 1 (v' q))
    (h549 : Stage_decode.generatedRoot.w_39_0 dctx = c0) (h550 : Stage_decode.generatedRoot.w_40_0 dctx = c1)
    (h552 : Stage_decode.generatedRoot.w_42_0 dctx = a10) (h553 : Stage_decode.generatedRoot.w_43_0 dctx = a11)
    (h556 : Stage_decode.generatedRoot.w_46_0 dctx = reduceMatrix Q N 60 1 kF')
    {m : Fin N → Int}
    (hbits : ∀ k : Fin N, ((2 * (((((cm - messageRow model key 1) -
      (∑ i : Fin 256, σs i (roundedAnchor model key bQ * reduceMatrix Q N 295 1 (p' i)) +
        ∑ i : Fin 256, σs i (compressedAnchor model key bQ) * reduceMatrix Q N 30 1 (vC' i) +
        ∑ q : Fin 11520, helperValue model key hf0 hf1 q * reduceMatrix Q N 60 1 (v' q) +
        (joinRow ![c0, c1] - joinRow ![a10, a11]) * reduceMatrix Q N 60 1 kF')) 0 0).coeff k).val : Int) +
      (Q : Int) / 2) / (Q : Int)) % 2 = m k) :
    o = m := by
  have hdec := decode_phase D
  simp only [] at hdec
  rw [h0, h1, h5, hanc, h537, h541, h542, h546, h549, h550, h552, h553, h556] at hdec
  funext k
  rw [hdec k]
  exact hbits k

end PeProof
