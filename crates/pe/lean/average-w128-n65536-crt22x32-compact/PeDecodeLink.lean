import PeDecryptBits

/-!
# The decryption stage

`decode_phase` with its inputs identified: the bits are those of the phase of `decrypt_bits`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
theorem decode_final {model : HashModel} {tape : SampleTape} {path : List Nat}
    {dctx : Stage_decode.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_8_params}
    {o : Fin N → Int} (D : Stage_decode.generatedRoot.Facts dctx o)
    {cm : ExactMatrix Q N 1 1} {key : ByteArray} {bQ : ExactMatrix PQ N 1 501}
    {p' : Fin 128 → ErrorMatrix N 501 1} {vC' : Fin 128 → ErrorMatrix N 44 1}
    {v' : Fin 4736 → ErrorMatrix N 88 1} {kF' : ErrorMatrix N 88 1}
    {hf0 hf1 : Fin 37 → ExactMatrix Q N 1 44} {c0 c1 a10 a11 : ExactMatrix Q N 1 44}
    (h0 : Stage_decode.generatedRoot.w_0_0 dctx = cm) (h1 : Stage_decode.generatedRoot.w_1_0 dctx = key)
    (h5 : Stage_decode.generatedRoot.w_5_0 dctx = bQ)
    (hanc : anchorIn dctx = fun i ↦ reduceMatrix Q N 501 1 (p' i))
    (h537 : Stage_decode.generatedRoot.w_537_0 dctx = fun i ↦ reduceMatrix Q N 44 1 (vC' i))
    (h541 : Stage_decode.generatedRoot.w_541_0 dctx = hf0)
    (h542 : Stage_decode.generatedRoot.w_542_0 dctx = hf1)
    (h546 : Stage_decode.generatedRoot.w_546_0 dctx = fun q ↦ reduceMatrix Q N 88 1 (v' q))
    (h549 : Stage_decode.generatedRoot.w_549_0 dctx = c0) (h550 : Stage_decode.generatedRoot.w_550_0 dctx = c1)
    (h552 : Stage_decode.generatedRoot.w_552_0 dctx = a10) (h553 : Stage_decode.generatedRoot.w_553_0 dctx = a11)
    (h556 : Stage_decode.generatedRoot.w_556_0 dctx = reduceMatrix Q N 88 1 kF')
    {m : Fin N → Int}
    (hbits : ∀ k : Fin N, ((2 * (((((cm - messageRow model key 1) -
      (∑ i : Fin 128, σs i (roundedAnchor model key bQ * reduceMatrix Q N 501 1 (p' i)) +
        ∑ i : Fin 128, σs i (compressedAnchor model key bQ) * reduceMatrix Q N 44 1 (vC' i) +
        ∑ q : Fin 4736, helperValue model key hf0 hf1 q * reduceMatrix Q N 88 1 (v' q) +
        (joinRow ![c0, c1] - joinRow ![a10, a11]) * reduceMatrix Q N 88 1 kF')) 0 0).coeff k).val : Int) +
      (Q : Int) / 2) / (Q : Int)) % 2 = m k) :
    o = m := by
  have hdec := decode_phase D
  simp only [] at hdec
  rw [h0, h1, h5, hanc, h537, h541, h542, h546, h549, h550, h552, h553, h556] at hdec
  funext k
  rw [hdec k]
  exact hbits k

end PeProof
