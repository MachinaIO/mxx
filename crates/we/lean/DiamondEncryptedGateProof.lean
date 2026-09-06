import Stage_decrypt
import Backend
import BggMultiplication
import DiamondProofParameters

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

noncomputable def gadget : ExactMatrix q n 1 ell := regularGadgetMatrix DiamondBackend.layout0

theorem generated_decomposition_reconstruct
    (params : Stage_decrypt.Params)
    (target : ExactMatrix q n 1 ell) (digits : ExactMatrix q n ell ell)
    (hrun : gadgetDecomposeRuns DiamondBackend.backend params.diamond_gadget_base params.diamond_digit_count target digits) :
    gadget * digits = target ∧ PreimageWithin digits D := by
  rcases hrun with ⟨layout, hlookup, _, _, hwidth, hdigits, _⟩
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlookup.symm
  subst layout
  have hd : digits = regularDecomposeMatrix DiamondBackend.layout0 target := by
    simpa [castMatrixRows] using hdigits
  rw [hd]
  constructor
  · exact regularGadgetMatrix_reconstruct DiamondBackend.layout0 target (by decide) (by decide) (by decide)
  · exact regularDecomposeMatrix_bounded DiamondBackend.layout0 target (by decide) (by decide)

theorem generated_encrypted_product
    (params : Stage_decrypt.Params)
    (leftPublic rightPublic leftCiphertext rightCiphertext leftError rightError
      productTerm messageTerm output : ExactMatrix q n 1 ell)
    (leftSecret rightSecret rightPayload messageMatrix : ExactMatrix q n 1 1)
    (rightMessage : ExactPoly q n) (digits : ExactMatrix q n ell ell)
    (leftEquation : leftCiphertext = leftSecret * leftPublic -
      messageMatrix 0 0 • (rightSecret * gadget) + leftError)
    (rightEquation : rightCiphertext = rightSecret * rightPublic -
      rightMessage • (rightPayload * gadget) + rightError)
    (hdecompose : gadgetDecomposeRuns DiamondBackend.backend params.diamond_gadget_base params.diamond_digit_count rightPublic digits)
    (hproduct : productTerm = matrixMul leftCiphertext digits)
    (hmessage : messageTerm = matrixMulScalarRight rightCiphertext messageMatrix)
    (hsum : output = matrixAdd productTerm messageTerm) :
    output = leftSecret * (leftPublic * digits) -
      (messageMatrix 0 0 * rightMessage) • (rightPayload * gadget) +
      (leftError * digits + messageMatrix 0 0 • rightError) ∧ PreimageWithin digits D := by
  obtain ⟨hreconstruct, hbound⟩ :=
    generated_decomposition_reconstruct params rightPublic digits hdecompose
  have hp : productTerm = leftCiphertext * digits := hproduct
  have hm : messageTerm = messageMatrix 0 0 • rightCiphertext := by
    change messageTerm = matrixMulScalarRight rightCiphertext messageMatrix at hmessage
    rw [hmessage]
    funext row column
    exact mul_comm _ _
  have hs : output = productTerm + messageTerm := hsum
  rw [hp, hm] at hs
  exact ⟨hs.trans (Mxx.Bgg.multiplication_core leftEquation rightEquation hreconstruct), hbound⟩

#print axioms generated_encrypted_product

end DiamondGeneratedProof
