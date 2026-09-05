import Stage_decrypt
import Backend
import MxxBgg.Multiplication
import DiamondProofParameters

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

noncomputable def gadget : ExactMatrix q n 1 ell := regularGadgetMatrix DiamondBackend.layout0

theorem generated_decomposition_reconstruct
    (params : Stage_decrypt.Params) (layer lane : Nat)
    (target : ExactMatrix q n 1 ell) (digits : ExactMatrix q n ell ell)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_67_19
      DiamondBackend.backend params layer lane target digits) :
    gadget * digits = target ∧ PreimageWithin digits D := by
  rcases hrun with ⟨value, hdecomp, hout⟩
  subst value
  rcases hdecomp with ⟨layout, hlookup, _, _, hwidth, hdigits⟩
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlookup.symm
  subst layout
  have hd : digits = regularDecomposeMatrix DiamondBackend.layout0 target := by
    simpa only [castMatrixRows] using hdigits
  subst digits
  constructor
  · exact regularGadgetMatrix_reconstruct DiamondBackend.layout0 target (by decide) (by decide)
  · exact regularDecomposeMatrix_bounded DiamondBackend.layout0 target (by decide) (by decide)

theorem generated_encrypted_product
    (params : Stage_decrypt.Params) (layer lane : Nat)
    (leftPublic rightPublic leftCiphertext rightCiphertext leftError rightError
      productTerm messageTerm output : ExactMatrix q n 1 ell)
    (leftSecret rightSecret rightPayload messageMatrix : ExactMatrix q n 1 1)
    (rightMessage : ExactPoly q n) (digits : ExactMatrix q n ell ell)
    (leftEquation : leftCiphertext = leftSecret * leftPublic -
      messageMatrix 0 0 • (rightSecret * gadget) + leftError)
    (rightEquation : rightCiphertext = rightSecret * rightPublic -
      rightMessage • (rightPayload * gadget) + rightError)
    (hdecompose : Stage_decrypt.parallel_sequential_generatedRoot_67_19
      DiamondBackend.backend params layer lane rightPublic digits)
    (hproduct : Stage_decrypt.parallel_sequential_generatedRoot_67_20
      DiamondBackend.backend params layer lane (leftCiphertext, digits, ()) productTerm)
    (hmessage : Stage_decrypt.parallel_sequential_generatedRoot_67_24
      DiamondBackend.backend params layer lane (rightCiphertext, messageMatrix, ()) messageTerm)
    (hsum : Stage_decrypt.parallel_sequential_generatedRoot_67_25
      DiamondBackend.backend params layer lane (productTerm, messageTerm, ()) output) :
    output = leftSecret * (leftPublic * digits) -
      (messageMatrix 0 0 * rightMessage) • (rightPayload * gadget) +
      (leftError * digits + messageMatrix 0 0 • rightError) ∧ PreimageWithin digits D := by
  obtain ⟨hreconstruct, hbound⟩ :=
    generated_decomposition_reconstruct params layer lane rightPublic digits hdecompose
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
