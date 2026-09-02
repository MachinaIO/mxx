import MxxPrimitives.Bounds
import MxxPrimitives.Matrix

namespace Mxx.Bgg

open Mxx.Primitives

variable {q n gadgetColumns secretColumns : Nat}

/- These are proof-side witnesses.  The error is an integer matrix, so a later
   theorem can consume it without applying a centered lift to an arbitrary
   exact value. -/
structure Encoding
    (ciphertext : ExactMatrix q n 1 gadgetColumns)
    (maskSecret payloadSecret : ExactMatrix q n 1 secretColumns)
    (publicMatrix gadget : ExactMatrix q n secretColumns gadgetColumns)
    (message : ExactPoly q n) where
  error : ErrorMatrix n 1 gadgetColumns
  equation :
    ciphertext =
      maskSecret * publicMatrix -
      message • (payloadSecret * gadget) +
      reduceMatrix q n 1 gadgetColumns error

structure GswCiphertext
    (ciphertext : ExactMatrix q n secretColumns gadgetColumns)
    (payloadSecret outputSecret : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (message : ExactPoly q n) where
  error : ErrorMatrix n 1 gadgetColumns
  equation :
    payloadSecret * ciphertext =
      message • (outputSecret * gadget) +
      reduceMatrix q n 1 gadgetColumns error

end Mxx.Bgg
