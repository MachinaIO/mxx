import PrimitivesBounds
import PrimitivesMatrix

namespace Mxx.Bgg

open Mxx.Primitives

/- A BGG+ encoding exposes one integer error witness.  The public equation is
   the complete local invariant used by application proofs. -/
structure Encoding {q n secretColumns gadgetColumns : Nat}
    (ciphertext : ExactMatrix q n 1 gadgetColumns)
    (mask payload : ExactMatrix q n 1 secretColumns)
    (publicMatrix gadget : ExactMatrix q n secretColumns gadgetColumns)
    (message : ExactPoly q n) where
  error : ErrorMatrix n 1 gadgetColumns
  equation :
    ciphertext = mask * publicMatrix - message • (payload * gadget) +
      reduceMatrix q n 1 gadgetColumns error

def EncodingErrorBound {q n secretColumns gadgetColumns : Nat}
    {ciphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {publicMatrix gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (encoding : Encoding ciphertext mask payload publicMatrix gadget message)
    (bound : Nat) : Prop :=
  CoeffBound encoding.error bound

end Mxx.Bgg
