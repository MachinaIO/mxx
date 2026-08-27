import Mxx.Certificate.OperationalNoise.TallSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.Semantic000

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

def selectedEvent : Nat := 5331
def selectedOwner : Owner := ⟨.program ⟨214⟩, ⟨18907⟩⟩
def selectedDetail : String := "actual coefficient merge output monomial"
def selectedScore : Nat := 5
def selectedKey : MonomialKey Owner := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨6396⟩⟩, ⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩
theorem kernelLongMonomial (env : Env Owner) :
  evalMonomial env selectedKey = evalMonomial env selectedKey := by
  exact evalMonomial_of_key env (left := selectedKey) (right := selectedKey) (List.Perm.refl _) rfl

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.Semantic000
