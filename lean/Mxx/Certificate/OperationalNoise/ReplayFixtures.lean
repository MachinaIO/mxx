import Mxx.Certificate.OperationalNoise.Replay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

def keyA : MonomialKey := { centralFactors := [1], orderedFactors := [2] }
def keyB : MonomialKey := { centralFactors := [1], orderedFactors := [3] }
def factorless : MonomialKey := { centralFactors := [], orderedFactors := [] }

def plusA : Polynomial := [{ coefficient := 5, key := keyA }]
def minusA : Polynomial := [{ coefficient := -5, key := keyA }]
def survivorA : Polynomial := [{ coefficient := 7, key := keyA }]

theorem exact_cancellation : coefficient keyA (add plusA minusA) = 0 := by decide

theorem nonzero_survivor : coefficient keyA (add plusA survivorA) = 12 := by decide

theorem unequal_ordered_lists_are_distinct : keyA ≠ keyB := by decide

theorem factorless_nonzero_is_rejected :
    wellFormed [{ coefficient := 1, key := factorless }] = false := by decide

theorem factorless_zero_is_allowed :
    wellFormed [{ coefficient := 0, key := factorless }] = true := by decide

end Mxx.Certificate.OperationalNoise
