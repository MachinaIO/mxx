namespace MxxIR

/- Exact integer expression helpers used by generated parameter expressions. -/
def exactDiv (numerator denominator : Int) : Int := numerator / denominator

def roundDiv (numerator denominator : Int) : Int :=
  Int.fdiv (2 * numerator + denominator) (2 * denominator)

def log2CeilNat : Nat → Nat
  | 0 => 0
  | 1 => 0
  | value + 2 => log2CeilNat ((value + 3) / 2) + 1
termination_by value => value
decreasing_by simp_wf; omega

def log2Ceil (value : Int) : Int := Int.ofNat (log2CeilNat value.toNat)

end MxxIR
