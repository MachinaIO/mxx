set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

structure MonomialKey where
  centralFactors : List Nat
  orderedFactors : List Nat
deriving DecidableEq, Repr

structure ExactTerm where
  coefficient : Int
  key : MonomialKey
deriving DecidableEq, Repr

abbrev Polynomial := List ExactTerm

def coefficient (key : MonomialKey) : Polynomial → Int
  | [] => 0
  | term :: terms =>
      if term.key = key then term.coefficient + coefficient key terms
      else coefficient key terms

def add (left right : Polynomial) : Polynomial := left ++ right

def negate : Polynomial → Polynomial
  | [] => []
  | term :: terms => { term with coefficient := -term.coefficient } :: negate terms

def subtract (left right : Polynomial) : Polynomial := add left (negate right)

def termWellFormed (term : ExactTerm) : Bool :=
  term.coefficient = 0 || term.key.centralFactors != [] || term.key.orderedFactors != []

def wellFormed : Polynomial → Bool
  | [] => true
  | term :: terms => termWellFormed term && wellFormed terms

theorem coefficient_add (key : MonomialKey) (left right : Polynomial) :
    coefficient key (add left right) = coefficient key left + coefficient key right := by
  induction left generalizing right with
  | nil => simp [add, coefficient]
  | cons term left ih =>
      by_cases h : term.key = key
      · simp only [add, List.cons_append, coefficient, if_pos h]
        change term.coefficient + coefficient key (add left right) =
          (term.coefficient + coefficient key left) + coefficient key right
        rw [ih right]
        exact (Int.add_assoc _ _ _).symm
      · simp only [add, List.cons_append, coefficient, if_neg h]
        exact ih right

theorem coefficient_negate (key : MonomialKey) (terms : Polynomial) :
    coefficient key (negate terms) = -coefficient key terms := by
  induction terms with
  | nil => simp [negate, coefficient]
  | cons term terms ih =>
      by_cases h : term.key = key
      · simp only [negate, coefficient, if_pos h]
        rw [ih]
        exact Int.neg_add.symm
      · simp only [negate, coefficient, if_neg h]
        exact ih

theorem coefficient_subtract (key : MonomialKey) (left right : Polynomial) :
    coefficient key (subtract left right) = coefficient key left - coefficient key right := by
  rw [subtract, coefficient_add, coefficient_negate]
  rfl

end Mxx.Certificate.OperationalNoise
