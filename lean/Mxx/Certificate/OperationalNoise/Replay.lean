set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise

structure MonomialKey (Factor : Type := Nat) where
  centralFactors : List Factor
  orderedFactors : List Factor
deriving DecidableEq, Repr

structure ExactTerm (Factor : Type := Nat) where
  coefficient : Int
  key : MonomialKey Factor
deriving DecidableEq, Repr

abbrev Polynomial (Factor : Type := Nat) := List (ExactTerm Factor)

def coefficient {Factor : Type} [DecidableEq Factor]
    (key : MonomialKey Factor) : Polynomial Factor → Int
  | [] => 0
  | term :: terms =>
      if term.key = key then term.coefficient + coefficient key terms
      else coefficient key terms

def add {Factor : Type} [DecidableEq Factor]
    (left right : Polynomial Factor) : Polynomial Factor := left ++ right

def negate {Factor : Type} [DecidableEq Factor] : Polynomial Factor → Polynomial Factor
  | [] => []
  | term :: terms => { term with coefficient := -term.coefficient } :: negate terms

def subtract {Factor : Type} [DecidableEq Factor]
    (left right : Polynomial Factor) : Polynomial Factor := add left (negate right)

def ProductKey {Factor : Type} (left right output : MonomialKey Factor) : Prop :=
  output.centralFactors.Perm (left.centralFactors ++ right.centralFactors) ∧
    output.orderedFactors = left.orderedFactors ++ right.orderedFactors

def ContextKey {Factor : Type} (source replacement output : MonomialKey Factor)
    (orderedStart orderedEndExclusive : Nat) : Prop :=
  output.centralFactors.Perm (source.centralFactors ++ replacement.centralFactors) ∧
    output.orderedFactors = source.orderedFactors.take orderedStart ++
      replacement.orderedFactors ++ source.orderedFactors.drop orderedEndExclusive

def termWellFormed (term : ExactTerm) : Bool :=
  term.coefficient = 0 || term.key.centralFactors != [] || term.key.orderedFactors != []

def wellFormed : Polynomial → Bool
  | [] => true
  | term :: terms => termWellFormed term && wellFormed terms

theorem coefficient_add {Factor : Type} [DecidableEq Factor]
    (key : MonomialKey Factor) (left right : Polynomial Factor) :
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

theorem coefficient_negate {Factor : Type} [DecidableEq Factor]
    (key : MonomialKey Factor) (terms : Polynomial Factor) :
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

theorem coefficient_subtract {Factor : Type} [DecidableEq Factor]
    (key : MonomialKey Factor) (left right : Polynomial Factor) :
    coefficient key (subtract left right) = coefficient key left - coefficient key right := by
  rw [subtract, coefficient_add, coefficient_negate]
  rfl

end Mxx.Certificate.OperationalNoise
