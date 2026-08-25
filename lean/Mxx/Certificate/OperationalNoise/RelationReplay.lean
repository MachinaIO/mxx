import Mxx.Certificate.OperationalNoise.ContractReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.EventReplay

open Mxx.Certificate.OperationalNoise

/-- Evaluate a signed polynomial under one semantic value for each monomial key. -/
def evaluatePolynomial {Factor : Type} (valuation : MonomialKey Factor → Int) :
    Polynomial Factor → Int
  | [] => 0
  | term :: terms =>
      term.coefficient * valuation term.key + evaluatePolynomial valuation terms

theorem evaluate_scalePolynomial {Factor : Type} (valuation : MonomialKey Factor → Int)
    (scalar : Int) (polynomial : Polynomial Factor) :
    evaluatePolynomial valuation (scalePolynomial scalar polynomial) =
      scalar * evaluatePolynomial valuation polynomial := by
  induction polynomial with
  | nil => simp [scalePolynomial, evaluatePolynomial]
  | cons term terms ih =>
      simp only [scalePolynomial, List.map, evaluatePolynomial]
      change scalar * term.coefficient * valuation term.key +
          evaluatePolynomial valuation (scalePolynomial scalar terms) =
        scalar * (term.coefficient * valuation term.key +
          evaluatePolynomial valuation terms)
      rw [ih]
      simp [Int.mul_add, Int.mul_assoc]

theorem evaluate_contextualize {Factor : Type} [CentralNormalizer Factor]
    (valuation : MonomialKey Factor → Int) (context : MonomialContext Factor)
    (polynomial : Polynomial Factor) :
    evaluatePolynomial valuation (contextualize context polynomial) =
      evaluatePolynomial (fun key => valuation (context.plug key)) polynomial := by
  induction polynomial with
  | nil => rfl
  | cons term terms ih =>
      simp only [contextualize, List.map, evaluatePolynomial]
      change term.coefficient * valuation (context.plug term.key) +
          evaluatePolynomial valuation (contextualize context terms) =
        term.coefficient * valuation (context.plug term.key) +
          evaluatePolynomial (fun key => valuation (context.plug key)) terms
      rw [ih]

theorem evaluate_contextualize_factor {Factor : Type} [CentralNormalizer Factor]
    (valuation : MonomialKey Factor → Int) (context : MonomialContext Factor)
    (contextMultiplier : Int) (polynomial : Polynomial Factor)
    (contextSound : ∀ key, valuation (context.plug key) = contextMultiplier * valuation key) :
    evaluatePolynomial valuation (contextualize context polynomial) =
      contextMultiplier * evaluatePolynomial valuation polynomial := by
  rw [evaluate_contextualize]
  induction polynomial with
  | nil => simp [evaluatePolynomial]
  | cons term terms ih =>
      simp only [evaluatePolynomial]
      rw [contextSound, ih]
      simp [Int.mul_add, Int.mul_left_comm]

theorem evaluate_relationReplacement {Factor : Type} [CentralNormalizer Factor]
    (valuation : MonomialKey Factor → Int) (context : MonomialContext Factor)
    (contextMultiplier outerCoefficient : Int) (polynomial : Polynomial Factor)
    (contextSound : ∀ key, valuation (context.plug key) = contextMultiplier * valuation key) :
    evaluatePolynomial valuation (relationReplacement context outerCoefficient polynomial) =
      outerCoefficient * contextMultiplier * evaluatePolynomial valuation polynomial := by
  rw [relationReplacement, evaluate_scalePolynomial,
    evaluate_contextualize_factor valuation context contextMultiplier polynomial contextSound]
  simp [Int.mul_assoc]

/-- Context and its outer coefficient preserve a modular base relation. -/
theorem relationReplacement_modular {Factor : Type} [CentralNormalizer Factor]
    (modulus : Nat) (valuation : MonomialKey Factor → Int) (context : MonomialContext Factor)
    (contextMultiplier outerCoefficient : Int) (left right : Polynomial Factor)
    (contextSound : ∀ key, valuation (context.plug key) = contextMultiplier * valuation key)
    (baseRelation : evaluatePolynomial valuation left % Int.ofNat modulus =
      evaluatePolynomial valuation right % Int.ofNat modulus) :
    evaluatePolynomial valuation (relationReplacement context outerCoefficient left) %
        Int.ofNat modulus =
      evaluatePolynomial valuation (relationReplacement context outerCoefficient right) %
        Int.ofNat modulus := by
  rw [evaluate_relationReplacement valuation context contextMultiplier outerCoefficient left
      contextSound,
    evaluate_relationReplacement valuation context contextMultiplier outerCoefficient right
      contextSound]
  calc
    (outerCoefficient * contextMultiplier * evaluatePolynomial valuation left) %
        Int.ofNat modulus =
      ((outerCoefficient * contextMultiplier) % Int.ofNat modulus *
        (evaluatePolynomial valuation left % Int.ofNat modulus)) % Int.ofNat modulus :=
      Int.mul_emod _ _ _
    _ = ((outerCoefficient * contextMultiplier) % Int.ofNat modulus *
        (evaluatePolynomial valuation right % Int.ofNat modulus)) % Int.ofNat modulus := by
      rw [baseRelation]
    _ = (outerCoefficient * contextMultiplier * evaluatePolynomial valuation right) %
        Int.ofNat modulus := (Int.mul_emod _ _ _).symm

/-- Contextual replay is congruent for every exact base relation. Terms may merge or cancel after
    canonical context insertion; no per-key equality or collision-free premise is required. -/
theorem relationReplacement_congruent {Factor : Type} [CentralNormalizer Factor]
    (valuation : MonomialKey Factor → Int) (context : MonomialContext Factor)
    (contextMultiplier outerCoefficient : Int) (left right : Polynomial Factor)
    (contextSound : ∀ key, valuation (context.plug key) = contextMultiplier * valuation key)
    (baseRelation : evaluatePolynomial valuation left = evaluatePolynomial valuation right) :
    evaluatePolynomial valuation (relationReplacement context outerCoefficient left) =
      evaluatePolynomial valuation (relationReplacement context outerCoefficient right) := by
  rw [evaluate_relationReplacement valuation context contextMultiplier outerCoefficient left
      contextSound,
    evaluate_relationReplacement valuation context contextMultiplier outerCoefficient right
      contextSound,
    baseRelation]

end Mxx.Certificate.OperationalNoise.EventReplay
